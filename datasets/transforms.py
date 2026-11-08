"""Augmentation policy for the Siamese pair.

Key rule (see DESIGN.md section 9): augmentation must never create or erase
a defect region, and it must preserve the meaning of "reference vs.
inspected" difference. That splits every transform into one of two buckets:

  SHARED / joint (applied with identical parameters to the input image, its
  boxes AND the reference image, so the pair stays pixel-aligned):
      crop, resize/letterbox, rotation, translation, scale, perspective, flips
      -> geometric only. Never Cutout/CutMix/random-erasing on the input
         (they can remove or synthesize what looks like a defect).

  INDEPENDENT (sampled separately for reference and inspected branches):
      brightness, contrast, hue/saturation, exposure, gaussian noise, blur,
      mild JPEG compression
      -> photometric only, simulates the fact that reference and inspected
         shots are captured at different times/lighting, which the encoder
         must learn to be invariant to. A *small* independent geometric
         jitter is also applied to the reference alone, to emulate imperfect
         ORB/RANSAC alignment rather than assuming pixel-perfect registration
         (configurable: data.augment.ref_jitter).

Two geometry modes:
  - letterbox (default, data.crop unset): the whole image is letterboxed to
    data.image_size. Suitable when defects are large relative to the image.
  - crop (data.crop set): train on aligned native-resolution windows of
    data.crop.size, optionally centered on a random defect. For tiny defects
    on large images (e.g. PCB), where letterboxing would shrink them to a few
    pixels. Evaluation in this mode uses full-resolution images (no resize)
    and sliding-window tiling, see inference/tiling.py.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np

_BBOX = dict(format="pascal_voc", label_fields=["labels"])
_PAIR = {"reference": "image"}


def _letterbox(h: int, w: int) -> list:
    return [A.LongestMaxSize(max_size=max(h, w)),
            A.PadIfNeeded(h, w, border_mode=cv2.BORDER_CONSTANT, fill=0)]


def _geometric_aug(extra_flips: bool, large: bool) -> list:
    augs = [A.HorizontalFlip(p=0.5)]
    if extra_flips:  # orientation-free objects (PCBs), not garments
        augs += [A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)]
    if large:
        augs += [A.Affine(rotate=(-8, 8), translate_percent=(-0.05, 0.05),
                          scale=(0.95, 1.05), shear=(-3, 3), p=0.7,
                          border_mode=cv2.BORDER_CONSTANT, fill=0),
                 A.Perspective(scale=(0.02, 0.05), p=0.3)]
    else:
        augs += [A.Affine(rotate=(-5, 5), scale=(0.95, 1.05), p=0.3,
                          border_mode=cv2.BORDER_CONSTANT, fill=0)]
    return augs


def build_shared_geometric_transform(image_size: Tuple[int, int], train: bool,
                                     extra_flips: bool = False) -> A.Compose:
    """Letterbox mode; applied jointly to input + reference."""
    h, w = image_size
    if not train:
        return A.Compose(_letterbox(h, w), bbox_params=A.BboxParams(**_BBOX),
                         additional_targets=_PAIR)
    return A.Compose(
        _letterbox(h, w) + _geometric_aug(extra_flips, large=True),
        bbox_params=A.BboxParams(**_BBOX, min_visibility=0.3),
        additional_targets=_PAIR,
    )


def build_independent_photometric_transform(train: bool) -> A.Compose:
    if not train:
        return A.Compose([])
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
        A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=15, val_shift_limit=10, p=0.4),
        A.RandomGamma(gamma_limit=(85, 115), p=0.3),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.MotionBlur(blur_limit=5)], p=0.2),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
        A.ImageCompression(quality_range=(70, 100), p=0.2),
    ])


def build_reference_geometric_jitter(rotate: float = 3.0, translate: float = 0.02,
                                     scale: float = 0.03, p: float = 0.5) -> A.Compose:
    """Small independent geometric perturbation applied only to the
    reference branch, to teach tolerance to imperfect alignment."""
    return A.Compose([
        A.Affine(rotate=(-rotate, rotate), translate_percent=(-translate, translate),
                 scale=(1 - scale, 1 + scale), p=p, border_mode=cv2.BORDER_CONSTANT, fill=0),
    ])


class SiamesePairTransform:
    """Bundles shared geometry + independent photometric + reference jitter
    into one callable used by the dataset.

    crop: None (letterbox mode) or dict with keys
        size (int), defect_centered_prob (float), scale ([lo, hi]: resize
        factor applied to the window, so >1 zooms in).
    augment: optional dict with keys extra_flips (bool) and ref_jitter
        (dict of build_reference_geometric_jitter kwargs).
    """

    def __init__(self, image_size: Tuple[int, int], train: bool = True,
                 crop: Optional[Dict] = None, augment: Optional[Dict] = None):
        augment = augment or {}
        self.train = train
        self.crop = crop
        extra_flips = augment.get("extra_flips", False)

        if crop is None:
            self.shared_geo = build_shared_geometric_transform(image_size, train, extra_flips)
        elif train:
            size = crop["size"]
            self.shared_geo = A.Compose(
                [A.Resize(size, size)] + _geometric_aug(extra_flips, large=False),
                bbox_params=A.BboxParams(**_BBOX, min_visibility=0.5),
                additional_targets=_PAIR,
            )
        else:
            self.shared_geo = None  # full-resolution eval, tiled downstream

        self.photo_input = build_independent_photometric_transform(train)
        self.photo_ref = build_independent_photometric_transform(train)
        self.ref_geo_jitter = (build_reference_geometric_jitter(**augment.get("ref_jitter", {}))
                               if train else A.Compose([]))

    # python `random` (not a numpy Generator made here): torch reseeds it per
    # DataLoader worker, so workers don't all draw the same crops.
    def _crop_window(self, h: int, w: int, boxes: List[List[float]]) -> Tuple[int, int, int]:
        lo, hi = self.crop.get("scale", [1.0, 1.0])
        win = int(round(self.crop["size"] / random.uniform(lo, hi)))
        win = max(32, min(win, h, w))
        if boxes and random.random() < self.crop.get("defect_centered_prob", 0.5):
            x1, y1, x2, y2 = random.choice(boxes)
            # random placement that keeps the chosen box inside the window when possible
            x_lo, x_hi = x2 - win, x1
            y_lo, y_hi = y2 - win, y1
            x0 = random.uniform(min(x_lo, x_hi), max(x_lo, x_hi))
            y0 = random.uniform(min(y_lo, y_hi), max(y_lo, y_hi))
        else:
            x0 = random.uniform(0, w - win)
            y0 = random.uniform(0, h - win)
        x0 = int(np.clip(x0, 0, w - win))
        y0 = int(np.clip(y0, 0, h - win))
        return x0, y0, win

    def __call__(self, input_image: np.ndarray, reference_image: np.ndarray,
                 boxes: List[List[float]], labels: List[int]) -> Dict:
        if self.crop is not None and self.train:
            if input_image.shape[:2] != reference_image.shape[:2]:
                raise ValueError("crop mode needs input and reference of the same size "
                                 f"(aligned pairs), got {input_image.shape[:2]} vs "
                                 f"{reference_image.shape[:2]}")
            h, w = input_image.shape[:2]
            x0, y0, win = self._crop_window(h, w, boxes)
            crop = A.Compose([A.Crop(x0, y0, x0 + win, y0 + win)],
                             bbox_params=A.BboxParams(**_BBOX, min_visibility=0.5),
                             additional_targets=_PAIR)
            c = crop(image=input_image, reference=reference_image, bboxes=boxes, labels=labels)
            input_image, reference_image = c["image"], c["reference"]
            boxes, labels = list(c["bboxes"]), list(c["labels"])

        if self.shared_geo is not None:
            g = self.shared_geo(image=input_image, reference=reference_image,
                                bboxes=boxes, labels=labels)
            input_image, reference_image = g["image"], g["reference"]
            boxes, labels = g["bboxes"], g["labels"]

        input_image = self.photo_input(image=input_image)["image"]
        reference_image = self.ref_geo_jitter(image=reference_image)["image"]
        reference_image = self.photo_ref(image=reference_image)["image"]

        return {
            "input_image": input_image,
            "reference_image": reference_image,
            "boxes": [list(map(float, b)) for b in boxes],
            "labels": [int(l) for l in labels],
        }
