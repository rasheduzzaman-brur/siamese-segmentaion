"""Augmentation policy for the Siamese pair.

Key rule (see DESIGN.md section 9): augmentation must never create or erase
a defect region, and it must preserve the meaning of "reference vs.
inspected" difference. That splits every transform into one of two buckets:

  SHARED / joint (applied to input image + its boxes together, so
  labels stay pixel-aligned):
      rotation, translation, scale, perspective, horizontal flip
      -> geometric only. Never Cutout/CutMix/random-erasing on the input
         (they can remove or synthesize what looks like a defect).

  INDEPENDENT (sampled separately for reference and inspected branches):
      brightness, contrast, hue/saturation, exposure, gaussian noise, blur,
      mild JPEG compression
      -> photometric only, simulates the fact that reference and inspected
         shots are captured at different times/lighting, which the encoder
         must learn to be invariant to. A *small* independent geometric
         jitter (+-3 deg rotation, +-3% scale/translate) is also applied to
         the reference alone, to emulate imperfect ORB/RANSAC alignment
         rather than assuming pixel-perfect registration.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np


def build_shared_geometric_transform(image_size: Tuple[int, int], train: bool) -> A.Compose:
    h, w = image_size
    if not train:
        return A.Compose(
            [A.LongestMaxSize(max_size=max(h, w)),
             A.PadIfNeeded(h, w, border_mode=cv2.BORDER_CONSTANT, value=0)],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )
    return A.Compose(
        [
            A.LongestMaxSize(max_size=max(h, w)),
            A.PadIfNeeded(h, w, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.HorizontalFlip(p=0.5),
            A.Affine(rotate=(-8, 8), translate_percent=(-0.05, 0.05),
                     scale=(0.95, 1.05), shear=(-3, 3), p=0.7,
                     mode=cv2.BORDER_CONSTANT, cval=0),
            A.Perspective(scale=(0.02, 0.05), p=0.3),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.3),
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


def build_reference_geometric_jitter() -> A.Compose:
    """Small independent geometric perturbation applied only to the
    reference branch, to teach tolerance to imperfect alignment."""
    return A.Compose([
        A.Affine(rotate=(-3, 3), translate_percent=(-0.02, 0.02),
                  scale=(0.97, 1.03), p=0.5, mode=cv2.BORDER_CONSTANT, cval=0),
    ])


class SiamesePairTransform:
    """Bundles the three pipelines above into one callable used by the dataset."""

    def __init__(self, image_size: Tuple[int, int], train: bool = True):
        self.shared_geo = build_shared_geometric_transform(image_size, train)
        self.photo_input = build_independent_photometric_transform(train)
        self.photo_ref = build_independent_photometric_transform(train)
        self.ref_geo_jitter = build_reference_geometric_jitter() if train else A.Compose([])

    def __call__(self, input_image: np.ndarray, reference_image: np.ndarray,
                 boxes: List[List[float]], labels: List[int]) -> Dict:
        geo_out = self.shared_geo(
            image=input_image,
            bboxes=boxes, labels=labels,
        )
        input_image_t = self.photo_input(image=geo_out["image"])["image"]

        ref_out = self.ref_geo_jitter(image=reference_image)
        reference_image_t = self.photo_ref(image=ref_out["image"])["image"]

        return {
            "input_image": input_image_t,
            "reference_image": reference_image_t,
            "boxes": geo_out["bboxes"],
            "labels": geo_out["labels"],
        }
