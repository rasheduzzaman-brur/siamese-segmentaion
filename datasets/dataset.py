"""Loads the custom annotation JSON (see data/annotations/example_annotation.json)
and yields (reference, inspected, targets) samples for the Siamese model.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from pycocotools import mask as coco_mask_utils
except ImportError:  # optional, only needed if RLE segmentations are used
    coco_mask_utils = None


def _polygon_to_mask(points: List[List[float]], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 1)
    return mask


def _decode_segmentation(seg: Dict, height: int, width: int) -> np.ndarray:
    if seg["format"] == "polygon":
        return _polygon_to_mask(seg["points"], height, width)
    if seg["format"] == "rle":
        if coco_mask_utils is None:
            raise ImportError("pycocotools is required to decode RLE segmentations")
        rle = {"size": seg["size"], "counts": seg["counts"]}
        return coco_mask_utils.decode(rle).astype(np.uint8)
    raise ValueError(f"Unknown segmentation format: {seg['format']}")


class SiameseDefectDataset(Dataset):
    """One sample = one inspected image + its selected reference image.

    Multiple references per garment (configs/config.yaml ->
    data.pair_sampling.max_references_per_garment) are handled by
    randomly choosing among `garment["reference_ids"]` at __getitem__ time
    (acts as a mild augmentation and trains robustness to reference
    variation -- see DESIGN.md section 11).
    """

    def __init__(self, annotation_file: str, images_root: str, class_names: List[str],
                 transform=None, train: bool = True):
        with open(annotation_file) as f:
            ann = json.load(f)

        self.images_root = images_root
        self.class_to_id = {name: i for i, name in enumerate(class_names)}
        self.transform = transform
        self.train = train

        self.garments = {g["garment_id"]: g for g in ann["garments"]}
        self.references = {r["reference_id"]: r for r in ann["references"]}
        self.images = ann["images"]

        self.rng = np.random.default_rng(seed=0 if not train else None)

    def __len__(self) -> int:
        return len(self.images)

    def _load_image(self, rel_path: str) -> np.ndarray:
        path = os.path.join(self.images_root, rel_path)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _select_reference(self, image_record: Dict) -> Dict:
        garment = self.garments.get(image_record.get("garment_id"))
        ref_ids = garment["reference_ids"] if garment else [image_record["reference_id"]]
        if self.train and len(ref_ids) > 1:
            ref_id = ref_ids[self.rng.integers(0, len(ref_ids))]
        else:
            ref_id = image_record["reference_id"]
        return self.references[ref_id]

    def __getitem__(self, idx: int) -> Dict:
        record = self.images[idx]
        input_image = self._load_image(record["file_name"])
        ref_record = self._select_reference(record)
        reference_image = self._load_image(ref_record["file_name"])

        h, w = input_image.shape[:2]
        boxes, labels, masks = [], [], []
        for defect in record.get("defects", []):
            x, y, bw, bh = defect["bbox"]
            boxes.append([x, y, x + bw, y + bh])
            labels.append(self.class_to_id[defect["category"]])
            masks.append(_decode_segmentation(defect["segmentation"], h, w))

        if self.transform is not None:
            out = self.transform(input_image, reference_image, boxes, labels, masks)
        else:
            out = {"input_image": input_image, "reference_image": reference_image,
                   "boxes": boxes, "labels": labels, "masks": masks}

        input_t = torch.from_numpy(out["input_image"]).permute(2, 0, 1).float() / 255.0
        ref_t = torch.from_numpy(out["reference_image"]).permute(2, 0, 1).float() / 255.0

        target = {
            "boxes": torch.tensor(out["boxes"], dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(out["labels"], dtype=torch.long),
            "masks": (torch.from_numpy(np.stack(out["masks"])).float()
                      if len(out["masks"]) > 0 else torch.zeros((0, *input_t.shape[1:]))),
            "tshirt_valid": torch.tensor(float(record.get("valid_tshirt", True))),
            # positive pair (defective) iff at least one defect instance is present
            "pair_label": torch.tensor(float(len(out["boxes"]) > 0)),
            "image_id": record["image_id"],
            "garment_id": record.get("garment_id"),
        }
        return {"reference": ref_t, "input": input_t, "target": target}


def siamese_collate_fn(batch: List[Dict]) -> Dict:
    references = torch.stack([b["reference"] for b in batch])
    inputs = torch.stack([b["input"] for b in batch])
    targets = [b["target"] for b in batch]
    return {"reference": references, "input": inputs, "targets": targets}
