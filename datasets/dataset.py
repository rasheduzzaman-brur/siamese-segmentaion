"""Loads a near-standard COCO annotation file (images/annotations/categories,
see data/annotations/example_annotation.json) and yields (reference,
inspected, targets) samples for the Siamese model.

The only departure from plain COCO is one extra field on each entry in
"images": `reference_image`, the path (relative to `reference_root`) of the
golden/approved image this inspected image should be compared against --
everything else (categories, per-annotation image_id/category_id/bbox)
is exactly the COCO detection format. See
scripts/convert_coco_to_siamese.py for how to add that field to an
existing COCO export.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List

import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class SiameseDefectDataset(Dataset):
    """One sample = one inspected image + its paired reference image."""

    def __init__(self, annotation_file: str, images_root: str, class_names: List[str],
                 transform=None, train: bool = True, reference_root: str = None):
        with open(annotation_file) as f:
            coco = json.load(f)

        self.images_root = images_root
        # reference images commonly live in a separate directory from the
        # inspected images (configs/config.yaml: data.reference_dir vs.
        # data.images_dir) -- default to images_root only for callers that
        # genuinely keep both under one directory.
        self.reference_root = reference_root if reference_root is not None else images_root
        self.class_to_id = {name: i for i, name in enumerate(class_names)}
        self.transform = transform
        self.train = train

        self.category_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
        self.images = coco["images"]

        self.annotations_by_image: Dict[int, List[Dict]] = defaultdict(list)
        for ann in coco["annotations"]:
            self.annotations_by_image[ann["image_id"]].append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def _load_image(self, root: str, rel_path: str) -> np.ndarray:
        path = os.path.join(root, rel_path)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int) -> Dict:
        record = self.images[idx]
        input_image = self._load_image(self.images_root, record["file_name"])
        reference_image = self._load_image(self.reference_root, record["reference_image"])

        boxes, labels = [], []
        for ann in self.annotations_by_image.get(record["id"], []):
            x, y, bw, bh = ann["bbox"]  # COCO xywh
            boxes.append([x, y, x + bw, y + bh])
            labels.append(self.class_to_id[self.category_id_to_name[ann["category_id"]]])

        if self.transform is not None:
            out = self.transform(input_image, reference_image, boxes, labels)
        else:
            out = {"input_image": input_image, "reference_image": reference_image,
                   "boxes": boxes, "labels": labels}

        input_t = torch.from_numpy(out["input_image"]).permute(2, 0, 1).float() / 255.0
        ref_t = torch.from_numpy(out["reference_image"]).permute(2, 0, 1).float() / 255.0

        target = {
            "boxes": torch.tensor(out["boxes"], dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(out["labels"], dtype=torch.long),
            # positive pair (defective) iff at least one defect instance is present
            "pair_label": torch.tensor(float(len(out["boxes"]) > 0)),
            "image_id": record["id"],
        }
        return {"reference": ref_t, "input": input_t, "target": target}


def siamese_collate_fn(batch: List[Dict]) -> Dict:
    references = torch.stack([b["reference"] for b in batch])
    inputs = torch.stack([b["input"] for b in batch])
    targets = [b["target"] for b in batch]
    return {"reference": references, "input": inputs, "targets": targets}
