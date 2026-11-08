"""Project file -> train/val/test COCO jsons that datasets/dataset.py loads.

Splitting and the optional self-contained copy reuse
scripts/convert_coco_to_siamese.py (write_splits / copy_dataset_files), so
the split policy (group by reference_image, random fallback) stays in one
place. Tool-only fields are stripped; "unlabeled" images are left out;
"good" images go out with zero annotations (negative pairs); unreviewed
model proposals are dropped unless include_unreviewed is set.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import cv2
import numpy as np

from annotator import geometry
from scripts.convert_coco_to_siamese import copy_dataset_files, write_splits

IMAGE_KEYS = ("id", "file_name", "width", "height", "reference_image")
ANN_KEYS = ("id", "image_id", "category_id", "bbox", "segmentation", "area", "iscrowd")


def build_coco(project: Dict, include_unreviewed: bool = False) -> Dict:
    images = [{k: img[k] for k in IMAGE_KEYS}
              for img in project["images"] if img.get("status") in ("labeled", "good")]
    keep_ids = {img["id"] for img in images}
    good_ids = {img["id"] for img in project["images"] if img.get("status") == "good"}
    annotations = []
    for a in project["annotations"]:
        if a["image_id"] not in keep_ids or a["image_id"] in good_ids:
            continue
        if a.get("source") == "model" and not include_unreviewed:
            continue
        annotations.append({k: a[k] for k in ANN_KEYS})
    return {"info": project.get("info", {}), "categories": project["categories"],
            "images": images, "annotations": annotations}


def write_semantic_masks(coco: Dict, masks_dir: str) -> int:
    """One uint8 PNG per image, pixel = index in data.defect_classes + 1
    (0 = background). Box-only annotations don't contribute."""
    n = 0
    by_image = {}
    for a in coco["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)
    for img in coco["images"]:
        out = np.zeros((img["height"], img["width"]), dtype=np.uint8)
        # larger regions first so small ones drawn on top stay visible
        for a in sorted(by_image.get(img["id"], []), key=lambda a: -a["area"]):
            if a["segmentation"]:
                geometry.rasterize(a["segmentation"], img["width"], img["height"],
                                   value=a["category_id"], out=out)
        path = os.path.join(masks_dir, os.path.splitext(img["file_name"])[0] + ".png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, out)
        n += 1
    return n


def export(project: Dict, annotations_dir: str, val_ratio: float = 0.15, test_ratio: float = 0.15,
           seed: int = 42, include_unreviewed: bool = False, masks_dir: Optional[str] = None,
           copy_to: Optional[str] = None, images_dir: Optional[str] = None,
           reference_dir: Optional[str] = None) -> Dict:
    coco = build_coco(project, include_unreviewed)
    n_skipped_model = sum(1 for a in project["annotations"]
                          if a.get("source") == "model" and not include_unreviewed)
    os.makedirs(annotations_dir, exist_ok=True)

    all_path = os.path.join(annotations_dir, "all.json")
    with open(all_path, "w") as f:
        json.dump(coco, f, indent=2)
    written = [all_path] + write_splits(coco, annotations_dir, val_ratio, test_ratio, seed)

    result = {"files": written, "images": len(coco["images"]),
              "annotations": len(coco["annotations"]),
              "good_images": sum(1 for i in project["images"] if i.get("status") == "good"),
              "unlabeled_skipped": sum(1 for i in project["images"] if i.get("status") == "unlabeled"),
              "unreviewed_model_skipped": n_skipped_model}
    if masks_dir:
        result["semantic_masks"] = write_semantic_masks(coco, masks_dir)
        result["masks_dir"] = masks_dir
    if copy_to and os.path.abspath(os.path.join(copy_to, "images")) != os.path.abspath(images_dir):
        result["copy"] = copy_dataset_files(coco, images_dir, reference_dir, copy_to)
    return result
