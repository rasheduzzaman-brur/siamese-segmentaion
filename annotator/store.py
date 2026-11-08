"""Working project file: COCO + `reference_image` (the format
datasets/dataset.py loads) plus a few tool-only fields that export strips:

    images[i].status       "unlabeled" | "labeled" | "good" (reviewed, no defects)
    annotations[j].shape   "box" | "polygon" | "mask"
    annotations[j].source  "manual" | "model" (unreviewed model proposal)
    annotations[j].score   model confidence, model proposals only

Every annotation carries a class (category_id), a COCO xywh `bbox` and a
COCO polygon `segmentation` ([] for box-only detection labels; for polygon
and mask shapes the bbox is the tight extent of the segmentation, so every
segmentation label is also a detection label).
"""
from __future__ import annotations

import json
import os
import shutil
import threading
from typing import Dict, List, Optional

import cv2

from annotator import geometry

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SHAPES = {"box", "polygon", "mask"}


def list_image_files(root: str) -> List[str]:
    """Image paths under `root`, relative to it, '/'-separated, sorted."""
    out = []
    for dirpath, _, files in os.walk(root):
        for name in files:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                rel = os.path.relpath(os.path.join(dirpath, name), root)
                out.append(rel.replace(os.sep, "/"))
    return sorted(out)


def image_size(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img.shape[1], img.shape[0]


def normalize_annotation(a: Dict, width: int, height: int) -> Optional[Dict]:
    """Clip one UI/importer annotation to the image and derive its COCO
    geometry. `a` has `shape` plus `bbox` (box), `segmentation` (polygon) or
    `mask_png` / `mask` (mask: browser PNG data URL or HxW bool array).
    Returns {shape, bbox, segmentation, area}, or None if nothing is left."""
    shape = a.get("shape", "box")
    if shape not in SHAPES:
        raise ValueError(f"bad annotation shape: {shape}")
    if shape == "box":
        bbox = geometry.clip_bbox(a["bbox"], width, height)
        if bbox is None:
            return None
        return {"shape": shape, "bbox": bbox, "segmentation": [], "area": round(bbox[2] * bbox[3], 2)}
    if shape == "mask":
        mask = a["mask"] if "mask" in a else geometry.decode_mask_png(a["mask_png"])
        if mask.shape != (height, width):
            raise ValueError(f"mask is {mask.shape[::-1]}, image is {(width, height)}")
        raw = geometry.mask_to_polygons(mask)
    else:
        raw = a.get("segmentation", [])
    seg = [p for p in (geometry.clip_polygon(p, width, height) for p in raw) if p]
    if not seg:
        return None
    return {"shape": shape, "bbox": geometry.polygons_bbox(seg), "segmentation": seg,
            "area": round(sum(geometry.polygon_area(p) for p in seg), 2)}


class ProjectStore:
    def __init__(self, project_path: str, images_dir: str, reference_dir: str,
                 class_names: List[str], default_reference: Optional[str] = None):
        self.path = project_path
        self.images_dir = images_dir
        self.reference_dir = reference_dir
        self.class_names = list(class_names)
        self.lock = threading.Lock()

        self.references = list_image_files(reference_dir)
        if default_reference is None and self.references:
            default_reference = self.references[0]
        self.default_reference = default_reference

        if os.path.isfile(project_path):
            with open(project_path) as f:
                self.data = json.load(f)
        else:
            self.data = {"info": {"description": "siamese-segmentaion annotator project"},
                         "images": [], "annotations": [], "categories": []}
        # categories always mirror config data.defect_classes (ids from 1);
        # remap any existing annotations by name in case the list changed
        self._sync_categories()
        self.scan()

    # ------------------------------------------------------------------
    def _sync_categories(self):
        old = {c["id"]: c["name"] for c in self.data.get("categories", [])}
        self.data["categories"] = [{"id": i + 1, "name": n, "supercategory": "defect"}
                                   for i, n in enumerate(self.class_names)]
        name_to_new = {n: i + 1 for i, n in enumerate(self.class_names)}
        kept = []
        for ann in self.data["annotations"]:
            name = old.get(ann["category_id"])
            if name in name_to_new:
                ann["category_id"] = name_to_new[name]
                kept.append(ann)
            else:
                print(f"WARNING: dropping annotation {ann.get('id')} with class {name!r} "
                      f"not in data.defect_classes {self.class_names}")
        self.data["annotations"] = kept

    def scan(self) -> int:
        """Add images found under images_dir that the project doesn't know yet."""
        known = {img["file_name"] for img in self.data["images"]}
        next_id = max((img["id"] for img in self.data["images"]), default=-1) + 1
        added = 0
        for rel in list_image_files(self.images_dir):
            if rel in known:
                continue
            w, h = image_size(os.path.join(self.images_dir, rel))
            self.data["images"].append({"id": next_id, "file_name": rel, "width": w, "height": h,
                                        "reference_image": self.default_reference,
                                        "status": "unlabeled"})
            next_id += 1
            added += 1
        for img in self.data["images"]:
            img.setdefault("status", "unlabeled")
            if not img.get("reference_image"):
                img["reference_image"] = self.default_reference
        if added:
            self.save()
        return added

    def import_coco(self, coco_path: str) -> Dict[str, int]:
        """Merge an existing COCO(+reference_image) file, matched by file_name.
        Images already holding annotations in the project are left untouched."""
        with open(coco_path) as f:
            coco = json.load(f)
        cat_names = {c["id"]: c["name"] for c in coco.get("categories", [])}
        name_to_id = {n: i + 1 for i, n in enumerate(self.class_names)}
        by_file = {img["file_name"]: img for img in self.data["images"]}
        has_anns = {a["image_id"] for a in self.data["annotations"]}
        anns_by_src = {}
        for a in coco.get("annotations", []):
            anns_by_src.setdefault(a["image_id"], []).append(a)

        n_img = n_ann = n_skip = 0
        next_ann = self._next_ann_id()
        for src in coco.get("images", []):
            img = by_file.get(src["file_name"])
            if img is None or img["id"] in has_anns:
                n_skip += 1
                continue
            if src.get("reference_image"):
                img["reference_image"] = src["reference_image"]
            src_anns = anns_by_src.get(src["id"], [])
            for a in src_anns:
                cid = name_to_id.get(cat_names.get(a["category_id"]))
                if cid is None:
                    continue
                seg = a.get("segmentation") or []
                if not isinstance(seg, list):  # RLE -> not supported, keep the box
                    seg = []
                self.data["annotations"].append({
                    "id": next_ann, "image_id": img["id"], "category_id": cid,
                    "bbox": a["bbox"], "segmentation": seg,
                    "area": a.get("area", a["bbox"][2] * a["bbox"][3]), "iscrowd": 0,
                    "shape": "polygon" if seg else "box", "source": "manual"})
                next_ann += 1
                n_ann += 1
            img["status"] = "labeled" if src_anns else img["status"]
            n_img += 1
        self.save()
        return {"images": n_img, "annotations": n_ann, "skipped": n_skip}

    # ------------------------------------------------------------------
    def _next_ann_id(self) -> int:
        return max((a["id"] for a in self.data["annotations"]), default=0) + 1

    def get_image(self, image_id: int) -> Dict:
        for img in self.data["images"]:
            if img["id"] == image_id:
                return img
        raise KeyError(image_id)

    def annotations_for(self, image_id: int) -> List[Dict]:
        return [a for a in self.data["annotations"] if a["image_id"] == image_id]

    def summary(self) -> Dict:
        counts = {}
        for a in self.data["annotations"]:
            counts[a["image_id"]] = counts.get(a["image_id"], 0) + 1
        return {
            "classes": self.class_names,
            "references": self.references,
            "default_reference": self.default_reference,
            "images": [{"id": i["id"], "file_name": i["file_name"], "status": i["status"],
                        "reference_image": i["reference_image"],
                        "n_annotations": counts.get(i["id"], 0)} for i in self.data["images"]],
        }

    def update_image(self, image_id: int, payload: Dict) -> Dict:
        """Replace one image's annotations/status/reference from the UI.

        payload.annotations[k]: {category_id, shape, source?, score?,
            bbox? (box), segmentation? (polygon), mask_png? (mask, full-image
            PNG data URL whose alpha marks the region)}
        """
        with self.lock:
            img = self.get_image(image_id)
            W, H = img["width"], img["height"]
            valid_cats = set(range(1, len(self.class_names) + 1))
            next_id = self._next_ann_id()
            new_anns = []
            for a in payload.get("annotations", []):
                if int(a["category_id"]) not in valid_cats:
                    raise ValueError(f"bad annotation class: {a['category_id']}")
                norm = normalize_annotation(a, W, H)
                if norm is None:
                    continue
                ann = {"id": next_id, "image_id": image_id, "category_id": int(a["category_id"]),
                       **norm, "iscrowd": 0, "source": a.get("source", "manual")}
                if ann["source"] == "model" and "score" in a:
                    ann["score"] = float(a["score"])
                new_anns.append(ann)
                next_id += 1

            # an image with defects is "labeled"; without, it is either
            # explicitly reviewed as "good" (negative pair) or still "unlabeled"
            if new_anns:
                img["status"] = "labeled"
            else:
                img["status"] = "good" if payload.get("status") == "good" else "unlabeled"
            ref = payload.get("reference_image")
            if ref:
                if ref not in self.references:
                    raise ValueError(f"unknown reference image {ref!r}")
                img["reference_image"] = ref

            self.data["annotations"] = [a for a in self.data["annotations"]
                                        if a["image_id"] != image_id] + new_anns
            self.save()
            return {"image": img, "annotations": new_anns}

    def save(self):
        """Atomic write, previous version kept as <project>.bak."""
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=1)
        if os.path.isfile(self.path):
            shutil.copy2(self.path, self.path + ".bak")
        os.replace(tmp, self.path)
