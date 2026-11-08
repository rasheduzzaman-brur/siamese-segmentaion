"""Convert the PKU-Market-PCB dataset (Pascal VOC xml + template boards) into
this project's Siamese COCO format (COCO + per-image `reference_image`),
split into train/val, and assemble a self-contained dataset folder.

Source layout (as distributed, e.g. Kaggle "PCB_DATASET"):

    PCB_DATASET/
      PCB_USED/01.JPG, 04.JPG, ... 12.JPG      defect-free template boards
      images/<Class>/NN_<class>_KK.jpg         defect images (NN = template id)
      Annotations/<Class>/NN_<class>_KK.xml    Pascal VOC boxes
      rotation/                                rotated copies (unannotated, unused)

Every defect image is the same size as, and pixel-aligned with, its template
`PCB_USED/NN.JPG`, so the template is used directly as `reference_image`.

Split: per image, stratified by (template, defect class), so every board and
every class appears in both train and val. The 10 templates are also added to
train as defect-free entries (input == reference, no annotations): the dataset
has no good images otherwise, and these form the negative / hard-negative pool
for datasets/pair_sampler.py.

    python -m scripts.convert_pcb_voc \\
        --src "~/Downloads/archive(1)/PCB_DATASET" --out data/pcb
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from typing import Dict, List

from scripts.convert_coco_to_siamese import copy_dataset_files, verify_paths

CLASSES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]


def find_templates(src: str) -> Dict[str, str]:
    """template id ('01') -> file name inside PCB_USED ('01.JPG')."""
    out = {}
    for p in glob.glob(os.path.join(src, "PCB_USED", "*")):
        stem, ext = os.path.splitext(os.path.basename(p))
        if ext.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            out[stem] = os.path.basename(p)
    if not out:
        raise FileNotFoundError(f"No template images under {src}/PCB_USED")
    return out


def parse_voc(src: str, templates: Dict[str, str]) -> List[Dict]:
    """One record per annotated image: file_name (relative to images/),
    template id, size, boxes (xyxy) + class names."""
    records = []
    for xml_path in sorted(glob.glob(os.path.join(src, "Annotations", "*", "*.xml"))):
        root = ET.parse(xml_path).getroot()
        class_dir = os.path.basename(os.path.dirname(xml_path))
        fname = root.findtext("filename").strip()
        template_id = fname.split("_")[0]
        if template_id not in templates:
            raise ValueError(f"{xml_path}: no template PCB_USED/{template_id}.* for {fname}")
        objects = []
        for obj in root.findall("object"):
            name = obj.findtext("name").strip().lower()
            if name not in CLASSES:
                raise ValueError(f"{xml_path}: unknown class {name!r}")
            bb = obj.find("bndbox")
            x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            objects.append({"name": name, "box": [x1, y1, x2, y2]})
        records.append({
            "file_name": f"{class_dir}/{fname}",
            "template_id": template_id,
            "width": int(root.findtext("size/width")),
            "height": int(root.findtext("size/height")),
            "class_dir": class_dir,
            "objects": objects,
        })
    if not records:
        raise FileNotFoundError(f"No VOC xml files under {src}/Annotations")
    return records


def stratified_split(records: List[Dict], val_ratio: float, seed: int):
    groups = defaultdict(list)
    for r in records:
        groups[(r["template_id"], r["class_dir"])].append(r)
    rng = random.Random(seed)
    train, val = [], []
    for key in sorted(groups):
        items = groups[key]
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_ratio)) if len(items) > 1 else 0
        val += items[:n_val]
        train += items[n_val:]
    return train, val


def to_coco(records: List[Dict], templates: Dict[str, str], template_sizes: Dict[str, tuple],
            add_templates: bool) -> Dict:
    categories = [{"id": i + 1, "name": n} for i, n in enumerate(CLASSES)]
    cat_id = {c["name"]: c["id"] for c in categories}
    images, annotations = [], []
    for r in records:
        img_id = len(images) + 1
        images.append({"id": img_id, "file_name": r["file_name"],
                       "reference_image": templates[r["template_id"]],
                       "width": r["width"], "height": r["height"]})
        for o in r["objects"]:
            x1, y1, x2, y2 = o["box"]
            annotations.append({"id": len(annotations) + 1, "image_id": img_id,
                                "category_id": cat_id[o["name"]],
                                "bbox": [x1, y1, x2 - x1, y2 - y1],
                                "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
    if add_templates:
        for tid in sorted(templates):
            w, h = template_sizes[tid]
            images.append({"id": len(images) + 1, "file_name": f"templates/{templates[tid]}",
                           "reference_image": templates[tid], "width": w, "height": h})
    return {"info": {"description": "PKU-Market-PCB in Siamese COCO format (COCO + reference_image)"},
            "categories": categories, "images": images, "annotations": annotations}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="PCB_DATASET root")
    parser.add_argument("--out", default="data/pcb", help="output dataset folder")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-template-negatives", action="store_true",
                        help="don't add the defect-free templates to train as negatives")
    args = parser.parse_args()

    src = os.path.expanduser(args.src)
    templates = find_templates(src)
    records = parse_voc(src, templates)
    template_sizes = {tid: next((r["width"], r["height"]) for r in records if r["template_id"] == tid)
                      for tid in templates if any(r["template_id"] == tid for r in records)}
    templates = {tid: f for tid, f in templates.items() if tid in template_sizes}

    train, val = stratified_split(records, args.val_ratio, args.seed)
    splits = {
        "train": to_coco(train, templates, template_sizes, not args.no_template_negatives),
        "val": to_coco(val, templates, template_sizes, False),
    }

    # templates/ under images_dir holds the templates used as defect-free *inputs*
    tmpl_img_dir = os.path.join(args.out, "images", "templates")
    os.makedirs(tmpl_img_dir, exist_ok=True)
    for f in templates.values():
        shutil.copy2(os.path.join(src, "PCB_USED", f), os.path.join(tmpl_img_dir, f))

    os.makedirs(os.path.join(args.out, "annotations"), exist_ok=True)
    for name, coco in splits.items():
        defect_only = dict(coco, images=[i for i in coco["images"]
                                          if not i["file_name"].startswith("templates/")])
        res = copy_dataset_files(defect_only, os.path.join(src, "images"),
                                 os.path.join(src, "PCB_USED"), args.out)
        problems = res["problems"] + verify_paths(coco, os.path.join(args.out, "images"),
                                                  os.path.join(args.out, "reference"))
        if problems:
            raise RuntimeError("\n".join(problems[:20]))
        path = os.path.join(args.out, "annotations", f"{name}.json")
        with open(path, "w") as f:
            json.dump(coco, f, indent=1)
        per_class = Counter(CLASSES[a["category_id"] - 1] for a in coco["annotations"])
        n_neg = sum(i["file_name"].startswith("templates/") for i in coco["images"])
        print(f"{path}: {len(coco['images'])} images ({n_neg} template negatives), "
              f"{len(coco['annotations'])} boxes, per class {dict(sorted(per_class.items()))}")


if __name__ == "__main__":
    main()
