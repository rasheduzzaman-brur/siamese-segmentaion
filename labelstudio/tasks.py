"""Build Label Studio tasks for Siamese pairs.

One task per inspected image:

    data.image            inspected image URL      (shown on the right, labelled)
    data.reference        reference image URL      (shown on the left)
    data.file_name        path relative to images_dir    } kept so the export
    data.reference_image  path relative to reference_dir } maps back exactly

URLs use Label Studio's local file serving (/data/local-files/?d=<path
relative to LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT>), see labelstudio/start.sh.

Existing labels (a COCO + reference_image file, or the annotator's
project.json) can be attached as editable pre-annotations ("predictions",
default) or as finished annotations.

    python -m labelstudio.tasks --document-root data --output data/annotations/ls_tasks.json \\
        [--import data/prepared/annotations/train.json ...] [--reference-map map.json]
"""
from __future__ import annotations

import argparse
import json
import os
import uuid
from typing import Dict, List, Optional
from urllib.parse import quote

import yaml

from annotator.store import image_size, list_image_files
from labelstudio.config import INSPECTED, PAIR
from scripts.convert_coco_to_siamese import build_reference_lookup, resolve_reference


def local_file_url(path: str, document_root: str) -> str:
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(document_root))
    if rel.startswith(".."):
        raise ValueError(f"{path} is outside the document root {document_root}")
    return "/data/local-files/?d=" + quote(rel.replace(os.sep, "/"))


def _rid() -> str:
    return uuid.uuid4().hex[:10]


def coco_to_results(anns: List[Dict], cat_names: Dict[int, str], classes: List[str],
                    width: int, height: int) -> List[Dict]:
    """COCO annotations of one image -> Label Studio result items."""
    results = []
    for a in anns:
        name = cat_names.get(a["category_id"])
        if name not in classes:
            continue
        common = {"to_name": INSPECTED, "original_width": width, "original_height": height,
                  "image_rotation": 0}
        seg = a.get("segmentation")
        if isinstance(seg, list) and seg:
            for poly in seg:
                pts = [[poly[i] * 100.0 / width, poly[i + 1] * 100.0 / height] for i in range(0, len(poly), 2)]
                results.append({"id": _rid(), "from_name": "poly", "type": "polygonlabels", **common,
                                "value": {"points": pts, "closed": True, "polygonlabels": [name]}})
        else:
            x, y, w, h = a["bbox"]
            results.append({"id": _rid(), "from_name": "box", "type": "rectanglelabels", **common,
                            "value": {"x": x * 100.0 / width, "y": y * 100.0 / height,
                                      "width": w * 100.0 / width, "height": h * 100.0 / height,
                                      "rotation": 0, "rectanglelabels": [name]}})
    return results


def pair_choice(value: str) -> Dict:
    return {"id": _rid(), "from_name": PAIR, "to_name": INSPECTED, "type": "choices",
            "value": {"choices": [value]}}


def make_task(file_name: str, reference_image: str, images_dir: str, reference_dir: str,
              document_root: str) -> Dict:
    return {"data": {
        "image": local_file_url(os.path.join(images_dir, file_name), document_root),
        "reference": local_file_url(os.path.join(reference_dir, reference_image), document_root),
        "file_name": file_name, "reference_image": reference_image,
        "image_name": file_name,  # display only (template header)
    }}


def build_tasks(images_dir: str, reference_dir: str, document_root: str, classes: List[str],
                default_reference: Optional[str] = None, reference_map: Optional[str] = None,
                imports: List[str] = (), as_annotations: bool = False) -> List[Dict]:
    lookup = build_reference_lookup(reference_map)
    if default_reference is None and not lookup:
        refs = list_image_files(reference_dir)
        if not refs:
            raise ValueError(f"no reference images found in {reference_dir}")
        default_reference = refs[0]

    # existing labels, by file_name: (image entry, its annotations, category names)
    known: Dict[str, tuple] = {}
    for path in imports:
        with open(path) as f:
            coco = json.load(f)
        cat_names = {c["id"]: c["name"] for c in coco.get("categories", [])}
        by_image: Dict[int, List[Dict]] = {}
        for a in coco.get("annotations", []):
            by_image.setdefault(a["image_id"], []).append(a)
        for img in coco.get("images", []):
            known[img["file_name"]] = (img, by_image.get(img["id"], []), cat_names)

    tasks = []
    for idx, rel in enumerate(list_image_files(images_dir)):
        img, anns, cat_names = known.get(rel, (None, [], {}))
        ref = (img or {}).get("reference_image") or resolve_reference(
            {"file_name": rel, "id": idx}, lookup, default_reference)
        task = make_task(rel, ref, images_dir, reference_dir, document_root)
        if img is not None:
            w, h = img.get("width"), img.get("height")
            if not (w and h):
                w, h = image_size(os.path.join(images_dir, rel))
            result = coco_to_results(anns, cat_names, classes, w, h)
            status = img.get("status")  # annotator project.json
            if status == "unlabeled":
                result = None
            elif not result:
                # listed with zero defects = negative pair, exactly as datasets/dataset.py reads it
                result = [pair_choice("good")]
            else:
                result.append(pair_choice("defect"))
            if result is not None:
                key = "annotations" if as_annotations else "predictions"
                entry = {"result": result}
                if not as_annotations:
                    entry["model_version"] = "imported"
                task[key] = [entry]
        tasks.append(task)
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--images", help="inspected images root (default: data.images_dir)")
    parser.add_argument("--references", help="reference images root (default: data.reference_dir)")
    parser.add_argument("--document-root", default="data",
                        help="LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT (must contain both image dirs)")
    parser.add_argument("--default-reference", help="reference_image for every image (default: first file)")
    parser.add_argument("--reference-map", help="json {file_name: reference_image}")
    parser.add_argument("--import", dest="imports", action="append", default=[],
                        help="existing COCO(+reference_image) json or annotator project.json (repeatable)")
    parser.add_argument("--as-annotations", action="store_true",
                        help="import existing labels as finished annotations instead of pre-annotations")
    parser.add_argument("--output", default="data/annotations/ls_tasks.json")
    args = parser.parse_args()

    with open(args.config) as f:
        data = yaml.safe_load(f)["data"]
    tasks = build_tasks(args.images or data["images_dir"], args.references or data["reference_dir"],
                        args.document_root, data["defect_classes"], args.default_reference,
                        args.reference_map, args.imports, args.as_annotations)
    with open(args.output, "w") as f:
        json.dump(tasks, f, indent=1)
    n_pre = sum(1 for t in tasks if "predictions" in t or "annotations" in t)
    print(f"wrote {len(tasks)} tasks ({n_pre} with existing labels) to {args.output}")


if __name__ == "__main__":
    main()
