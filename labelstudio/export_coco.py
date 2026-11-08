"""Label Studio JSON export -> COCO + reference_image splits for training.

Per task the latest non-cancelled annotation is used:
  regions present        -> defect image ("labeled"), one COCO annotation per region
  no regions             -> "good" image (negative pair), unless `pair` = defect
  no annotation          -> not labelled yet, left out

Regions: rectanglelabels -> box (bbox only, detection); polygonlabels and
brushlabels -> COCO polygon `segmentation` with bbox = its extent. The split /
semantic-mask writing is annotator/export.py (same output as the annotator's
Export button), so --project-out can also hand the result to the annotator.

    python -m labelstudio.export_coco --ls-json export.json [--out data/prepared/annotations]
    python -m labelstudio.export_coco --url http://localhost:8080 --token TOKEN --project 1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import yaml

from annotator.export import export
from annotator.store import image_size, normalize_annotation
from labelstudio import rle
from labelstudio.config import PAIR, TEMPLATE_STATE


def _rect_bbox(v: Dict, W: int, H: int) -> List[float]:
    x, y = v["x"] * W / 100.0, v["y"] * H / 100.0
    w, h = v["width"] * W / 100.0, v["height"] * H / 100.0
    r = math.radians(v.get("rotation", 0) or 0)
    if not r:
        return [x, y, w, h]
    # Label Studio rotates clockwise around the top-left corner
    c, s = math.cos(r), math.sin(r)
    xs = [x + dx * c - dy * s for dx, dy in ((0, 0), (w, 0), (w, h), (0, h))]
    ys = [y + dx * s + dy * c for dx, dy in ((0, 0), (w, 0), (w, h), (0, h))]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def region_to_annotation(r: Dict, classes: List[str], W: int, H: int) -> Optional[Dict]:
    """One Label Studio result item -> annotator-style {category_id, shape, ...}."""
    t, v = r.get("type"), r.get("value", {})
    labels = v.get(t) or v.get("labels") or []
    if t not in ("rectanglelabels", "polygonlabels", "brushlabels"):
        return None
    if not labels or labels[0] not in classes:
        print(f"WARNING: skipping region with unknown label {labels!r}")
        return None
    out = {"category_id": classes.index(labels[0]) + 1}
    if t == "rectanglelabels":
        out.update(shape="box", bbox=_rect_bbox(v, W, H))
    elif t == "polygonlabels":
        out.update(shape="polygon", segmentation=[[c for x, y in v["points"]
                                                   for c in (x * W / 100.0, y * H / 100.0)]])
    else:
        out.update(shape="mask", mask=rle.decode(v["rle"], W, H))
    return out


def _resolve_files(task: Dict, images_dir: str, document_root: Optional[str],
                   ls_data_dir: Optional[str], reference_image: Optional[str]) -> Tuple[str, str]:
    """-> (file_name relative to images_dir, reference_image) for one task."""
    data = task["data"]
    ref = data.get("reference_image") or reference_image
    if not ref:
        raise ValueError(f"task {task.get('id')}: no data.reference_image; pass --reference-image "
                         f"or generate the interface with `labelstudio.config --template`")
    if data.get("file_name"):
        return data["file_name"], ref
    url = data["image"]
    path = urlparse(url).path
    if path.startswith("/data/upload/"):
        # uploaded with Label Studio's Import button: the file lives in Label
        # Studio's media dir -> copy it next to the other inspected images
        if not ls_data_dir:
            raise ValueError("uploaded tasks need --ls-data-dir (Label Studio's base data dir)")
        rel_upload = unquote(path[len("/data/upload/"):])
        src = os.path.join(ls_data_dir, "media", "upload", rel_upload)
        rel = "ls_upload/" + rel_upload
        dst = os.path.join(images_dir, rel)
        if not os.path.isfile(dst):
            if not os.path.isfile(src):
                raise ValueError(f"task {task.get('id')}: uploaded file not found at {src}")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        return rel, ref
    q = parse_qs(urlparse(url).query).get("d")
    if q and document_root:
        local = os.path.join(document_root, unquote(q[0]))
        return os.path.relpath(local, images_dir).replace(os.sep, "/"), ref
    raise ValueError(f"task {task.get('id')}: cannot map image {url!r} to a file")


def _latest_annotation(task: Dict) -> Optional[Dict]:
    anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    if not anns:
        return None
    return max(anns, key=lambda a: (a.get("updated_at") or "", a.get("id") or 0))


def ls_to_project(ls_tasks: List[Dict], classes: List[str], images_dir: str,
                  document_root: Optional[str] = None, ls_data_dir: Optional[str] = None,
                  reference_image: Optional[str] = None) -> Tuple[Dict, Dict]:
    """-> (annotator project dict, stats)."""
    images, annotations = [], []
    stats = {"tasks": len(ls_tasks), "labeled": 0, "good": 0, "unlabeled": 0, "regions": 0,
             "skipped_regions": 0}
    for img_id, task in enumerate(ls_tasks):
        file_name, ref = _resolve_files(task, images_dir, document_root, ls_data_dir, reference_image)
        ann = _latest_annotation(task)
        result = ann["result"] if ann else []
        sized = [r for r in result if r.get("original_width")]
        if sized:
            W, H = sized[0]["original_width"], sized[0]["original_height"]
        else:
            W, H = image_size(os.path.join(images_dir, file_name))
        pair = next((r["value"]["choices"][0] for r in result
                     if r.get("from_name") == PAIR and r.get("value", {}).get("choices")), None)

        anns = []
        for r in result:
            a = region_to_annotation(r, classes, W, H)
            if a is None:
                continue
            norm = normalize_annotation(a, W, H)
            if norm is None:
                stats["skipped_regions"] += 1
                continue
            anns.append({"id": len(annotations) + len(anns) + 1, "image_id": img_id,
                         "category_id": a["category_id"], **norm, "iscrowd": 0, "source": "manual"})

        if ann is None:
            status = "unlabeled"
        elif anns:
            status = "labeled"
            if pair == "good":
                print(f"WARNING: {file_name}: marked good but has {len(anns)} region(s); exported as defect")
        elif pair == "defect":
            print(f"WARNING: {file_name}: marked defect but has no regions; left out")
            status = "unlabeled"
        else:
            status = "good"
        stats[status] += 1
        stats["regions"] += len(anns)
        images.append({"id": img_id, "file_name": file_name, "width": W, "height": H,
                       "reference_image": ref, "status": status})
        annotations.extend(anns)
    return {"images": images, "annotations": annotations,
            "categories": [{"id": i + 1, "name": n, "supercategory": "defect"}
                           for i, n in enumerate(classes)]}, stats


def fetch_export(url: str, token: str, project: int) -> List[Dict]:
    from labelstudio.setup_project import LabelStudioClient
    return LabelStudioClient(url, token).get(f"/api/projects/{project}/export?exportType=JSON&download_all_tasks=true")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--ls-json", help="file from Label Studio Export -> JSON")
    src.add_argument("--project", type=int, help="fetch the export of this project id over the API")
    parser.add_argument("--url", default=os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080"))
    parser.add_argument("--token", default=os.environ.get("LABEL_STUDIO_API_KEY"))
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--images", help="inspected images root (default: data.images_dir)")
    parser.add_argument("--references", help="reference images root (default: data.reference_dir)")
    parser.add_argument("--document-root", default="data")
    parser.add_argument("--ls-data-dir", default=os.environ.get("LABEL_STUDIO_BASE_DATA_DIR", "data/label-studio"),
                        help="Label Studio base data dir, for images uploaded through its Import button")
    parser.add_argument("--reference-image",
                        help="template paired with tasks that don't name one (default: the one "
                             f"recorded by `labelstudio.config --template` in {TEMPLATE_STATE})")
    parser.add_argument("--out", help="annotations dir (default: directory of data.train_split)")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--semantic-masks", help="also write per-image class-index mask PNGs here")
    parser.add_argument("--copy-to", help="also copy images + references here (see convert_coco_to_siamese)")
    parser.add_argument("--project-out", help="also write an annotator project.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data = cfg["data"]
    images_dir = args.images or data["images_dir"]
    if args.ls_json:
        with open(args.ls_json) as f:
            ls_tasks = json.load(f)
    else:
        if not args.token:
            parser.error("--token (or LABEL_STUDIO_API_KEY) is required with --project")
        ls_tasks = fetch_export(args.url, args.token, args.project)

    reference_image = args.reference_image
    if reference_image is None and os.path.isfile(TEMPLATE_STATE):
        with open(TEMPLATE_STATE) as f:
            reference_image = json.load(f).get("reference_image")
    project, stats = ls_to_project(ls_tasks, data["defect_classes"], images_dir, args.document_root,
                                   args.ls_data_dir, reference_image)
    print("label studio:", stats)
    if args.project_out:
        with open(args.project_out, "w") as f:
            json.dump(project, f, indent=1)
    res = export(project, args.out or os.path.dirname(data["train_split"]),
                 val_ratio=args.val_ratio, test_ratio=args.test_ratio,
                 seed=cfg["experiment"]["seed"] if args.seed is None else args.seed,
                 masks_dir=args.semantic_masks, copy_to=args.copy_to,
                 images_dir=images_dir, reference_dir=args.references or data["reference_dir"])
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
