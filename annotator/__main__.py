"""python -m annotator [--config configs/config.yaml] [--images DIR]
    [--references DIR] [--project FILE] [--import COCO.json ...]
    [--weights CKPT] [--host 127.0.0.1] [--port 8000]

Defaults come from configs/config.yaml: images/reference dirs from
data.images_dir / data.reference_dir, classes from data.defect_classes,
export target = the directory of data.train_split.
"""
from __future__ import annotations

import argparse
import os

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--images", help="inspected images root (default: data.images_dir)")
    parser.add_argument("--references", help="reference images root (default: data.reference_dir)")
    parser.add_argument("--project", default="data/annotations/project.json",
                        help="working annotation file (created if missing)")
    parser.add_argument("--default-reference",
                        help="reference_image given to newly found images (default: first file)")
    parser.add_argument("--import", dest="imports", action="append", default=[],
                        help="merge an existing COCO(+reference_image) json, e.g. "
                             "data/prepared/annotations/train.json (repeatable)")
    parser.add_argument("--weights", help="trained checkpoint, enables model-assisted labelling")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data = cfg["data"]

    from annotator.server import create_app
    from annotator.store import ProjectStore

    store = ProjectStore(args.project, args.images or data["images_dir"],
                         args.references or data["reference_dir"], data["defect_classes"],
                         default_reference=args.default_reference)
    for path in args.imports:
        print(f"import {path}: {store.import_coco(path)}")

    annotations_dir = os.path.dirname(data["train_split"])
    export_defaults = {"annotations_dir": annotations_dir,
                       "masks_dir": os.path.join(os.path.dirname(annotations_dir.rstrip("/")), "masks"),
                       "val_ratio": 0.15, "test_ratio": 0.15, "seed": cfg["experiment"]["seed"],
                       "semantic_masks": False, "include_unreviewed": False, "copy_to": ""}

    assistant = None
    if args.weights:
        from annotator.assist import ModelAssistant
        assistant = ModelAssistant(cfg, args.weights)

    import uvicorn
    print(f"{len(store.data['images'])} images, {len(store.references)} reference image(s), "
          f"classes {data['defect_classes']}\nopen http://{args.host}:{args.port}")
    uvicorn.run(create_app(store, export_defaults, assistant), host=args.host, port=args.port,
                log_level="warning")


if __name__ == "__main__":
    main()
