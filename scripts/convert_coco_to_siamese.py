"""Add reference-image pairing to an existing COCO-format annotation file
for this project's Siamese defect detector.

This keeps the COCO structure almost entirely unchanged -- "images",
"annotations", "categories" (and "info"/"licenses" if present) all stay
exactly as they are, with per-annotation image_id/category_id/bbox
untouched. The **only** addition is a `reference_image` field on each
entry in "images": the path (relative to `--reference-dir` /
`configs/config.yaml -> data.reference_dir`) of the golden/approved image
that inspected image should be compared against. See
datasets/dataset.py:SiameseDefectDataset and
data/annotations/example_annotation.json for the exact schema this
produces.

Reference-image assignment (see --reference-image / --reference-map below):
  Today every inspected image in a dataset is typically compared against a
  single reference image. This script supports that as the default, but
  is written so a later per-image or per-style reference assignment is a
  config change, not a rewrite -- see `--reference-map`.

  IMPORTANT: reference paths are resolved at training time relative to
  `configs/config.yaml -> data.reference_dir`, which is a *different*
  directory from `data.images_dir`/`file_name`. Give paths relative to
  `reference_dir` (e.g. just `ST001_BLACK_L_v1.png` if
  `reference_dir: data/reference`), not paths that already include a
  `reference/` prefix.

Usage (single reference for the whole file, today's case):

    python -m scripts.convert_coco_to_siamese \\
        --coco-json data/train/_annotations.coco.json \\
        --reference-image ST001_BLACK_L_v1.png \\
        --output data/annotations/train.json

Usage (per-image/per-style references, once you have more than one):

    python -m scripts.convert_coco_to_siamese \\
        --coco-json data/train/_annotations.coco.json \\
        --reference-map data/raw/reference_map.json \\
        --default-reference-image ST001_BLACK_L_v1.png \\
        --output data/annotations/train.json

`--reference-map` is a JSON object keyed by either the COCO image's
`file_name` (as it appears in the input file, before --file-name-prefix
is applied) or its `id` (as a string), mapping to a reference image path
(relative to `reference_dir`, see above):

    {"tshirt_001_defect_001.png": "ST001_BLACK_L_v1.png",
     "42": "ST002_WHITE_M_v1.png"}

Any COCO image not present in the map falls back to
`--default-reference-image` (or `--reference-image` if that's all you gave).

Multiple COCO export folders (e.g. Roboflow's train/valid/test, each with
its own images/ + _annotations.coco.json under one shared images_dir root):
run this script once per folder with `--file-name-prefix <folder>/` so the
output `file_name` stays resolvable from one shared `images_dir`, then use
the per-folder outputs directly as your train/val/test splits (skip
--make-splits in that case -- the split is already given).

Assembling a self-contained dataset folder (--copy-to):
  By default this script only writes the annotation json(s); the actual
  image/reference files stay wherever they already are. Pass `--copy-to
  DIR` (together with `--images-dir`/`--reference-dir`, so the files can be
  found) to also copy every referenced image and reference image into
  `DIR/images/` and `DIR/reference/` (preserving each file's relative path,
  including any --file-name-prefix), and copy the annotation json(s) this
  run wrote into `DIR/annotations/`. Run once per COCO export folder with
  the same `--copy-to DIR` to accumulate train/valid/test into one combined,
  ready-to-train dataset folder.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from typing import Dict, List, Optional


def load_json(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def build_reference_lookup(reference_map_path: Optional[str]) -> Dict[str, str]:
    """Keys may be a COCO file_name or a stringified COCO image id -- both
    are checked at lookup time, whichever matches."""
    if not reference_map_path:
        return {}
    return load_json(reference_map_path)


def resolve_reference(image_entry: Dict, reference_lookup: Dict[str, str],
                       default_reference: Optional[str]) -> str:
    file_name = image_entry["file_name"]
    image_id = str(image_entry["id"])
    if file_name in reference_lookup:
        return reference_lookup[file_name]
    if image_id in reference_lookup:
        return reference_lookup[image_id]
    if default_reference is not None:
        return default_reference
    raise ValueError(
        f"No reference image for COCO image id={image_entry['id']} "
        f"file_name={file_name!r}: not in --reference-map and no "
        f"--reference-image/--default-reference-image given."
    )


def add_reference_images(coco: Dict, reference_lookup: Dict[str, str],
                          default_reference: Optional[str], file_name_prefix: str,
                          class_map: Dict[str, str]) -> Dict:
    """Mutates and returns `coco` with a `reference_image` field added to
    every entry in "images". `annotations`/`categories` are left as-is,
    except category *names* are renamed in place if `class_map` covers them
    -- category ids, and therefore every annotation's `category_id`, are
    untouched."""
    if class_map:
        for cat in coco["categories"]:
            if cat["name"] in class_map:
                cat["name"] = class_map[cat["name"]]

    for img in coco["images"]:
        # resolve using the *original* file_name/id, before any prefix is applied
        img["reference_image"] = resolve_reference(img, reference_lookup, default_reference)
        if file_name_prefix:
            img["file_name"] = file_name_prefix + img["file_name"]

    return coco


def verify_paths(coco: Dict, images_dir: Optional[str], reference_dir: Optional[str]) -> List[str]:
    """Optional sanity check: every file_name/reference_image actually
    resolves on disk. Returns a list of problems (empty == all good)."""
    problems = []
    for img in coco["images"]:
        if images_dir is not None:
            p = os.path.join(images_dir, img["file_name"])
            if not os.path.isfile(p):
                problems.append(f"image id={img['id']}: missing image file {p}")
        if reference_dir is not None:
            rp = os.path.join(reference_dir, img["reference_image"])
            if not os.path.isfile(rp):
                problems.append(f"image id={img['id']}: missing reference file {rp}")
    return problems


def copy_dataset_files(coco: Dict, images_dir: str, reference_dir: str, copy_to: str) -> Dict:
    """Copy every referenced image and reference image into
    `copy_to/images/` and `copy_to/reference/`, preserving each file's
    relative path (so file_name/reference_image in the annotation json stay
    valid, unmodified, once `data.images_dir`/`data.reference_dir` point at
    `copy_to/images`/`copy_to/reference`). Returns counts + a list of
    problems (missing source files); everything else is copied regardless."""
    images_out_dir = os.path.join(copy_to, "images")
    reference_out_dir = os.path.join(copy_to, "reference")

    problems = []
    n_images, n_refs = 0, 0
    copied_refs = set()
    for img in coco["images"]:
        src = os.path.join(images_dir, img["file_name"])
        dst = os.path.join(images_out_dir, img["file_name"])
        if not os.path.isfile(src):
            problems.append(f"image id={img['id']}: missing source image file {src}")
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        n_images += 1

        ref_rel = img["reference_image"]
        if ref_rel in copied_refs:
            continue
        ref_src = os.path.join(reference_dir, ref_rel)
        ref_dst = os.path.join(reference_out_dir, ref_rel)
        if not os.path.isfile(ref_src):
            problems.append(f"image id={img['id']}: missing source reference file {ref_src}")
            continue
        os.makedirs(os.path.dirname(ref_dst), exist_ok=True)
        shutil.copy2(ref_src, ref_dst)
        copied_refs.add(ref_rel)
        n_refs += 1

    return {"n_images": n_images, "n_refs": n_refs, "problems": problems}


def write_splits(coco: Dict, output_prefix: str, val_ratio: float, test_ratio: float,
                  seed: int) -> List[str]:
    """Split by `reference_image` (see DESIGN.md section 5.1: never split by
    image id, since two captures of the same reference/garment are
    correlated). With only one distinct reference image this degenerates to
    a single group -- i.e. everything lands in one split -- which is
    expected until multiple references exist; a warning is printed in that
    case rather than silently producing an unusable val/test split."""
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for img in coco["images"]:
        groups[img["reference_image"]].append(img)

    ref_ids = list(groups.keys())
    if len(ref_ids) < 3:
        print(f"WARNING: only {len(ref_ids)} distinct reference image(s) in this dataset -- "
              f"splitting by reference_image is not meaningful yet (see DESIGN.md 5.1). "
              f"Writing train/val/test as random per-image splits instead, purely so the "
              f"pipeline is runnable; revisit once more reference images/styles exist.")
        all_images = list(coco["images"])
        rng = random.Random(seed)
        rng.shuffle(all_images)
        n = len(all_images)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        splits = {
            "test": all_images[:n_test],
            "val": all_images[n_test:n_test + n_val],
            "train": all_images[n_test + n_val:],
        }
    else:
        rng = random.Random(seed)
        rng.shuffle(ref_ids)
        n = len(ref_ids)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        ref_splits = {
            "test": set(ref_ids[:n_test]),
            "val": set(ref_ids[n_test:n_test + n_val]),
            "train": set(ref_ids[n_test + n_val:]),
        }
        splits = {name: [img for ref in refs for img in groups[ref]]
                  for name, refs in ref_splits.items()}

    written = []
    for name, images in splits.items():
        image_ids = {img["id"] for img in images}
        annotations = [a for a in coco["annotations"] if a["image_id"] in image_ids]
        out = {k: v for k, v in coco.items() if k not in ("images", "annotations")}
        out["images"] = images
        out["annotations"] = annotations
        path = f"{output_prefix}/{name}.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {path} ({len(images)} images, {len(annotations)} annotations)")
        written.append(path)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco-json", required=True, help="Path to the standard COCO annotations file")
    parser.add_argument("--reference-image",
                         help="Single reference image path, relative to configs/config.yaml's "
                              "data.reference_dir (NOT data.images_dir), used for every inspected "
                              "image. Shorthand for --default-reference-image when there's no "
                              "--reference-map yet.")
    parser.add_argument("--default-reference-image",
                         help="Fallback reference image for any COCO image not covered by "
                              "--reference-map.")
    parser.add_argument("--reference-map",
                         help="JSON file mapping COCO file_name or image id (string) -> "
                              "reference image path, for per-image/per-style references.")
    parser.add_argument("--class-map",
                         help="Optional JSON file mapping COCO category name -> a renamed "
                              "category name (e.g. to match configs/config.yaml -> "
                              "data.defect_classes). Category ids/annotations are untouched.")
    parser.add_argument("--file-name-prefix", default="",
                         help="Prepended to every COCO file_name in the output, e.g. 'train/' "
                              "when converting a Roboflow-style per-split export folder so the "
                              "path stays resolvable from one shared data.images_dir.")
    parser.add_argument("--output", required=True, help="Output path for the annotated COCO json")
    parser.add_argument("--images-dir", help="If given, verify every file_name resolves under this dir "
                                              "(required if --copy-to is given)")
    parser.add_argument("--reference-dir", help="If given, verify every reference_image resolves under "
                                                 "this dir (required if --copy-to is given)")
    parser.add_argument("--make-splits", action="store_true",
                         help="Also write train.json/val.json/test.json next to --output "
                              "(split by reference_image per DESIGN.md 5.1)")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-to",
                         help="Assemble a self-contained dataset folder at this path: copies every "
                              "referenced image/reference image into <copy-to>/images and "
                              "<copy-to>/reference, and copies this run's annotation json(s) into "
                              "<copy-to>/annotations. Requires --images-dir and --reference-dir. "
                              "Run once per COCO export folder with the same --copy-to to accumulate "
                              "train/valid/test into one combined dataset folder.")
    args = parser.parse_args()

    if not args.reference_image and not args.default_reference_image and not args.reference_map:
        parser.error("Give at least one of --reference-image, --default-reference-image, "
                      "or --reference-map.")
    if args.copy_to and not (args.images_dir and args.reference_dir):
        parser.error("--copy-to requires both --images-dir and --reference-dir "
                      "(so the source image/reference files can be found).")

    default_reference = args.default_reference_image or args.reference_image
    reference_lookup = build_reference_lookup(args.reference_map)
    class_map = load_json(args.class_map) if args.class_map else {}

    coco = load_json(args.coco_json)
    coco = add_reference_images(coco, reference_lookup, default_reference,
                                 args.file_name_prefix, class_map)

    if args.images_dir or args.reference_dir:
        problems = verify_paths(coco, args.images_dir, args.reference_dir)
        if problems:
            print(f"WARNING: {len(problems)} path(s) did not resolve on disk:")
            for p in problems[:20]:
                print(f"  - {p}")
            if len(problems) > 20:
                print(f"  ... and {len(problems) - 20} more")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(coco, f, indent=2)
    category_names = sorted({c["name"] for c in coco["categories"]})
    print(f"Wrote {args.output}: {len(coco['images'])} images, {len(coco['annotations'])} "
          f"annotations, categories {category_names}")
    written = [args.output]

    if args.make_splits:
        split_dir = os.path.dirname(args.output) or "."
        written += write_splits(coco, split_dir, args.val_ratio, args.test_ratio, args.seed)

    if args.copy_to:
        result = copy_dataset_files(coco, args.images_dir, args.reference_dir, args.copy_to)
        problems = result["problems"]
        if problems:
            print(f"WARNING: {len(problems)} file(s) could not be copied (missing at source):")
            for p in problems[:20]:
                print(f"  - {p}")
            if len(problems) > 20:
                print(f"  ... and {len(problems) - 20} more")

        annotations_out_dir = os.path.join(args.copy_to, "annotations")
        os.makedirs(annotations_out_dir, exist_ok=True)
        for path in written:
            dest = os.path.join(annotations_out_dir, os.path.basename(path))
            if os.path.abspath(path) != os.path.abspath(dest):
                shutil.copy2(path, dest)

        print(f"Copied dataset into {args.copy_to}: {result['n_images']} image(s), "
              f"{result['n_refs']} reference image(s), annotations under {annotations_out_dir}")


if __name__ == "__main__":
    main()
