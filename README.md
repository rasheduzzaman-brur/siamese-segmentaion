# Siamese T-shirt Defect Inspection

Production-oriented Siamese object detection + defect classification system for
automated T-shirt garment inspection. Compares an inspected garment image against an
approved reference image to detect and classify defects with bounding boxes, with an
auxiliary unknown-defect signal and a path to OpenVINO deployment.

**Note:** this codebase is detection-only — there is no instance-segmentation (mask)
branch. `DESIGN.md` was originally written for a Siamese instance-segmentation variant
(with a prototype-mask branch and BCE+Dice mask loss); the mask branch/loss described
there has since been removed from the code. Everything else in `DESIGN.md` (backbone,
Siamese FPN, feature comparison, FCOS-style detection head, training strategy,
evaluation, deployment) still applies.

**Start here:** [`DESIGN.md`](DESIGN.md) — the architecture rationale, dataset design,
loss equations, training strategy, evaluation methodology, deployment plan, and
comparison with alternative architectures. Read it before the code, keeping the note
above in mind for the segmentation-specific sections.

## Project layout

```
configs/config.yaml           all architecture / loss / training / export hyperparameters
scripts/                      adds reference_image pairing to a COCO annotation file
data/annotations/             near-COCO annotation schema (COCO + reference_image) + worked example
models/                       backbone, Siamese FPN encoder, feature comparison, detection/cls head
losses/                       focal + CIoU detection loss, contrastive loss
datasets/                     dataset loader, augmentation policy, pair sampler + hard-neg mining
training/                     train / validate loops, LR scheduler
inference/                    single-image-pair prediction CLI
evaluation/                   box mAP, per-class PRF1, FAR/FRR metrics
export/                       ONNX -> OpenVINO IR export, INT8 quantization
```

## Quickstart

```bash
pip install -r requirements.txt

# add reference_image pairing to your existing COCO-format annotations and
# assemble a self-contained dataset folder (images + reference + annotations
# copied together) -- run once per split/export folder, --copy-to
# accumulates into the same destination (see scripts/convert_coco_to_siamese.py
# --help for per-image/per-style reference mapping once you have more than
# one reference image)
python -m scripts.convert_coco_to_siamese \
    --coco-json data/train/_annotations.coco.json \
    --reference-image ST001_BLACK_L_v1.png \
    --file-name-prefix train/ \
    --output data/prepared/annotations/train.json \
    --images-dir data --reference-dir data/reference \
    --copy-to data/prepared

python -m scripts.convert_coco_to_siamese \
    --coco-json data/valid/_annotations.coco.json \
    --reference-image ST001_BLACK_L_v1.png \
    --file-name-prefix valid/ \
    --output data/prepared/annotations/val.json \
    --images-dir data --reference-dir data/reference \
    --copy-to data/prepared

# train
python -m training.train --config configs/config.yaml

# run inference on one reference/inspected pair
python -m inference.predict --config configs/config.yaml --weights runs/.../best.pt \
    --reference data/reference/ST001_BLACK_L_v1.png --input data/images/tshirt_001_defect_001.png

# export to OpenVINO
python -m export.export_openvino --config configs/config.yaml --weights runs/.../best.pt \
    --out export/artifacts/siamese_defect --precision FP16
```

## Status

This is a from-scratch design + reference implementation (no prior codebase existed).
All Python files are syntax-checked (`python -m py_compile`), but this environment has no
PyTorch installed, so nothing has been executed against real tensors/data yet. Before
training on real data, run the smoke test described in `DESIGN.md` section 17.
