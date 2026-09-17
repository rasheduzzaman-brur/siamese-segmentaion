# Siamese T-shirt Defect Inspection

Production-oriented Siamese instance segmentation + defect classification system for
automated T-shirt garment inspection. Compares an inspected garment image against an
approved reference image to detect, segment, and classify defects, with an auxiliary
unknown-defect signal and a path to OpenVINO deployment.

**Start here:** [`DESIGN.md`](DESIGN.md) — the full architecture rationale, dataset design,
loss equations, training strategy, evaluation methodology, deployment plan, comparison with
alternative architectures (Mask R-CNN / YOLO-Seg / Mask2Former / Anomalib), and the
experiment plan that should be run before trusting the Siamese design's benefit in
production. Read it before the code — the code implements exactly what it describes.

## Project layout

```
configs/config.yaml           all architecture / loss / training / export hyperparameters
data/annotations/             custom COCO-extended annotation schema + worked example
models/                       backbone, Siamese FPN encoder, feature comparison, seg/cls heads
losses/                       focal + CIoU detection loss, BCE+Dice mask loss, contrastive loss
datasets/                     dataset loader, augmentation policy, pair sampler + hard-neg mining
training/                     train / validate loops, LR scheduler
inference/                    single-image-pair prediction CLI
evaluation/                   mask mAP, per-class PRF1, FAR/FRR metrics
export/                       ONNX -> OpenVINO IR export, INT8 quantization
```

## Quickstart

```bash
pip install -r requirements.txt

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
