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
annotator/                    local web app for labelling boxes / polygons / brush masks per class
labelstudio/                  Label Studio template, upload page, task import, COCO export, ML backend
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

## PCB dataset (PKU-Market-PCB)

`scripts/convert_pcb_voc.py` converts the PCB_DATASET release (Pascal VOC xml
plus defect-free template boards in `PCB_USED/`) into this repo's format,
COCO plus `reference_image`:

```bash
python -m scripts.convert_pcb_voc --src "~/Downloads/archive(1)/PCB_DATASET" --out data/pcb
python -m training.train --config configs/pcb.yaml
```

- The reference for `NN_<class>_KK.jpg` is template `NN.JPG`. Each image is
  the same size as its template and pixel-aligned with it.
- Train/val is an 80/20 split per image, stratified by (board, class).
  Use `--val-ratio` and `--seed` to change it.
- The 10 templates are added to train as defect-free (negative) images.
  Use `--no-template-negatives` to skip them.
- The `rotation/` folder is not used, because it has no annotations.
- Defects are tiny (median ~64 px on ~3000×2000 boards). `configs/pcb.yaml`
  therefore trains on aligned 640 px native-resolution crops (`data.crop`),
  and validates and predicts with a sliding window over the full board
  (`inference.tile`, see `inference/tiling.py`). Predict:
  `python -m inference.predict --config configs/pcb.yaml --weights runs/pcb_siamese_v1/best.pt --reference data/pcb/reference/01.JPG --input data/pcb/images/Short/01_short_01.jpg --save-vis vis.jpg`

## Annotating with Label Studio

`labelstudio/` sets up [Label Studio](https://labelstud.io) for Siamese pairs:
- Each task shows the **reference (golden) image on the left** and the
  **inspected image on the right**. Defects are drawn on the inspected image
  as **boxes** (detection) or **polygons / brush masks** (segmentation),
  each with a class from `data.defect_classes`.
- A `defect / good` choice marks defect-free images, which become negative
  pairs. An annotation submitted with no regions also counts as good.

**Upload references and their defect images in Label Studio's UI (per-pair references).**
Label Studio's file upload can't attach two images to one task, so this uses two
projects plus a small helper that pairs uploads by file name:

```bash
labelstudio/start.sh                                       # Label Studio on :8080
conda activate omnisee
python -m labelstudio.pairing --token $TOKEN               # keep running while uploading
```

1. The first run creates **Siamese inbox** and **Siamese pairs** and prints their links.
2. In **Siamese inbox**, use **Import** to drag in references *and* defect images together.
   Name them like this:
   - **Reference:** `shirt01_ref.png`. The `_ref` ending can also be `-ref`, `_reference`,
     `_template` or `_golden`, in any case.
   - **Defect image:** `shirt01_front.jpg`, `shirt01-2.jpg`, ... (the reference name plus
     `_`, `-` or `.` and anything after it).
   - If there is only one reference, every image pairs with it.
3. Within a few seconds each defect image moves to **Siamese pairs**, shown next to
   its reference. Label it there.
   - An image whose reference hasn't been uploaded yet stays in the inbox, and its
     `pairing` column explains what's missing.
4. Export from **Siamese pairs**:
   ```bash
   python -m labelstudio.export_coco --project <pairs id> --token $TOKEN
   ```
   References are copied to `data.reference_dir/ls_upload/` and images to
   `data.images_dir/ls_upload/`.

**Simplest: just Label Studio's own UI (one template for all images).**

```bash
labelstudio/start.sh                                   # http://localhost:8080
python -m labelstudio.config --template data/reference/ChatGPT-original.png \
    --output data/annotations/siamese_template.xml     # the Siamese labelling interface
```

1. In Label Studio, **Create Project** → **Labeling Setup** → **Custom template**.
   Paste the contents of `data/annotations/siamese_template.xml` into **Code**,
   then **Save**. The golden template is embedded in the interface and shown on
   the left of every task. To change templates, re-run the command and paste
   the new XML.
2. **Import** → upload your raw images (drag files in). Each image becomes a
   task, shown on the right next to the template.
3. Label the defects, then **Export** → **JSON**, and convert the export for training:
   ```bash
   python -m labelstudio.export_coco --ls-json ~/Downloads/project-1-at-....json
   ```
   This copies the uploaded images out of `data/label-studio/media/upload/`
   into `data.images_dir/ls_upload/` and pairs them with the template that
   `labelstudio.config --template` recorded.

The scripted setup below (`setup_project`, the upload page) is for per-image
templates or pre-filled labels.

Label Studio lives in its own conda env, so its Django pins stay out of the
training env:

```bash
conda create -n labelstudio python=3.11 -y && conda run -n labelstudio pip install label-studio

# 1. start it, serving files from data/ (add --allow-local-ml for model assist)
labelstudio/start.sh --allow-local-ml           # http://localhost:8080, sign up

# 2. create the project (token: Account & Settings -> Personal Access Token);
#    --import adds existing labels as editable pre-annotations
python -m labelstudio.setup_project --token $TOKEN \
    --import data/prepared/annotations/train.json --import data/prepared/annotations/val.json \
    [--ml-backend http://localhost:9090]

# optional: model pre-labels (boxes) from a trained checkpoint, in the training env
python -m labelstudio.ml_backend --weights runs/siamese_defect_v1/best.pt --port 9090

# 3. after labelling: Label Studio -> COCO + reference_image train/val/test for training
python -m labelstudio.export_coco --project 1 --token $TOKEN [--semantic-masks data/prepared/masks]
#    (or --ls-json export.json from the UI's Export -> JSON)
```

**Uploading raw data and the template from the browser.** Instead of putting
files into `data/` by hand, run the upload page next to Label Studio:

```bash
python -m labelstudio.uploader --token $TOKEN [--project 1]   # http://127.0.0.1:8090
```

1. Drop the **template** (the golden reference image) on the left. It is saved to
   `data.reference_dir` and becomes the current template.
2. Drop **raw images** (files or whole folders) on the right. Each one is saved under
   `data.images_dir/raw/<date>/`, paired with the current template, and added to
   the Label Studio project as a task right away.

Notes:
- Re-uploading the same file is detected and skipped.
- A different file with the same name is saved with a `_1` suffix.
- Without `--project`, a new project is created and its id is printed.

- **Hotkeys:** box classes `1`–`9`, polygon classes `q`…`o`, brush classes `z`…`m`,
  and `d` / `g` for defect / good.
- **Pairing:** each task's reference comes from the imported labels,
  `--default-reference`, or `--reference-map`. Its `data.file_name` /
  `data.reference_image` carry the pairing through to the export.
- **Masks:** brush masks and polygons are exported as COCO polygon
  `segmentation`, with the bbox set to their extent.
- **Overwrite warning:** like the annotator, the export writes to the
  `data.train_split` directory by default and replaces the json files there.
  Use `--out` to write elsewhere.

## Annotating with the built-in annotator

`annotator/` is a lighter, dependency-free alternative to Label Studio: a local browser-based labelling tool that writes the same
COCO + `reference_image` format the dataset loader reads. Each annotation has
a class from `data.defect_classes` and is either a **box** (detection) or a
**polygon / brush mask** (segmentation, stored as COCO polygon
`segmentation`, with `bbox` set to the mask's extent so it also trains the
detector).

```bash
# reads images/reference dirs and classes from configs/config.yaml;
# --import merges existing labels (repeatable), --weights enables model assist
python -m annotator --import data/prepared/annotations/train.json \
    --import data/prepared/annotations/val.json \
    [--weights runs/siamese_defect_v1/best.pt]
# open http://127.0.0.1:8000
```

- The reference image is shown next to the inspected one, with zoom and pan kept in sync. Each image's
  reference can be changed from the dropdown.
- Tools: `B` box, `P` polygon, `R` brush, `E` eraser, `V` select/edit. `1`–`9` picks
  the class (and re-classes the selected shape). `G` marks an image *good* (no
  defects), which is exported with zero annotations as a negative pair. `A` runs the
  model and adds dashed box proposals: accept them with `K` or `Shift+K`, or delete them. `?` lists all
  shortcuts.
- Work autosaves to `data/annotations/project.json` (previous version in `.bak`).
- **Export…** writes `all.json` + `train/val/test.json` to the
  `data.train_split` directory, using `scripts/convert_coco_to_siamese.py`'s split
  logic. It can also write per-image semantic mask PNGs
  (pixel = class index + 1). Unlabeled images and unreviewed model proposals
  are left out. **Note:** exporting to the default directory overwrites the
  existing `train.json`/`val.json` there, so `--import` them first.

The current model only consumes `bbox` + `category_id`; the segmentation
labels are kept for a mask head (see `DESIGN.md`).

## Status

This is a from-scratch design + reference implementation (no prior codebase existed).
All Python files are syntax-checked (`python -m py_compile`), but this environment has no
PyTorch installed, so nothing has been executed against real tensors/data yet. Before
training on real data, run the smoke test described in `DESIGN.md` section 17.
