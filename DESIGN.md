# Siamese Instance Segmentation + Defect Classification for T-shirt Inspection

> **Scope change:** the implementation has since been reduced to **detection only** —
> the instance-segmentation (prototype-mask) branch, mask coefficients, `ProtoNet`,
> mask BCE+Dice loss, and mask mAP evaluation described throughout this document have
> been **removed from the code**. `SegmentationHead`/`SiameseInstanceSegmentation` are
> now `DetectionHead`/`SiameseDefectDetector` (`models/detection_head.py`,
> `models/siamese_detector.py`), predicting only `cls_logits`/`box_reg`/`centerness`
> per location; evaluation uses box IoU / box mAP (`evaluation/metrics.py`,
> `training/validate.py`) instead of mask IoU / mask mAP. Every section below that
> discusses masks, `ProtoNet`, or mask losses (§2–3.4, §7.2, §10's segmentation row,
> §17's `segmentation_head.py`/`SiameseInstanceSegmentation` rows) documents the
> *original* design intent, not the current code — kept for architectural context, not
> as an accurate description of what's implemented. See `README.md` for the current
> status.
>
> **Annotation schema change:** §5.2 below describes a bespoke "minimal COCO-style"
> schema with a nested per-image `defects[]` list. The actual schema
> (`datasets/dataset.py`, `data/annotations/example_annotation.json`) is now **plain
> COCO** (`images`/`annotations`/`categories`, unmodified) plus exactly one added
> field: `reference_image` on each `images[]` entry. This was a deliberate choice to
> keep existing COCO tooling/exports (e.g. Roboflow) usable with minimal transformation
> — see `scripts/convert_coco_to_siamese.py`, which only adds `reference_image` (and
> optionally renames category names / prefixes `file_name`), rather than restructuring
> annotations into a custom per-image list.

Full system design for an industrial garment-inspection model. Companion to the working
code under `models/`, `losses/`, `datasets/`, `training/`, `inference/`, `evaluation/`,
`export/`, `configs/config.yaml`.

**How to read this document.** Every non-trivial claim is tagged:

| Tag | Meaning |
|---|---|
| **[REC]** | Architectural recommendation based on established literature / engineering reasoning. Not yet measured on this dataset. |
| **[VALIDATED-ELSEWHERE]** | Well-established result from published work (e.g. focal loss beats plain CE under class imbalance), being reused, not re-derived. |
| **[NEEDS ABLATION]** | A specific, cheap experiment is defined (section 22) to confirm or reject this choice before it is trusted in production. |
| **[ASSUMPTION]** | Something taken as given because it wasn't specified (e.g. dataset size, GPU budget) — flagged so it can be corrected. |

Nothing in this document should be read as "the Siamese architecture is proven better than
a plain detector" — section 20/22 exist specifically because that has *not* been shown, and
must be measured, not assumed.

---

## 1. Objective recap

Given a camera image of an inspected T-shirt and a golden/approved reference image of the
same style/color/size, the system must: (a) gate on "is this a valid T-shirt", (b) detect
and segment every defect instance, (c) classify each defect, (d) produce confidences, and
(e) flag visual anomalies that don't match any trained defect class — while remaining
deployable at line-rate through OpenVINO.

---

## 2. Architecture space evaluated, and the recommendation

### 2.1 Ten candidate designs

| # | Approach | Verdict |
|---|---|---|
| 1 | Feature concatenation `cat(F_in, F_ref)` | Doubles channel count into the head; head must *learn* to subtract — wastes capacity. Used only as an ingredient, not standalone. |
| 2 | Absolute difference `\|F_in - F_ref\|` | Cheap, effective for *aligned* pairs, but discards sign (can't distinguish "ink added" from "ink missing") and discards raw appearance (can't detect a defect visible without any reference, e.g. novel stain shape). |
| 3 | Subtraction `F_in - F_ref` | Keeps sign, still fully destroys absolute appearance signal — same limitation as #2 for reference-independent evidence. |
| 4 | Cosine similarity | A single scalar per location is a strong 1-D anomaly-ness signal, cheap, differentiable, good auxiliary channel — too weak alone as the only comparison signal for locating *where* and *what shape* a defect is. |
| 5 | Cross-attention (ref ↔ input) | Best at modeling *global* geometric correspondence (useful when alignment is imperfect), but O((HW)²) cost is only tractable at the coarsest FPN levels (P6/P7, 4×4–8×8 grids here). |
| 6 | Multi-scale feature comparison | Not an alternative to the above — an orthogonal axis. Defects range from hairline cracks (needs P3, stride 8) to large stains/misprints (needs P5+). **Multi-scale is mandatory, not optional.** |
| 7 | Siamese FPN | This is the neck architecture, not a head choice — adopted regardless of which detector head sits on top. |
| 8 | Siamese Mask R-CNN | Two-stage (RPN → RoIAlign → box/mask heads). Mature, well-understood accuracy ceiling, but: two forward passes through the RoI heads, dynamic-shape RoIAlign/RoIPool complicates ONNX→OpenVINO export and batching, and RPN proposal count varies per image (bad for fixed-latency line-rate guarantees). |
| 9 | Siamese Mask2Former | Best segmentation quality in the general vision literature (transformer decoder + masked attention), but: heavy (decoder + pixel-decoder + multi-round attention), harder to export (deformable attention / masked attention ops are not first-class ONNX ops in every opset), higher latency, and its main strength (panoptic-quality boundaries on complex natural scenes) is overkill for well-lit, near-planar garment defects. |
| 10 | Siamese YOLO-style single-stage segmentation | Anchor-free/anchor-based single-stage detector + a lightweight per-instance mask branch (YOLACT/CondInst/YOLOv8-seg family). Single forward pass, static-shape-friendly, mature ONNX/OpenVINO export path, real-time on CPU/iGPU/VPU. Segmentation quality is slightly behind Mask2Former/Mask R-CNN on complex natural imagery, but garment defects are comparatively simple, mostly-convex regions on a near-planar, well-lit surface — the accuracy gap here is expected to be small. **[NEEDS ABLATION]** |

### 2.2 Recommendation

**[REC]** Use **option 10 as the base detector/segmenter** (single-stage, anchor-free,
FCOS-style dense head + YOLACT/CondInst-style shared-prototype mask branch), with:
- **Siamese FPN** (option 7) as the shared-weight neck — non-negotiable, this *is* the
  Siamese part of the architecture.
- **Multi-scale comparison** (option 6) applied at every FPN level.
- **Concat + abs-diff + cosine, fused via a residual 1×1 conv** (options 1+2+4 combined)
  as the per-level fusion operator at P3–P5.
- **Cross-attention** (option 5) reserved for P6/P7 only, where its quadratic cost is
  cheap (≤64 tokens) and it captures long-range/global misalignment cues that local
  diff/concat cannot.
- Mask R-CNN and Mask2Former are kept as the **explicit accuracy baselines** the final
  choice must beat by an acceptable margin relative to its latency cost (see sections 18-19)
  — they are not ruled out permanently, just not the default starting point.

This is implemented exactly as described in `models/siamese_instance_segmentation.py`,
`models/feature_comparison.py`, `models/segmentation_head.py`.

---

## 3. Detailed model architecture

### 3.1 Backbone

| Backbone | Params | Rel. speed | ONNX/OpenVINO maturity | Verdict |
|---|---|---|---|---|
| ResNet-50 | 25M | 1.0x (reference) | Excellent — canonical export target for years | **[REC] default** |
| ResNet-34 | 21M | ~1.4x faster | Excellent | **[REC] for CPU/edge line rate**, small accuracy cost |
| EfficientNet-B0..B3 | 5-12M | Similar-to-faster FLOPs, worse wall-clock on CPU (depthwise convs, more ops in the OpenVINO graph) | Good but depthwise/SE blocks are less throughput-optimal on many CPU/VPU targets than plain convs | Only worth it if GPU-only deployment |
| ConvNeXt-Tiny | 28M | ~0.9x | Good (plain conv/LN-based, but LayerNorm-over-CHW needs a permute that adds export nodes) | Slightly better accuracy than ResNet-50 in general benchmarks; consider once baseline is trusted |
| MobileNetV3-Large | 5M | ~3-4x faster | Good, common OpenVINO demo target | **[REC] fallback for CPU-only inspection stations** where latency dominates over the last few AP points |
| DINO/ViT-B | 86M+ | Much slower on CPU, needs large batches/GPU to be efficient | Weaker (attention + patch embed reshape ops less optimized in some OpenVINO releases); great as a *frozen* self-supervised feature source for anomaly detection research, not for the real-time head | Not recommended for the deployed detector; noted as a candidate for an offline research track on unknown-defect detection only |

**[REC] Default: ResNet-50, ImageNet-pretrained, shared between both branches** (`models/backbone.py:ResNetBackbone`).
Rationale: best-understood export path to OpenVINO, good accuracy/latency balance, and the
industrial requirement (section 15) weighs export/runtime maturity heavily — this is not
the highest-accuracy option available, it is the *lowest-risk* one that meets the
accuracy bar, per **[NEEDS ABLATION]** in section 22 (backbone ablation).

`MobileNetV3Backbone` is included in the same file as the drop-in CPU-constrained option.

### 3.2 Feature pyramid / multi-scale features

Standard top-down FPN (`models/siamese_encoder.py:FPN`) on top of C3/C4/C5
(strides 8/16/32), with two extra strided levels P6/P7 (strides 64/128) added directly
from C5, giving 5 levels: **P3, P4, P5, P6, P7**, all at **256 channels**
(`configs/config.yaml: model.fpn.out_channels`). Scale assignment per level
(`models/segmentation_head.py: SCALE_RANGES`) follows FCOS convention:

```
P3: objects with max(l,t,r,b) in [0,   64)   px  <- hairline cracks, small stains
P4: [64,  128)  px  <- medium print defects
P5: [128, 256)  px  <- large misprints, big stains
P6: [256, 512)  px  <- near-whole-garment color deviation
P7: [512, inf)  px  <- whole-panel ghosting / global misalignment
```

Both the reference image and the inspected image are pushed through **the same**
`SiameseEncoder` instance — true weight sharing (single `nn.Module`, called twice), not
two networks kept in sync by a loss term. This halves parameter count vs. a two-tower
design and, more importantly, guarantees the two branches' features live in the exact same
embedding space, which is what makes a distance between them meaningful.

### 3.3 Feature comparison (implementation: `models/feature_comparison.py`)

For P3–P5 (`LevelFusion`):

```
diff      = |F_in - F_ref|
cos_map   = cosine_similarity(F_in, F_ref)     # per-pixel, 1 channel
combined  = ReLU(GroupNorm(Conv1x1(cat[F_in, F_ref, diff, cos_map])))
F_fused   = F_in + combined                     # residual
```

The residual connection is the key design decision: **if the reference is missing, noisy,
or misaligned, `F_fused` degrades gracefully to `F_in` (+ a small, bounded correction)
rather than collapsing** — the detector still functions as a plain single-image defect
detector, it simply loses the comparison signal. This directly supports "unknown defect /
no reference available" operation modes without a second model.

For P6/P7 (`CrossAttentionFusion`): single-head scaled-dot-product cross-attention,
`Q` from `F_in`, `K/V` from `F_ref`, residual-added back onto `F_in`. Restricted to the two
coarsest levels (≤ 8×8 = 64 tokens at 800×800 input) to keep the O(N²) cost negligible
(<0.1% of total FLOPs) while still letting the model reason about global
translation/rotation offsets between reference and input that a purely local per-pixel diff
cannot capture.

### 3.4 Instance segmentation head (implementation: `models/segmentation_head.py`)

Per FPN location (dense, anchor-free — FCOS-style), four parallel branches share a 4-conv
tower each (`cls_tower`, `box_tower`):

```
cls_logits   : (num_classes, H, W)   -> per-location defect-class logits
box_reg      : (4, H, W)             -> distances (l, t, r, b) to instance edges, in pixels
centerness   : (1, H, W)             -> "how central is this point in its instance" quality score
mask_coeff   : (K=32, H, W)          -> coefficients over K shared prototypes
```

A single `ProtoNet` (on P3, the highest-resolution level) produces **K=32 prototype masks**
at stride 4 (`configs/config.yaml: model.seg_head.proto_channels/proto_level`), shared by
every instance in the image — this is the YOLACT/CondInst trick that lets mask quality scale
with image resolution rather than a fixed 28×28 RoI grid, while staying O(1) in the number
of instances (only the K coefficients are predicted per-instance, not a full per-instance
mask decoder).

**Multiple instances** are handled the same way FCOS/YOLACT handle them: every spatial
location predicts an independent set of (class, box, centerness, coefficients); the
model imposes no upper bound on instance count architecturally. At inference,
per-class NMS (`torchvision.ops.batched_nms`) reduces the dense set of ~10⁴-10⁵ candidate
locations down to the actual instances (`SiameseInstanceSegmentation.postprocess`).
Final per-instance mask = `sigmoid(Σ_k coeff_k · prototype_k)`, cropped to the predicted box
via a differentiable affine-grid RoI crop (`models/segmentation_head.py:roi_crop_resize`) —
chosen over `torchvision.ops.roi_align` specifically because `affine_grid`/`grid_sample`
export to a plain ONNX op with no custom-op risk in the OpenVINO conversion path.

Example decoded output (matches the brief's format exactly):

```
Instance 1: class=print_crack   confidence=0.96  bbox=(640,300,820,390)  mask=<28x28 logits, upsampled to bbox>
Instance 2: class=spot_stain    confidence=0.91  bbox=(1200,900,1260,955) mask=<...>
```

### 3.5 Defect classification

**Per-instance defect class** is `cls_logits` inside `SegmentationHead` — a dense,
per-location, multi-class (not multi-label) prediction, trained with focal loss. It is
*not* a separate whole-image classifier; keeping it fused with the box/mask branches lets
one NMS pass resolve class + location + mask jointly, avoiding an inconsistent
detector-then-classifier cascade.

---

## 4. Tensor / data flow diagram

```
reference_image (B,3,800,800)          input_image (B,3,800,800)
         │                                      │
         └──────────────┬───────────────────────┘
                         │  SAME shared-weight module, called twice
                 ┌───────▼────────┐
                 │  ResNet-50      │   C3 (B,512,100,100) stride 8
                 │  (frozen stem)  │   C4 (B,1024,50,50)  stride 16
                 └───────┬────────┘   C5 (B,2048,25,25)   stride 32
                         │
                 ┌───────▼────────┐
                 │      FPN        │   P3(B,256,100,100) P4(B,256,50,50)
                 │                 │   P5(B,256,25,25)  P6(B,256,13,13) P7(B,256,7,7)
                 └───────┬────────┘
        ref_feats{P3..P7}│  inp_feats{P3..P7}          ref_embed(B,256)  inp_embed(B,256)
                 │        │                                    │              │
                 └───┬────┘                                    └──────┬───────┘
                     │  FeatureComparison (per level)                 │
             ┌───────▼────────┐                             ┌─────────▼─────────┐
             │ fused{P3..P7}   │                             │ L_siamese (contrastive)│
             │ (B,256,H,W) ea. │                             └────────────────────┘
             └───────┬────────┘
        ┌────────────┼─────────────────────┐
        │            │                     │
 ┌──────▼──────┐┌────▼─────┐        ┌──────▼──────┐
 │ cls_tower    ││ box_tower │        │  ProtoNet   │  (on fused P3)
 │→cls_logits   ││→box_reg   │        │→prototypes  │
 │(B,C,H,W)     ││→centerness│        │(B,32,200,200)│
 │              ││→mask_coeff││        └─────────────┘
 └──────────────┘└───────────┘
        │            │                     │
        └─────┬──────┴──────────┬──────────┘
              │  per-level, all locations   │
       decode_boxes + score threshold + batched_nms
              │
     surviving instances: boxes, scores, labels, mask_coeffs
              │
      assemble_instance_masks(prototypes, coeffs, boxes) -> per-instance (28,28) mask logits
              │
      resize each mask to its own box, paste onto full-resolution canvas
              │
   ┌──────────▼───────────────────────────────────────────────┐
   │ final output: instances[{class, confidence, bbox, mask}],│
   │ embedding_distance (anomaly score)                         │
   └────────────────────────────────────────────────────────────┘
```

---

## 5. Dataset design

### 5.1 Directory structure (extends the brief's proposal)

The brief's `reference/` + `images/` split is kept. Each inspected image simply names the
reference image it is paired against; no separate garment/production identity registry is
kept (deliberately minimal — see §5.2). Annotations are centralized (COCO-style) rather than
one file per image:

```
data/
├── reference/
│   ├── ST001_BLACK_L_v1.png
│   └── ...
├── images/
│   ├── tshirt_001_defect_001.png
│   ├── tshirt_001_good_001.png       # good (no-defect) captures matter just as much
│   └── ...
└── annotations/
    ├── annotations.json              # full COCO-extended dataset (schema below)
    ├── train.json / val.json / test.json   # disjoint by reference_image, never by image_id
    └── example_annotation.json       # see data/annotations/example_annotation.json
```

**Why split by `reference_image`, not `image_id`:** two captures of the *same physical
garment* (e.g. re-shot after a lighting adjustment) are strongly correlated; letting one land
in train and the other in val leaks information and inflates validation metrics. **[REC]**

### 5.2 Annotation format: minimal COCO-style JSON

**Recommendation: a minimal COCO-style JSON** — deliberately stripped down to exactly what
the model consumes, not raw COCO and not YOLO-seg. Reasoning:

| Format | Verdict |
|---|---|
| YOLO-seg (`.txt` per image, normalized polygon) | No native place for `reference_image`; would need a parallel sidecar file anyway. Rejected. |
| Raw COCO | Has `images`/`annotations`/`categories`, well-supported by `pycocotools` — good foundation, but has no concept of "reference image"/pairing at all. |
| **Minimal COCO-style (chosen)** | Keep COCO's polygon-or-RLE segmentation/bbox/category conventions (so `pycocotools` mask utilities keep working), and add only a `reference_image` field per inspected image. No garment/production identity, camera, alignment, lighting, or severity metadata — see below. |

Required fields per entity (also see the working example at
`data/annotations/example_annotation.json`, and the loader `datasets/dataset.py`):

- **Image**: `image_id`, `file_name`, `reference_image` (path to its paired reference),
  `width`, `height`, and `defects[]`.
- **Defect instance**: `category`, `bbox` (COCO xywh), `segmentation` (`polygon` or `rle`,
  both supported — see `_decode_segmentation` in `datasets/dataset.py`). Nothing else —
  no `instance_id`, `category_id`, `area`, `severity`, or `iscrowd`; `category_id` is
  derived from `configs/config.yaml -> data.defect_classes` at load time and `area` from the
  decoded mask if a metric needs it.

Multiple defect instances per image are simply multiple entries in the `defects[]` array —
no special encoding needed since segmentation is instance-level (each polygon/RLE is one
instance), unlike a single-mask-per-image semantic-segmentation encoding which cannot
separate touching/overlapping defects of different classes.

**What this intentionally drops** (garment/style/color/size/design/production-order/
production-line identity, camera id, capture alignment, lighting condition, per-defect
severity, and the whole-image `valid_tshirt` gate/model head) was present in an earlier
revision of this schema. It was cut to keep the annotation surface to exactly
{reference image, per-defect class + bbox + mask} — if garment traceability, severity-based
FRR/FAR calibration (§10), or a pre-segmentation "is this even a valid T-shirt capture" gate
(§14) become requirements again, they should be reintroduced as their own explicit fields/
heads rather than assumed implicit in the defect annotation.

A full worked example is at **`data/annotations/example_annotation.json`**.

---

## 6. Positive/negative pair generation (implementation: `datasets/pair_sampler.py`)

### 6.1 Pair taxonomy

```
(reference=golden, inspected=good, same reference_image)      -> pair_label = 0 (negative)
(reference=golden, inspected=defective, same reference_image) -> pair_label = 1 (positive)
```

`pair_label` is derived automatically in `SiameseDefectDataset.__getitem__` as
`float(len(defects) > 0)` — no separate manual labeling step needed beyond the defect
annotations already required for segmentation.

### 6.2 Variation axes deliberately included in training pairs

| Axis | Same across ref/input? | Why |
|---|---|---|
| Reference image | Same `reference_image` required for a valid pair | A different reference is not a meaningful "defect" comparison — that's a *reference-selection* error, handled upstream by whatever assigns `reference_image` per capture, not a training pair. |
| Lighting | **Deliberately varied** (independent photometric augmentation, §9) | The model must be invariant to normal line-to-line lighting drift; this is the single most important axis for reducing false rejects. |
| Camera position / minor viewpoint | **Deliberately varied** (small independent geometric jitter on the reference branch, §9) | Reference and inspected shots are never pixel-identical even after ORB+RANSAC alignment; the model must tolerate residual misalignment rather than treating it as a defect. |
| Fabric deformation / wrinkle | **Deliberately present in "good" captures** | Real production garments wrinkle; if all "good" training examples are pressed-flat, the model will flag every wrinkle as an anomaly. This is a **data collection requirement**, not something augmentation can safely fabricate (synthetically warping a flat good image risks looking like fabric damage — a label-destroying augmentation, see §9). |

### 6.3 Hard-negative mining

Implemented in `datasets/pair_sampler.py: HardNegativePool`:

1. Maintain a pool of all "good" (negative) pairs.
2. Every N epochs (`configs/config.yaml: hard_negative_refresh_every_epochs`, starting at
   `hard_negative_mining.start_epoch` so the embedding space is not still random), run the
   **encoder only** (no detection/seg heads — cheap) over the negative pool and rank by
   embedding distance.
3. Pairs with **the largest distance despite being a true no-defect pair** are, by
   definition, the ones the model is currently closest to false-rejecting. Keep the top
   `hard_negative_pool_size` as the hard-negative pool.
4. `SiamesePairBatchSampler` then draws each training batch with a fixed ratio
   (`positive_ratio`, `hard_negative_ratio` in config) — e.g. 50% positive, 30% of the
   remaining negatives drawn from the hard pool, 70% from the general negative pool —
   guaranteeing hard cases are over-represented without discarding easy-negative signal
   entirely (which would destabilize the decision boundary).

**[REC]** This targets exactly the production failure mode that matters most economically:
false rejects from benign lighting/wrinkle/alignment variation, not just held-out defect
recall.

---

## 7. Loss functions

### 7.1 Detection (implementation: `losses/detection_loss.py`)

**Classification — Focal Loss** (chosen over plain BCE/CE): with ~10⁴-10⁵ candidate
locations per image and only a handful truly positive, plain CE is dominated by easy
negatives. **[VALIDATED-ELSEWHERE]** (Lin et al., RetinaNet, 2017).

```
FL(p_t) = -α_t (1 - p_t)^γ log(p_t)     where  p_t = p if y=1 else (1-p)
```
Defaults `α=0.25, γ=2.0` (config: `loss.detection.focal_alpha/gamma`).

**Box regression — CIoU** (chosen over Smooth-L1/GIoU): Smooth-L1 optimizes 4 independent
coordinates and is scale-sensitive across FPN levels; plain IoU/GIoU ignores aspect-ratio
consistency, which matters for thin defects like print cracks. CIoU adds a center-distance
and aspect-ratio-consistency term on top of IoU:

```
L_CIoU = 1 - IoU + ρ²(b, b_gt)/c²  +  α·v
  v = (4/π²) (arctan(w_gt/h_gt) - arctan(w/h))²
  α = v / (1 - IoU + v)                     (detached, no gradient through α)
```
where `ρ` is the Euclidean distance between predicted/GT box centers and `c` is the
diagonal of the smallest enclosing box. **[VALIDATED-ELSEWHERE]** (Zheng et al., 2020).

**Centerness — BCE** against `sqrt((min(l,r)/max(l,r)) · (min(t,b)/max(t,b)))`, standard
FCOS quality target, used to down-weight low-quality (off-center) predictions at NMS time.

### 7.2 Segmentation (implementation: `losses/segmentation_loss.py`)

**BCE + Dice, combined** (chosen over BCE-only, Focal-only, or Tversky-only):

```
L_dice = 1 - (2 Σ(p·y) + ε) / (Σp + Σy + ε)
L_mask = w_bce · BCE(p, y) + w_dice · L_dice          (w_bce = w_dice = 1.0 default)
```

- Pure BCE is dominated by background pixels inside a small mask crop (severe
  foreground/background imbalance even within a single 28×28 instance crop for a thin
  crack). **[VALIDATED-ELSEWHERE]**
- Pure Dice is known to give unstable gradients very early in training when predictions
  start near 0 everywhere; combining with BCE stabilizes early training while Dice drives
  the actual overlap objective. **[VALIDATED-ELSEWHERE]**
- **Tversky loss** (`losses/segmentation_loss.py: tversky_loss`, `α<β` biases recall over
  precision) is implemented but **not the default** — reserved as a **[NEEDS ABLATION]**
  knob for the specific production requirement "missed defects (false accepts) are costlier
  than over-segmented boundaries", to be tuned once FAR/FRR from section 10 are measured on
  a real validation set.

### 7.3 Classification

Per-instance defect class already uses focal loss (§7.1, shared tensors with the box
head — not duplicated). There is no separate whole-image classification loss: the
`valid_tshirt` gate and `GarmentAttributeHead` sanity-check described in an earlier revision
of this design were removed along with their annotation fields (§5.2); reintroduce this
subsection alongside them if they come back.

### 7.4 Siamese metric-learning loss (implementation: `losses/contrastive_loss.py`)

| Loss | Verdict |
|---|---|
| **Contrastive (chosen, [REC])** | `L = (1-y)d² + y·max(0, margin-d)²`. Directly optimizes the same quantity (`d = ‖emb_a - emb_b‖`) that is thresholded at inference for unknown-defect detection (§12) — training and deployment share one metric, no calibration gap. |
| Cosine Embedding Loss | Equivalent in spirit, implemented as an alternative (`CosineEmbeddingLossWrapper`) — use if embedding norm proves unstable and cosine similarity is preferred over Euclidean distance operationally. |
| Triplet Loss | Requires mining a negative *per anchor* (not just per-batch pair), adds a whole extra mining subsystem beyond the hard-negative *pair* mining already needed; only worth it if pairwise contrastive plateaus. Implemented (`TripletLoss`) for that ablation. |
| InfoNCE | Needs in-batch negatives to be meaningful and is most natural for large-batch, GPU-heavy contrastive pretraining (SimCLR-style) — the industrial batch sizes here (8-32) are too small for InfoNCE's negatives to be informative. Implemented (`InfoNCELoss`) but not recommended as the primary loss at this scale. |

### 7.5 Total loss (implementation: `losses/total_loss.py`)

```
L_total = w_det   · (L_cls_focal + L_box_ciou + L_centerness_bce)
        + w_seg   · L_mask_bce_dice
        + w_siam  · L_contrastive
```

**[REC]** Start with weights that equalize each loss's *initial* magnitude (log once per
loss at step 0, set `w_i ∝ 1/initial_loss_i`), not with arbitrary defaults — the values in
`configs/config.yaml` (`detection=1.0, segmentation=2.0, siamese=0.5`) are a
reasonable starting point but **[NEEDS ABLATION]**: re-tune once real loss curves exist,
and consider annealing `w_siam` down mid-training once the encoder embedding space has
stabilized (an over-weighted metric-learning loss late in training can pull the shared
backbone away from the fine-grained localization features the segmentation head needs).

---

## 8. Training strategy (implementation: `training/train.py`, `training/scheduler.py`)

### 8.1 Staged pipeline

1. **Dataset prep**: reference-image-level train/val/test split (§5.1), verify every image's
   `reference_image` path resolves, verify annotation polygon/RLE decode round-trips.
2. **Pair generation**: `datasets/pair_sampler.split_positive_negative` + hard-negative pool
   initialized from all negatives (uniform at epoch 0, since embeddings aren't meaningful
   yet).
3. **Augmentation**: per §9, joint on (input image, boxes, masks), independent on
   photometrics + a small reference-only geometric jitter.
4. **Backbone init**: ImageNet-pretrained ResNet-50, **`frozen_stages=1`** (stem + layer1
   frozen) for the first `warmup_epochs` — prevents the randomly-initialized heads from
   destroying good low-level ImageNet features early via large gradients.
5. **Phase A — joint warm-up** (epochs 0 → `warmup_epochs`): all heads training, backbone
   stem/layer1 frozen, LR ramping via `WarmupCosineScheduler`, hard-negative mining
   disabled (embeddings not stable enough to rank yet).
6. **Phase B — full joint training** (through `hard_negative_mining.start_epoch`): unfreeze
   backbone fully, cosine LR decay, all losses active with fixed/initial-magnitude-balanced
   weights.
7. **Phase C — hard-negative-mining phase** (`start_epoch` → end): pool refreshed every
   `hard_negative_refresh_every_epochs`, batch sampler shifts toward hard negatives per
   §6.3.
8. **Validation** every epoch (`training/validate.py`): full decode+NMS+mask-assembly on
   held-out garments, mask mAP / AP50 / AP75.
9. **Fine-tuning**: once mask mAP plateaus, a short (~10-20 epoch) low-LR
   (`min_lr_ratio`-level) fine-tune restricted to the *segmentation* and *siamese* heads
   only (freeze detection cls/box) is a useful next step if mask boundary quality lags
   box/class accuracy — **[NEEDS ABLATION]**, not yet in `train.py` as an automatic phase.
10. **Early stopping** on `val/mask_mAP` with patience (config: `early_stopping.patience`),
    tracked via an EMA model (`ema_decay=0.999`) which is what's actually checkpointed and
    evaluated — EMA weights are consistently smoother/better for detection-style dense
    heads than raw SGD/AdamW weights. **[VALIDATED-ELSEWHERE]**

### 8.2 Hyperparameters and how to scale them

| Parameter | Default (config) | Scaling rule |
|---|---|---|
| Optimizer | AdamW | **[REC]** — dense detection heads with GroupNorm (not BatchNorm) train stably with AdamW at moderate LR without the SGD momentum/LR-warmup sensitivity of RetinaNet-era recipes. |
| LR | 2e-4 | Scale ~linearly with **effective** batch size (`batch_size × grad_accum_steps`); the config's 2e-4 assumes an effective batch of 16. Halve LR if effective batch is halved. |
| Weight decay | 0.05 | Standard for AdamW + ViT/ConvNeXt-era backbones; lower to 1e-4-style if using plain ResNet-only recipes and seeing under-fitting. |
| Batch size | 8 (× grad_accum=2 → effective 16) | **[ASSUMPTION]** sized for a single 24GB GPU at 800×800 input with ResNet-50 + FPN; raise `grad_accum_steps` (not `batch_size`) first if GPU memory is the binding constraint, since accumulation gives the same effective batch at lower peak memory. |
| Epochs | 120 | Depends heavily on dataset size — **[ASSUMPTION]**. For a few thousand annotated images this is a reasonable ballpark (COCO-scale detectors typically train 12-36 "epochs" over ~118k images, i.e. a similar total-iteration budget); for a smaller pilot dataset (hundreds of images) prefer *fewer* epochs with heavier augmentation and closer early-stopping monitoring to avoid overfitting the reference garments seen during training. |
| Mixed precision | On (`torch.autocast` + `GradScaler`) | **[REC]** — ~40-50% memory/time savings on Ampere+ GPUs with negligible accuracy impact for conv-heavy detectors; disable only if NaNs appear in the focal-loss/CIoU terms (rare, but CIoU's `atan`/division terms are the most fp16-sensitive part of this model — consider running just the box-loss computation in fp32 if this happens). |
| Grad accumulation | 2 | Pure memory/throughput knob, no accuracy implication at fixed effective batch size. |
| EMA decay | 0.999 | Lower (e.g. 0.998) for smaller total step counts so EMA converges within the training budget. |

---

## 9. Data augmentation (implementation: `datasets/transforms.py`)

**Golden rule, stated explicitly because it's the easiest thing to get wrong in this
project: augmentation must never create or destroy a defect region.** This rules out, for
the *inspected* image specifically:

- Cutout / random erasing / CutMix / MixUp — can blank out a real defect (false negative
  label) or paste in a patch that looks like a new defect (false positive label) without
  updating the mask annotation.
- Aggressive elastic/thin-plate-spline deformation strong enough to plausibly mimic fabric
  damage.
- Synthetic copy-paste of a defect crop onto a different image *unless* done as an explicit,
  separately-tracked data-augmentation defect-class balancing technique (noted as a
  **[NEEDS ABLATION]** option for rare classes like `fabric_damage`, but requires its own
  masking of "this is a synthetic instance" for later filtering — not in the default
  pipeline).

| Bucket | Transforms | Applied to |
|---|---|---|
| **Shared/joint** (`build_shared_geometric_transform`) | Horizontal flip, affine (rotate ±8°, translate ±5%, scale ±5%, shear ±3°), perspective (≤5%) | Input image **and** its boxes/masks together, single random draw — geometric only, so pixel-level label alignment is preserved exactly. |
| **Independent photometric** (`build_independent_photometric_transform`) | Brightness/contrast, hue/saturation/value, gamma, blur (Gaussian/motion), Gaussian noise, JPEG compression | Sampled **separately** for reference and input — this is what teaches invariance to the fact that reference and inspected shots are never captured under identical lighting. |
| **Independent reference-only geometric jitter** (`build_reference_geometric_jitter`) | Small affine (±3° rotate, ±2% translate/scale) | Reference branch only, small magnitude — simulates realistic residual ORB/RANSAC misalignment rather than assuming perfect registration, without ever needing to touch (or invalidate) the input's defect masks. |

No transform is applied identically-but-independently-seeded to both branches for anything
beyond the small reference jitter above — the reference and inspected images are
fundamentally *different captures* in production, and training on near-identical pairs
(same random draw for both) would teach the model to expect near-pixel-perfect alignment,
which will not hold on the line.

---

## 10. Evaluation metrics (implementation: `evaluation/metrics.py`)

| Task | Metrics | Notes |
|---|---|---|
| Defect classification | Per-class Precision/Recall/F1, confusion matrix (`classification_prf1`) | Report **per class**, never only a macro-average — `other_anomaly` and rare classes (`fabric_damage`) will have much noisier estimates and must be tracked separately, not hidden inside an aggregate. |
| Instance segmentation | Mask IoU/Dice per-instance (`mask_iou`, `mask_dice`), AP50/AP75/mask-mAP via greedy score-ranked matching (`match_predictions_to_gt`, `mask_map`) | Implementation here is a simplified single-IoU-threshold matcher for tractability inside `training/validate.py`; for a publishable/contractual number, cross-check against `pycocotools.cocoeval.COCOeval` directly on exported COCO-format predictions. |
| **Industrial** | **False Reject Rate (FRR)**, **False Accept Rate (FAR)**, garment-level defect recall, latency (P50/P95), FPS | `false_reject_rate`, `false_accept_rate`, `defect_recall` in `evaluation/metrics.py`, all computed at the **garment pass/fail decision level**, not the instance level — a garment is "defective" if it has ≥1 GT defect instance, "flagged" if the model produces ≥1 instance above threshold. |

**Which metrics matter most in production, and why:** **FAR (false accept rate) is the
metric to optimize hardest** — a defective garment reaching the customer is a quality
escape with brand/cost consequences far larger than one extra manual re-inspection from a
false reject. **[REC]** Report FAR/FRR at multiple operating thresholds (a curve, not a
point) and let QA pick the deployed `inference.score_threshold` from that curve rather than
fixing it during model development — this decouples "is the model good" (mask mAP, PR
curves) from "what threshold does the business want" (a threshold choice, not a modeling
choice).

---

## 11. Reference image handling

**[REC] Selection: fixed per annotation record, resolved upstream.** Each image record
carries a single `reference_image` path (§5.2); the dataset loader (`datasets/dataset.py`)
just reads it directly — there is no garment-identity lookup or dynamic multi-reference
selection inside the training/inference code path. Whatever upstream process (production-
order barcode/RFID, line-station config, or a manual annotation step) decides which
reference a given capture should be compared against is responsible for writing the correct
`reference_image` into the annotation, before the file ever reaches this pipeline.

```
Input capture  -> reference_image already resolved upstream and written into the
                   annotation record (image record's `reference_image` field)
                          ↓
               training/inference: load that single reference_image and compare
```

**Why not a single learned prototype embedding instead of an actual reference image:** a
learned per-style prototype vector would be more compact and would remove the need to store
reference images, but it discards *spatial* information the segmentation head needs (a
prototype vector cannot tell the model *where* on the garment a print element should be,
only that "on average this style looks like X") — the multi-scale feature-map comparison in
this architecture is precisely what a single embedding vector cannot provide. A learned
prototype is a reasonable **[NEEDS ABLATION]** lightweight alternative worth measuring if
reference-image storage/retrieval becomes an infrastructure burden, but it should be
expected to underperform on precise print-alignment defects specifically.

**Multiple approved references per style** (e.g. two acceptable print-registration
tolerances) are not modeled by this schema — if that becomes a requirement, it should be
reintroduced as an explicit list-valued field (e.g. `candidate_reference_images`) with its
own dataset-loader selection logic, rather than assumed.

---

## 12. Unknown/unseen defect detection

**[REC — combine both signals, don't rely on either alone]**

```
Reference embedding  ──┐
                        ├─► d = ‖emb_ref - emb_inspected‖  ──► threshold (config:
Inspected embedding  ──┘                                        inference.anomaly_embedding_threshold)
                                                                        │
                                                          d > threshold ├── AND no supervised
                                                                        │   instance above
                                                                        │   score_threshold
                                                                        ▼
                                                          "unknown_defect_suspected" flag
                                                          (inference/predict.py)
```

The supervised per-instance classifier (§3.4) will, by construction, only ever output one
of the trained defect classes — it cannot flag "something is wrong but I don't know what".
The global embedding distance *can*, because contrastive training pushes "matching" pairs
together in embedding space regardless of *why* a pair doesn't match; a novel defect type
never seen in training will still typically pull the inspected embedding away from the
reference embedding, since it makes the two images genuinely dissimilar.

**Combination logic** (`inference/predict.py: predict`): flag `unknown_defect_suspected`
when embedding distance exceeds threshold **but** the supervised head found *no* localized
instance — i.e. "the model senses something is different, but can't point at a trained
defect class for it." When the supervised head *does* fire, trust its localized output
(it's more actionable — a bounding box + mask beats a single scalar) and treat the embedding
distance purely as an auxiliary confidence signal, not a competing decision.

This is **explicitly a heuristic, not a proven open-set detector** — a global P5-pooled
embedding distance is a coarse anomaly signal (it will miss small/local unknown defects
that don't move the *global* average feature much) and should be paired with classical
anomaly-detection tooling for anything safety/compliance-critical (§13, Anomalib
integration) rather than trusted alone. **[NEEDS ABLATION]**: measure recall of this signal
specifically on a held-out set of defect types *excluded* from training, before relying on
it operationally.

---

## 13. Integration with existing classical-CV inspection algorithms

**None of ORB+RANSAC, SSIM, CIEDE2000, coverage masks, or Anomalib should be replaced
wholesale by the Siamese model.** They solve different, complementary problems, several of
which the Siamese model depends on as *preconditions* rather than duplicates:

| Component | Role | AI or Classical? |
|---|---|---|
| T-shirt detection (locate + crop garment in frame) | Precondition for everything downstream | Either — a small detector (could even be a lightweight head reusing this same backbone) or classical background-subtraction if the line has a fixed, controlled backdrop. |
| **ORB+RANSAC alignment** | Coarse geometric registration of inspected image to reference before comparison | **Classical CV — keep.** The Siamese model's cross-attention (P6/P7) and reference-jitter augmentation (§9) are designed to tolerate *residual* misalignment after this step, not to replace it; expecting the neural net to solve large, unconstrained geometric registration end-to-end is both wasteful (a well-understood 10-line OpenCV solution already exists) and less reliable than a dedicated keypoint-matching algorithm. |
| **SSIM** | Fast, cheap, interpretable structural-similarity pre-check | **Classical CV — keep as a fast pre-filter / secondary evidence signal.** A very low SSIM combined with a very low Siamese embedding distance is suspicious (possible misalignment/lighting issue rather than a real defect) — a useful sanity cross-check between the two signals, cheap enough to run on every frame before the (heavier) neural forward pass. |
| **CIEDE2000** | Perceptually-calibrated color-difference metric | **Classical CV — keep, feed as a verification signal for the `color_deviation` class specifically.** The learned classifier can flag "color_deviation" from texture/context cues, but CIEDE2000 gives a calibrated, auditable ΔE number a QA engineer can defend in a customer dispute — pair the two rather than trusting the network's confidence alone for this one class. |
| **Coverage masks** (print completeness) | Deterministic check: "is X% of the expected print area present" | **Rule-based, keep.** This is exactly the kind of check that should stay a transparent, deterministic rule (coverage % vs. a threshold) rather than being folded into a black-box network output, precisely *because* it's auditable and easy to explain to a customer/auditor. Complements `print_missing`/`print_misalignment` classes. |
| **Anomalib** (surface anomaly detection, e.g. PatchCore/PaDiM) | General unsupervised anomaly localization on the fabric surface | **AI, but a separate specialized model — keep running in parallel, not replaced.** Anomalib-style memory-bank/normalizing-flow approaches are specifically tuned for texture-level surface anomaly localization without needing per-defect labels, and are a strong complement to the Siamese model's unknown-defect signal (§12) — an ensemble/agreement check between "Siamese embedding distance flags an anomaly" and "Anomalib flags a surface anomaly at the same location" is a much stronger unknown-defect decision than either alone. |
| **Siamese instance segmentation + classification (this system)** | Localized, classified, multi-instance defect detection against a reference | **AI — the new component**, positioned *after* alignment/gating and *alongside* (not instead of) the classical checks above. |
| **Final decision** | Combine supervised classification + classical CV verification + rule thresholds into pass/fail | **Rule-based fusion layer** — e.g. "fail if any supervised instance confidence > threshold, OR coverage-mask rule fails, OR (Siamese anomaly AND Anomalib anomaly agree on unknown defect)". Keep this fusion logic simple, deterministic, and auditable — it is the layer QA/compliance will actually need to inspect and justify. |

### Hybrid pipeline

```
Camera
  │
  ▼
T-shirt Detection            (AI, lightweight, or classical if backdrop is controlled)
  │
  ▼
Alignment (ORB + RANSAC)     (Classical CV)
  │
  ▼
Reference Selection          (Resolved upstream and written into the annotation
  │                            record's `reference_image` field, §11)
  │
  ▼
┌─────────────────────────────┬─────────────────────────────┐
│ Siamese Instance Segmentation│ SSIM / CIEDE2000 / Coverage │ Anomalib surface
│ + Defect Classification (AI) │ masks (Classical CV)         │ anomaly (AI, separate model)
└──────────────┬───────────────┴───────────────┬─────────────┘
               │                                │
               ▼                                ▼
      Rule-based / Classical CV Verification & Fusion
               │
               ▼
          Final Decision  (pass / fail / manual-review)
```

---

## 14. Industrial real-time inference pipeline

```
Camera → GStreamer capture → Preprocessing (resize/letterbox to 800x800, normalize)
       → T-shirt Detection gate (reject non-garment/mis-framed frames fast, cheap;
         classical or a separate lightweight detector, §13 — not part of this model)
       → Reference Selection (resolved upstream, §11)
       → Alignment (ORB+RANSAC, classical, ~5-15ms on CPU)
       → Siamese Instance Segmentation (the model in this document)
       → Post-processing (NMS + mask assembly, outside the exported graph, §15)
       → Rule-based fusion with classical CV verification (§13)
       → Defect Decision (pass/fail/manual-review)
       → Visualization overlay (§16)
       → GStreamer/UI output
```

- **Expected latency [ASSUMPTION — needs on-target measurement]:** ResNet-50 + FPN +
  single-stage head at 800×800, FP16 on a mid-range discrete GPU: roughly 15-30ms per
  image pair (two backbone passes, shared weights so no extra parameter load, plus a light
  head) → 30-60 FPS single-stream. On CPU/OpenVINO with INT8 (§15), expect substantially
  higher latency (likely 80-200ms depending on CPU/VPU class) — **this must be measured on
  the actual target hardware before being written into a line-rate SLA.**
- **Batch vs. single-image:** the reference image for a given (style, color, size) is
  constant across many consecutive inspected images on a production line — **cache the
  reference branch's FPN features** (`SiameseEncoder._encode_one(reference)`) once per
  style/color/size change rather than recomputing them per frame; only the inspected branch
  needs to run per-frame. This roughly halves the effective per-frame backbone cost in
  steady-state line operation and is a bigger practical win than batching multiple frames
  together (which adds latency/queueing that a single inspection station usually can't
  afford).
- **Memory:** model + FPN + heads in FP16 is well under 200MB; the dominant runtime memory
  cost is decoded frame buffers, not the model.
- **Model loading:** load once at process start; reference-embedding cache (previous
  bullet) should be invalidated on any style/color/size changeover signal from the line
  controller.
- **Threshold configuration:** `inference.score_threshold`, `nms_iou_threshold`, and
  `anomaly_embedding_threshold` must be **externally configurable per style/line**, not
  hardcoded — different garment styles will have different natural FAR/FRR operating points
  (§10).
- **Failure handling:** if alignment score (from ORB+RANSAC) is below a sanity threshold,
  do **not** feed the pair into the Siamese model at all — route to manual review instead;
  a badly-misaligned pair will produce a meaningless (and potentially high-confidence-wrong)
  comparison. Similarly, the upstream T-shirt Detection gate (§13) should short-circuit
  before this model runs at all on a non-garment/empty frame — this model no longer exposes
  its own `valid_tshirt` output (§3.5, removed).

---

## 15. OpenVINO deployment (implementation: `export/export_openvino.py`)

```
PyTorch (ExportWrapper around SiameseInstanceSegmentation)
   │  torch.onnx.export, opset 17, static shapes
   ▼
ONNX  (dense per-point outputs: cls_logits, box_reg, centerness, mask_coeff, points,
        prototypes, both embeddings — NMS and mask-crop NOT in the graph)
   │  openvino.convert_model / ovc
   ▼
OpenVINO IR (.xml/.bin), FP16 by default
   │  optional: nncf.quantize with a real calibration DataLoader
   ▼
OpenVINO Runtime (CPU / iGPU / VPU inference)
```

**Design choice, and why NMS/mask-assembly are excluded from the exported graph:**
`torchvision.ops.batched_nms` has a data-dependent output size, which some OpenVINO IR
versions handle inconsistently across releases; and while `affine_grid`/`grid_sample`
(used for RoI mask cropping) *do* export to standard ONNX ops, running that crop for
10⁴-10⁵ candidate locations (before NMS reduces the count) rather than for the handful of
surviving instances (after NMS) is pure wasted compute. **[REC]** Export only the dense
per-point predictions; do NMS + mask crop as a small, fast NumPy/OpenCV post-processing
step in the application layer, using the exact same math as
`SiameseInstanceSegmentation.postprocess` (kept in sync deliberately — same formulas, just
outside the traced graph).

- **Dynamic shapes:** `configs/config.yaml: export.dynamic_axes = false` — fixed input size
  is preferred for an industrial line where the camera resolution/crop is fixed anyway;
  fixed shapes let OpenVINO fully constant-fold and optimize the graph, and remove an entire
  class of export/runtime shape-mismatch bugs. Only enable dynamic batch if genuinely
  batching multiple stations' frames together.
- **Unsupported ops to watch for:** GroupNorm (used throughout instead of BatchNorm, since
  batch sizes can be small) is supported in recent ONNX opsets/OpenVINO but verify against
  the specific OpenVINO version pinned for deployment; the custom `Scale` learnable-scalar
  module is just an elementwise multiply, no risk; `F.grid_sample`/`affine_grid` need
  opset ≥ 16 (config already requests opset 17).
- **Precision:**
  - **FP32**: baseline correctness reference only, not for deployment.
  - **FP16**: **[REC] default deployment precision** — negligible accuracy loss for
    conv-heavy detection/segmentation networks, ~2x memory/bandwidth reduction,
    well-supported on both CPU and any GPU/VPU target.
  - **INT8 (post-training quantization via NNCF, `export/export_openvino.py:quantize_int8`)**:
    highest throughput, but **must** be validated with a representative calibration set
    (a few hundred real reference/input pairs spanning the full style/color/lighting
    distribution, not a synthetic or narrow set) and **re-measured on the full evaluation
    suite (§10)** before trusting it — quantization error tends to concentrate on exactly
    the fine boundary/low-contrast regions (hairline cracks, subtle color deviation) that
    matter most for this task, so a global "accuracy barely changed" number is not
    sufficient; check the affected defect classes specifically.
- **Accuracy validation after conversion:** run the *identical* `evaluation/metrics.py`
  suite against OpenVINO IR outputs (via the OpenVINO Python API) on the held-out test set,
  and diff per-class AP/recall against the PyTorch FP32 numbers — treat any per-class AP
  drop beyond a pre-agreed tolerance (e.g. 1-2 AP points) as a blocking regression, not a
  rounding error, before shipping INT8 to the line.

---

## 16. Explainability / visualization

The UI should render, per inspected frame:

```
┌───────────────────┬───────────────────┬──────────────────────────────┐
│  Reference image   │  Inspected image   │  Overlay: boxes + masks +    │
│  (golden, aligned)  │  (as captured)     │  class label + confidence    │
└───────────────────┴───────────────────┴──────────────────────────────┘
   embedding_distance = 0.41 (threshold 0.35 -> anomalous)
   Instance 1: print_crack   conf 0.96   bbox (640,300,820,390)
   Instance 2: spot_stain    conf 0.91   bbox (1200,900,1260,955)
```

**Feature-difference map visualization** (for engineers debugging *why* the model fired):
render the per-level `cos_map` computed inside `FeatureComparison` / `LevelFusion` (the
1-channel per-pixel cosine-similarity map already computed in the forward pass — no extra
inference cost, just expose it) as a heatmap upsampled back to image resolution and overlaid
on the inspected image. Low-similarity regions are literally "where in feature space this
image disagrees with the reference," which is the most direct, cheapest-to-compute
explanation available — far cheaper than a post-hoc Grad-CAM pass, and specific to the
Siamese comparison mechanism itself rather than a generic classifier saliency map. **[REC]**
Surface this as an optional "explain" toggle in the inspection UI rather than always-on,
since it's mainly useful for engineering debugging/audits, not routine line operation.

---

## 17. PyTorch implementation map

| File | Contents |
|---|---|
| `configs/config.yaml` | All architecture/loss/training/export hyperparameters. |
| `models/backbone.py` | `ResNetBackbone`, `MobileNetV3Backbone`, `build_backbone()`. |
| `models/siamese_encoder.py` | `FPN`, `GeMPooling`, `SiameseEncoder` (weight-shared two-branch forward). |
| `models/feature_comparison.py` | `LevelFusion`, `CrossAttentionFusion`, `FeatureComparison` (config-switchable comparison mode). |
| `models/segmentation_head.py` | `ConvTower`, `Scale`, `ProtoNet`, `SegmentationHead`, `generate_points`, `assemble_instance_masks`, `roi_crop_resize`. |
| `models/siamese_instance_segmentation.py` | Top-level model, FCOS-style `assign_targets`, inference `postprocess` (decode + NMS + mask assembly). |
| `losses/detection_loss.py` | `sigmoid_focal_loss`, `ciou_loss`, `DetectionLoss`. |
| `losses/segmentation_loss.py` | `dice_loss`, `tversky_loss`, `SegmentationLoss`. |
| `losses/contrastive_loss.py` | `ContrastiveLoss`, `CosineEmbeddingLossWrapper`, `TripletLoss`, `InfoNCELoss`, `build_siamese_loss`. |
| `losses/total_loss.py` | `TotalLoss` — combines all of the above with configured weights, per-image target assignment loop. |
| `datasets/dataset.py` | `SiameseDefectDataset` (custom-JSON loader, polygon/RLE decode), `siamese_collate_fn`. |
| `datasets/transforms.py` | `SiamesePairTransform` and the shared/independent augmentation split (§9). |
| `datasets/pair_sampler.py` | `HardNegativePool`, `SiamesePairBatchSampler`, `split_positive_negative`. |
| `training/scheduler.py` | `WarmupCosineScheduler`. |
| `training/train.py` | Full training loop (AMP, grad accumulation, EMA, hard-negative refresh, checkpointing, early stopping). |
| `training/validate.py` | Full-decode validation loop producing mask mAP / AP50 / AP75. |
| `inference/predict.py` | Single-image-pair inference CLI, structured instance output, anomaly flag. |
| `evaluation/metrics.py` | All metrics from §10. |
| `export/export_openvino.py` | `ExportWrapper`, ONNX export, OpenVINO IR conversion, NNCF INT8 quantization. |

**Environment note:** this sandbox does not have PyTorch/torchvision installed, so the code
above has been syntax-checked (`python -m py_compile`, all files pass) and manually
traced for shape/logic correctness, but has **not** been executed end-to-end against real
tensors or real data in this session. Before trusting it for training, run a smoke test:
instantiate `SiameseInstanceSegmentation(cfg)`, forward a random `(1,3,800,800)` pair
through it, and confirm `outputs["levels"][lvl]["cls_logits"].shape` matches the expected
`(1, num_classes, H_lvl, W_lvl)` for each level, then run one `TotalLoss` step against a
synthetic target with a couple of random boxes/masks.

### 17.1 Training pseudocode (mirrors `training/train.py`)

```
model = SiameseInstanceSegmentation(cfg)
ema_model = deepcopy(model).frozen()
criterion = TotalLoss(cfg)
optimizer = AdamW(model.parameters(), lr, weight_decay)
scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
hard_pool = HardNegativePool(negative_indices)

for epoch in range(epochs):
    if epoch >= hard_negative_start: hard_pool.refresh(model.encoder, train_set, device)
    for step, (reference, input, targets) in enumerate(train_loader):
        with autocast():
            outputs = model(reference, input)
            losses = criterion(outputs, targets)
        scaler.scale(losses.total / accum_steps).backward()
        if (step+1) % accum_steps == 0:
            clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            scheduler.step()
            ema_model = ema_decay * ema_model + (1-ema_decay) * model
    metrics = validate(ema_model, val_loader)
    if metrics.mask_mAP > best: save_checkpoint(); best = metrics.mask_mAP
    else: patience_counter += 1; if patience_counter > patience: break
```

### 17.2 Inference (mirrors `inference/predict.py`)

```
model = load(checkpoint).eval()
reference_t, input_t = preprocess(reference_img), preprocess(input_img)
outputs = model(reference_t, input_t)
instances = model.postprocess(outputs, image_size, score_thresh, nms_iou, max_dets)
for inst in instances: paste_mask_on_image(inst.mask_logits, inst.bbox, image_size)
embedding_distance = ||outputs.reference_embedding - outputs.inspected_embedding||
unknown_defect_suspected = embedding_distance > threshold and len(instances) == 0
```

---

## 18. Recommended baseline vs. recommended final architecture

**Recommended baseline (build and validate first, cheapest to get right):**
Siamese ResNet-34 + FPN, **abs-diff-only** feature comparison (option 2 alone, no residual
fusion, no cross-attention), single-scale-aware but otherwise plain FCOS head, BCE-only mask
loss, contrastive Siamese loss. This is intentionally the simplest version of the full
architecture — every component that section 2 recommends adding on top (concat+cosine
fusion, cross-attention at coarse levels, Dice+BCE mask loss, ResNet-50) should be justified
by an ablation (§22) showing it beats this baseline by a meaningful margin, not assumed.

**Recommended final architecture (this document's default, `configs/config.yaml`):**
Siamese ResNet-50 + FPN (P3-P7) → per-level `concat_diff_residual` fusion at P3-P5 +
cross-attention fusion at P6-P7 → FCOS-style anchor-free dense head (focal cls + CIoU box +
centerness) + YOLACT/CondInst-style shared-prototype mask branch (BCE+Dice) → separate
T-shirt-validity head → contrastive Siamese embedding loss on GeM-pooled P5 features.

---

## 19. Comparison with Mask R-CNN, YOLO-Seg, Mask2Former, Anomalib

| | This design (Siamese FCOS+Proto) | Siamese Mask R-CNN | Siamese YOLO-Seg (e.g. YOLOv8-seg) | Siamese Mask2Former | Anomalib (PatchCore/PaDiM etc.) |
|---|---|---|---|---|---|
| Stage count | 1 | 2 (RPN + RoI heads) | 1 | 1 (but heavy transformer decoder) | N/A — not a detector, patch-level anomaly scoring |
| Needs per-instance labels | Yes | Yes | Yes | Yes | **No** — typically trained on "good" images only |
| Multi-class defect output | Yes | Yes | Yes | Yes | No — anomaly score only, no class label |
| Expected mask boundary quality | Good | Good-Very good | Good | **Best** (general vision benchmarks) | N/A (coarse anomaly heatmap, not instance masks) |
| Latency (relative) | 1x (reference) | ~1.3-1.8x (two-stage RoI heads) | ~0.8-1x (comparable single-stage family) | ~2-4x (decoder + pixel decoder + attention rounds) | Fast per-patch, but a different pipeline (memory bank lookup, not a forward net) |
| ONNX/OpenVINO export maturity | High (no RoIAlign, no learned NMS) | Medium (RoIAlign supported but adds export complexity; dynamic proposal count) | High | Medium-Low (deformable/masked attention ops less uniformly supported across opsets) | High for the backbone; the memory-bank/kNN lookup step is usually implemented outside the exported graph anyway |
| Unknown-defect capability | Via embedding distance (§12), heuristic | Same option available if a Siamese embedding head is added | Same option available if a Siamese embedding head is added | Same option available if a Siamese embedding head is added | **Purpose-built for this** — its whole design point is unsupervised anomaly localization |
| Best-fit role in this system | **Primary defect detector/segmenter/classifier** | Accuracy fallback if the single-stage head underperforms on boundary quality | Viable near-equivalent alternative to the recommended head; worth ablating | Accuracy ceiling reference, not the deployment default given latency/export cost | **Complementary, run in parallel** for unsupervised surface-anomaly cross-check (§13), not a replacement |

**Bottom line:** Mask2Former is the accuracy ceiling in general instance segmentation
literature but is the wrong tool here given the export/latency requirements and the
relatively simple, near-planar, well-lit nature of garment defects. Mask R-CNN is a
credible, lower-risk fallback if the single-stage head's mask quality proves insufficient.
Anomalib is not a competitor to this system at all — it answers a different question
("is anything unusual here") and should run alongside it, feeding the unknown-defect
decision (§12/§13), not replacing the supervised classifier.

---

## 20. Expected advantages and limitations of the Siamese approach

**Advantages [REC — expected, pending §22 validation]:**
- Directly encodes the actual production inspection process (compare-to-golden), rather
  than asking a plain detector to memorize "what a crack looks like" independent of
  garment-specific print context — this should particularly help distinguish
  **print misalignment / ghosting / color deviation**, which are inherently *relative*
  defects that a reference-free detector cannot define at all (there is no such thing as
  "misaligned" without something to be misaligned *relative to*).
- The embedding-distance side-channel gives an unknown-defect signal a plain supervised
  detector structurally cannot produce (§12).
- Weight sharing halves backbone parameters vs. two independent per-branch towers and
  guarantees comparable feature spaces.

**Limitations / open risks [explicitly not yet validated]:**
- **A plain (non-Siamese) detector trained directly on defect labels might perform just as
  well or better on the classes that are *not* inherently relative** (`print_crack`,
  `fabric_damage`, `spot_stain` arguably have absolute visual signatures a reference-free
  detector can learn directly) — the Siamese design adds real cost (two forward passes,
  the comparison module, reference-image logistics/storage, reference-selection failure
  modes) that is only worth it if it measurably helps. **This must be tested, not
  assumed** — see §22.
- Reference-selection errors (wrong style/color/size looked up) become a *new* failure mode
  a plain detector doesn't have.
- The embedding-distance anomaly signal is coarse (global P5 pooling) and likely has poor
  recall on small/local unknown defects — §12 already flags this.
- Residual misalignment (even after ORB+RANSAC) can leak into the comparison signal as
  spurious "differences", risking false rejects specifically at print edges/seams — the
  cross-attention/residual-fusion design (§3.3) is an attempt to mitigate this, not a
  guarantee.

---

## 21. Failure cases and mitigation

| Failure case | Mitigation |
|---|---|
| Wrong reference selected (upstream lookup error, §11) | This model has no way to detect it internally now that reference selection is fully upstream and unvalidated by any garment-attribute check; flag mismatches at the upstream reference-resolution step, before the annotation/inference record is ever written. |
| Reference image itself has degraded (dust on lens, faded over time) | Reference approval/retirement tracking (approved-date, golden/retired status) is no longer part of this schema (§5.2); if this becomes a requirement again, track it in whatever upstream system resolves `reference_image`, not in the defect annotation. |
| Alignment failure (ORB+RANSAC low confidence) | Route to manual review before the Siamese model runs at all (§14) — never feed a low-confidence-aligned pair into the comparison. |
| Fabric wrinkle/deformation misclassified as `fabric_damage` | Ensure training "good" set includes realistic wrinkled captures (§6.2). |
| Lighting drift across shifts/days | Independent photometric augmentation (§9) + hard-negative mining (§6.3) specifically targets this; monitor FRR trend over time in production as an early-warning signal of augmentation coverage gaps. |
| Rare defect class (`fabric_damage`, `other_anomaly`) has too few examples | Per-class metrics (§10) surface this rather than hiding it in a macro average; consider focal-loss `α` re-weighting per class, or the (flagged, not default) synthetic copy-paste augmentation (§9) as a deliberate, tracked mitigation. |
| Truly novel defect type never seen in training | Embedding-distance anomaly flag (§12) + Anomalib cross-check (§13); explicitly **not** solved by the supervised classifier alone. |
| INT8 quantization silently degrades subtle-defect classes | Mandatory per-class re-validation after quantization (§15), not just a global accuracy check. |
| Model over-confident on out-of-distribution input (not a T-shirt, empty frame, wrong garment type) | The upstream T-shirt Detection gate (§13/§14) — not a head of this model — should short-circuit the rest of the pipeline before this model ever runs. |

---

## 22. Recommended experiment / ablation plan

The goal here is to answer, with evidence, the question this whole document is not allowed
to assume the answer to: **does the Siamese (reference-comparison) design actually improve
defect detection over a plain single-image detector, and which of the specific design
choices in §2-§7 are load-bearing?**

1. **Siamese vs. non-Siamese baseline.** Train (a) this architecture and (b) an identical
   backbone/FPN/head with the comparison module removed (detector sees only the inspected
   image, no reference branch at all) on the same data/splits. Compare per-class AP,
   especially on the *inherently relative* classes (`print_misalignment`, `ghosting`,
   `color_deviation`) where the hypothesis (§20) predicts the largest gap, vs. the
   *absolute-signature* classes (`print_crack`, `fabric_damage`) where the gap is expected
   to be smaller or absent. **This single experiment is the most important one in this
   entire plan** — everything else is refinement conditional on this showing a real gain.
2. **Feature comparison mode ablation** (§2.1, §3.3): hold everything else fixed, sweep
   `model.comparison.mode` across `abs_diff`, `concat`, `cosine`, `concat_diff_residual`,
   `cross_attention` (config switch already implemented, no code change needed). Measure
   mask mAP and, separately, FRR/FAR at a fixed operating threshold.
3. **Backbone ablation**: ResNet-34 vs. ResNet-50 vs. MobileNetV3-Large, holding the rest of
   the architecture fixed — quantify the actual accuracy/latency trade-off *on this specific
   dataset* rather than trusting general ImageNet-benchmark intuition.
4. **Mask loss ablation**: BCE-only vs. BCE+Dice (default) vs. Tversky (α<β, recall-biased)
   — report FAR/FRR specifically, since this is the knob most directly tied to the
   business-critical "don't ship defective garments" objective.
5. **Siamese loss type ablation**: contrastive (default) vs. cosine-embedding vs. triplet —
   measure both mask mAP (does it help the main task) and, separately, the quality of the
   embedding-distance anomaly signal (§12) on a held-out set of defect types *excluded*
   from supervised training, as the actual test of "unknown defect detection" capability.
6. **Hard-negative mining ablation**: with vs. without (§6.3), measured specifically as FRR
   on a validation slice deliberately over-sampled for lighting/wrinkle/alignment variation
   — this is the mechanism's specific target, so the ablation should use a matching test
   slice, not just the overall validation set.
7. **Reference strategy ablation** (§11): single fixed reference vs. multiple approved
   references (min-distance-wins) vs. a learned per-style prototype embedding — measure
   both accuracy and operational simplicity (storage, lookup latency, maintenance burden).
8. **Quantization accuracy check** (§15): FP32 vs FP16 vs INT8, full per-class metric suite,
   not just an aggregate — required before any precision below FP16 is deployed.

Each ablation should hold all other configuration fixed and change exactly one knob — every
one of these is already exposed as a single `configs/config.yaml` field specifically so this
plan can be executed without code changes.

---

## 23. Deliverables checklist

1. High-level architecture — §2, §14 (hybrid pipeline diagram)
2. Detailed model architecture — §3
3. Tensor/data flow diagram — §4
4. Dataset structure — §5.1
5. Example annotation — `data/annotations/example_annotation.json`, §5.2
6. Pair-generation strategy — §6
7. Loss equations — §7
8. Training strategy — §8
9. PyTorch implementation — §17 (file map), all files under `models/`, `losses/`, `datasets/`, `training/`, `inference/`, `evaluation/`, `export/`
10. Training pseudocode — §17.1
11. Inference code — `inference/predict.py`, §17.2
12. Evaluation methodology — §10
13. OpenVINO deployment — §15
14. Industrial optimization strategy — §14
15. Failure cases and mitigation — §21
16. Recommended baseline model — §18
17. Recommended final architecture — §18
18. Comparison with Mask R-CNN / YOLO-Seg / Mask2Former / Anomalib — §19
19. Advantages/limitations of the Siamese approach — §20
20. Experiment plan to validate the Siamese approach — §22
