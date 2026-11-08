"""Validation loop: runs full inference (decode + NMS) and computes the
metrics used for early stopping / checkpoint selection.

With `inference.tile` set in the config (small defects on large images, e.g.
configs/pcb.yaml) each full-resolution val pair is evaluated with
sliding-window tiling (inference/tiling.py); the val loader must then use
batch_size 1, since images differ in size.

Returned metrics: AP50 / AP75 (pooled over classes), box_mAP = mean(AP50,
AP75), mAP50 (mean of per-class AP50) and AP50_<class> per class.
"""
from __future__ import annotations

from typing import Dict, List

import torch
from tqdm import tqdm

from evaluation.metrics import box_map, match_predictions_to_gt
from inference.tiling import predict_tiled


def _per_class_matches(preds: List[Dict], gt_boxes, gt_labels, num_classes: int,
                       iou: float) -> List[Dict]:
    out = []
    for c in range(num_classes):
        p = [r for r in preds if r["class_id"] == c]
        g = [b for b, l in zip(gt_boxes, gt_labels) if l == c]
        out.append(match_predictions_to_gt([r["bbox"] for r in p], [c] * len(p),
                                           [r["confidence"] for r in p], g, [c] * len(g), iou))
    return out


@torch.no_grad()
def validate(model, val_loader, device, cfg: dict) -> Dict[str, float]:
    model.eval()
    icfg = cfg["inference"]
    tile_cfg = icfg.get("tile")
    class_names = cfg["data"]["defect_classes"]
    num_classes = len(class_names)
    matches_50, matches_75 = [], []
    per_class_50 = [[] for _ in range(num_classes)]

    for batch in tqdm(val_loader, desc="validate", leave=False):
        reference = batch["reference"].to(device)
        inputs = batch["input"].to(device)
        targets = batch["targets"]

        if tile_cfg:
            per_image = [predict_tiled(model, reference[b], inputs[b], tile_cfg["size"],
                                       tile_cfg["overlap"], icfg["score_threshold"],
                                       icfg["nms_iou_threshold"], icfg["max_detections"],
                                       batch_size=tile_cfg.get("batch_size", 8))
                         for b in range(len(targets))]
        else:
            outputs = model(reference, inputs)
            h, w = inputs.shape[-2:]
            per_image = []
            for b in range(len(targets)):
                single_out = {
                    "levels": {lvl: {k: v[b:b + 1] for k, v in pred.items()}
                               for lvl, pred in outputs["levels"].items()},
                    "points": outputs["points"],
                }
                per_image.append(model.postprocess(
                    single_out, (h, w), score_thresh=icfg["score_threshold"],
                    nms_iou=icfg["nms_iou_threshold"], max_dets=icfg["max_detections"]))

        for results, target in zip(per_image, targets):
            pred_boxes = [r["bbox"] for r in results]
            pred_labels = [r["class_id"] for r in results]
            pred_scores = [r["confidence"] for r in results]
            gt_boxes = target["boxes"].cpu().tolist()
            gt_labels = target["labels"].cpu().tolist()

            matches_50.append(match_predictions_to_gt(pred_boxes, pred_labels, pred_scores,
                                                      gt_boxes, gt_labels, iou_thresh=0.5))
            matches_75.append(match_predictions_to_gt(pred_boxes, pred_labels, pred_scores,
                                                      gt_boxes, gt_labels, iou_thresh=0.75))
            for c, m in enumerate(_per_class_matches(results, gt_boxes, gt_labels,
                                                     num_classes, 0.5)):
                per_class_50[c].append(m)

    ap50 = box_map(matches_50)
    ap75 = box_map(matches_75)
    class_ap50 = {f"AP50_{name}": box_map(per_class_50[c]) for c, name in enumerate(class_names)}
    return {
        "box_mAP": (ap50 + ap75) / 2,
        "AP50": ap50,
        "AP75": ap75,
        "mAP50": sum(class_ap50.values()) / max(1, num_classes),
        **class_ap50,
    }
