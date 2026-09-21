"""Validation loop: runs full inference (decode + NMS) and computes the
metrics used for early stopping / checkpoint selection.
"""
from __future__ import annotations

from typing import Dict

import torch
from tqdm import tqdm

from evaluation.metrics import box_map, match_predictions_to_gt


@torch.no_grad()
def validate(model, val_loader, device, cfg: dict) -> Dict[str, float]:
    model.eval()
    matches_50, matches_75 = [], []

    for batch in tqdm(val_loader, desc="validate", leave=False):
        reference = batch["reference"].to(device)
        inputs = batch["input"].to(device)
        targets = batch["targets"]

        outputs = model(reference, inputs)

        for b, target in enumerate(targets):
            single_out = {
                "levels": {lvl: {k: v[b:b + 1] for k, v in pred.items()}
                           for lvl, pred in outputs["levels"].items()},
                "points": outputs["points"],
            }
            h, w = inputs.shape[-2:]
            results = model.postprocess(
                single_out, (h, w),
                score_thresh=cfg["inference"]["score_threshold"],
                nms_iou=cfg["inference"]["nms_iou_threshold"],
                max_dets=cfg["inference"]["max_detections"],
            )
            pred_boxes = [r["bbox"] for r in results]
            pred_labels = [r["class_id"] for r in results]
            pred_scores = [r["confidence"] for r in results]

            gt_boxes = target["boxes"].cpu().tolist()
            gt_labels = target["labels"].cpu().tolist()

            matches_50.append(match_predictions_to_gt(pred_boxes, pred_labels, pred_scores,
                                                        gt_boxes, gt_labels, iou_thresh=0.5))
            matches_75.append(match_predictions_to_gt(pred_boxes, pred_labels, pred_scores,
                                                        gt_boxes, gt_labels, iou_thresh=0.75))

    ap50 = box_map(matches_50)
    ap75 = box_map(matches_75)
    return {
        "box_mAP": (ap50 + ap75) / 2,
        "AP50": ap50,
        "AP75": ap75,
    }
