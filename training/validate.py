"""Validation loop: runs full inference (decode + NMS + mask assembly) and
computes the metrics used for early stopping / checkpoint selection.
"""
from __future__ import annotations

from typing import Dict

import torch
from tqdm import tqdm

from evaluation.metrics import mask_map, match_predictions_to_gt


@torch.no_grad()
def validate(model, val_loader, device, cfg: dict) -> Dict[str, float]:
    model.eval()
    matches_50, matches_75 = [], []
    tshirt_correct, tshirt_total = 0, 0

    for batch in tqdm(val_loader, desc="validate", leave=False):
        reference = batch["reference"].to(device)
        inputs = batch["input"].to(device)
        targets = batch["targets"]

        outputs = model(reference, inputs)
        tshirt_pred = (torch.sigmoid(outputs["tshirt_logit"]) > 0.5).float().cpu()
        tshirt_gt = torch.stack([t["tshirt_valid"] for t in targets])
        tshirt_correct += (tshirt_pred == tshirt_gt).sum().item()
        tshirt_total += len(targets)

        for b, target in enumerate(targets):
            single_out = {
                "levels": {lvl: {k: v[b:b + 1] for k, v in pred.items()}
                           for lvl, pred in outputs["levels"].items()},
                "prototypes": outputs["prototypes"][b:b + 1],
                "points": outputs["points"],
            }
            h, w = inputs.shape[-2:]
            results = model.postprocess(
                single_out, (h, w),
                score_thresh=cfg["inference"]["score_threshold"],
                nms_iou=cfg["inference"]["nms_iou_threshold"],
                max_dets=cfg["inference"]["max_detections"],
            )
            pred_masks = [torch.sigmoid(r["mask_logits"]) for r in results]
            pred_labels = [r["class_id"] for r in results]
            pred_scores = [r["confidence"] for r in results]

            gt_masks = list(target["masks"].cpu())
            gt_labels = target["labels"].cpu().tolist()

            matches_50.append(match_predictions_to_gt(pred_masks, pred_labels, pred_scores,
                                                        gt_masks, gt_labels, iou_thresh=0.5))
            matches_75.append(match_predictions_to_gt(pred_masks, pred_labels, pred_scores,
                                                        gt_masks, gt_labels, iou_thresh=0.75))

    ap50 = mask_map(matches_50)
    ap75 = mask_map(matches_75)
    return {
        "mask_mAP": (ap50 + ap75) / 2,
        "AP50": ap50,
        "AP75": ap75,
        "tshirt_accuracy": tshirt_correct / max(1, tshirt_total),
    }
