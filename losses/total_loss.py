"""Combines all task losses into the single multi-task objective:

    L_total = w_det   * L_detection      (focal cls + CIoU box + BCE centerness;
                                           per-instance defect class is folded into
                                           the focal classification term)
            + w_siam  * L_siamese        (contrastive embedding loss)

Per-task weights are config-driven (configs/config.yaml -> loss.weights) and
should be tuned via uncertainty weighting or a short grid search once all
tasks converge individually -- see DESIGN.md section 6/8 for the balancing
strategy (start with fixed weights that equalize each loss's initial
magnitude, then anneal L_siamese down if it dominates once the encoder has
stabilized).
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from models.siamese_detector import SiameseDefectDetector
from losses.detection_loss import DetectionLoss
from losses.contrastive_loss import build_siamese_loss


class TotalLoss(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        num_classes = len(cfg["data"]["defect_classes"])
        lcfg = cfg["loss"]
        self.weights = lcfg["weights"]

        self.detection_loss = DetectionLoss(
            num_classes, lcfg["detection"]["focal_alpha"], lcfg["detection"]["focal_gamma"])
        self.siamese_loss = build_siamese_loss(lcfg["siamese"])
        self._siamese_kind = lcfg["siamese"]["type"]

    def forward(self, outputs: Dict, targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """targets: list (len B) of dicts with keys
        boxes (M,4), labels (M,), pair_label (scalar)
        """
        batch_size = len(targets)
        device = outputs["reference_embedding"].device

        det_terms = {"cls_loss": [], "box_loss": [], "centerness_loss": []}

        for b in range(batch_size):
            gt = targets[b]
            levels_b = {lvl: {k: v[b] for k, v in pred.items()}
                        for lvl, pred in outputs["levels"].items()}
            assign = SiameseDefectDetector.assign_targets(
                outputs["points"], gt["boxes"].to(device), gt["labels"].to(device))

            det_out = self.detection_loss(levels_b, outputs["points"], assign)
            det_terms["cls_loss"].append(det_out["cls_loss"])
            det_terms["box_loss"].append(det_out["box_loss"])
            det_terms["centerness_loss"].append(det_out["centerness_loss"])

        cls_loss = torch.stack(det_terms["cls_loss"]).mean()
        box_loss = torch.stack(det_terms["box_loss"]).mean()
        centerness_loss = torch.stack(det_terms["centerness_loss"]).mean()

        detection_total = cls_loss + box_loss + centerness_loss

        pair_labels = torch.stack([t["pair_label"].float() for t in targets]).to(device)
        if self._siamese_kind == "infonce":
            siamese_loss = self.siamese_loss(outputs["reference_embedding"],
                                              outputs["inspected_embedding"])
        else:
            siamese_loss = self.siamese_loss(outputs["reference_embedding"],
                                              outputs["inspected_embedding"], pair_labels)

        total = (self.weights["detection"] * detection_total
                 + self.weights["siamese"] * siamese_loss)

        return {
            "total": total, "detection": detection_total, "cls_loss": cls_loss,
            "box_loss": box_loss, "centerness_loss": centerness_loss,
            "siamese_loss": siamese_loss,
        }
