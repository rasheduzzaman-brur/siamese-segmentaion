"""Combines all task losses into the single multi-task objective:

    L_total = w_det   * L_detection      (focal cls + CIoU box + BCE centerness)
            + w_seg   * L_segmentation   (BCE + Dice, matched instances only)
            + w_cls   * L_classification (currently folded into detection focal
                                           loss for per-instance defect class;
                                           reserved here for optional aux heads)
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

from models.segmentation_head import roi_crop_resize, assemble_instance_masks
from models.siamese_instance_segmentation import SiameseInstanceSegmentation
from losses.detection_loss import DetectionLoss
from losses.segmentation_loss import SegmentationLoss
from losses.contrastive_loss import build_siamese_loss


class TotalLoss(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        num_classes = len(cfg["data"]["defect_classes"])
        lcfg = cfg["loss"]
        self.weights = lcfg["weights"]
        self.mask_out_size = cfg["model"]["seg_head"]["mask_out_size"]

        self.detection_loss = DetectionLoss(
            num_classes, lcfg["detection"]["focal_alpha"], lcfg["detection"]["focal_gamma"])
        self.segmentation_loss = SegmentationLoss(
            lcfg["segmentation"]["bce_weight"], lcfg["segmentation"]["dice_weight"])
        self.siamese_loss = build_siamese_loss(lcfg["siamese"])
        self._siamese_kind = lcfg["siamese"]["type"]

    def forward(self, outputs: Dict, targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """targets: list (len B) of dicts with keys
        boxes (M,4), labels (M,), masks (M,H,W), pair_label (scalar)
        """
        batch_size = len(targets)
        device = outputs["reference_embedding"].device

        det_terms = {"cls_loss": [], "box_loss": [], "centerness_loss": []}
        mask_losses = []

        for b in range(batch_size):
            gt = targets[b]
            levels_b = {lvl: {k: v[b] for k, v in pred.items()}
                        for lvl, pred in outputs["levels"].items()}
            assign = SiameseInstanceSegmentation.assign_targets(
                outputs["points"], gt["boxes"].to(device), gt["labels"].to(device))

            det_out = self.detection_loss(levels_b, outputs["points"], assign)
            det_terms["cls_loss"].append(det_out["cls_loss"])
            det_terms["box_loss"].append(det_out["box_loss"])
            det_terms["centerness_loss"].append(det_out["centerness_loss"])

            mask_losses.append(self._mask_loss_single(levels_b, outputs["points"], assign,
                                                        outputs["prototypes"][b], gt, device))

        cls_loss = torch.stack(det_terms["cls_loss"]).mean()
        box_loss = torch.stack(det_terms["box_loss"]).mean()
        centerness_loss = torch.stack(det_terms["centerness_loss"]).mean()
        mask_loss = torch.stack(mask_losses).mean()

        detection_total = cls_loss + box_loss + centerness_loss

        pair_labels = torch.stack([t["pair_label"].float() for t in targets]).to(device)
        if self._siamese_kind == "infonce":
            siamese_loss = self.siamese_loss(outputs["reference_embedding"],
                                              outputs["inspected_embedding"])
        else:
            siamese_loss = self.siamese_loss(outputs["reference_embedding"],
                                              outputs["inspected_embedding"], pair_labels)

        total = (self.weights["detection"] * detection_total
                 + self.weights["segmentation"] * mask_loss
                 + self.weights["siamese"] * siamese_loss)

        return {
            "total": total, "detection": detection_total, "cls_loss": cls_loss,
            "box_loss": box_loss, "centerness_loss": centerness_loss,
            "mask_loss": mask_loss, "siamese_loss": siamese_loss,
        }

    def _mask_loss_single(self, levels_b, points, assign, prototypes, gt, device) -> torch.Tensor:
        coeffs, gt_idx = [], []
        for lvl, pred in levels_b.items():
            labels = assign[lvl]["labels"]
            pos = labels >= 0
            if pos.any():
                mc = pred["mask_coeff"].permute(1, 2, 0).reshape(-1, pred["mask_coeff"].shape[0])
                coeffs.append(mc[pos])
                gt_idx.append(assign[lvl]["matched_gt_idx"][pos])

        if not coeffs or gt["boxes"].numel() == 0:
            return prototypes.new_zeros(())

        coeffs = torch.cat(coeffs)
        gt_idx = torch.cat(gt_idx)
        boxes = gt["boxes"].to(device)[gt_idx]
        masks = gt["masks"].to(device)[gt_idx]

        pred_mask_logits = assemble_instance_masks(prototypes, coeffs, boxes, self.mask_out_size)
        target_masks = roi_crop_resize(masks, boxes, self.mask_out_size, source_stride=1.0)
        return self.segmentation_loss(pred_mask_logits, (target_masks > 0.5).float())
