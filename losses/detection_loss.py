"""Detection-side losses: focal classification, CIoU box regression, BCE centerness.

Equations (see DESIGN.md section 6):

  Focal Loss:      FL(p_t) = -alpha_t (1 - p_t)^gamma log(p_t)
  CIoU Loss:       L_ciou = 1 - IoU + rho^2(b, b_gt)/c^2 + alpha * v
                     v = (4/pi^2) * (arctan(w_gt/h_gt) - arctan(w/h))^2
  Centerness BCE:  L_ctr = BCE(sigmoid(pred), sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b)))
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor,
                        alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """logits, targets: (N, C) one-hot. Returns sum over all elements."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.sum()


def ciou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor,
              eps: float = 1e-7) -> torch.Tensor:
    """pred_boxes, target_boxes: (N, 4) xyxy. Returns per-box loss, shape (N,)."""
    px1, py1, px2, py2 = pred_boxes.unbind(-1)
    tx1, ty1, tx2, ty2 = target_boxes.unbind(-1)

    pw, ph = (px2 - px1).clamp(min=eps), (py2 - py1).clamp(min=eps)
    tw, th = (tx2 - tx1).clamp(min=eps), (ty2 - ty1).clamp(min=eps)

    inter_x1, inter_y1 = torch.max(px1, tx1), torch.max(py1, ty1)
    inter_x2, inter_y2 = torch.min(px2, tx2), torch.min(py2, ty2)
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = pw * ph + tw * th - inter + eps
    iou = inter / union

    cx1, cy1 = torch.min(px1, tx1), torch.min(py1, ty1)
    cx2, cy2 = torch.max(px2, tx2), torch.max(py2, ty2)
    c2 = (cx2 - cx1).pow(2) + (cy2 - cy1).pow(2) + eps

    pred_cx, pred_cy = (px1 + px2) / 2, (py1 + py2) / 2
    tgt_cx, tgt_cy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
    rho2 = (pred_cx - tgt_cx).pow(2) + (pred_cy - tgt_cy).pow(2)

    v = (4 / (torch.pi ** 2)) * (torch.atan(tw / th) - torch.atan(pw / ph)).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    return 1 - iou + rho2 / c2 + alpha * v


class DetectionLoss(nn.Module):
    def __init__(self, num_classes: int, focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(self, levels: dict, points: dict, targets: dict) -> dict:
        """levels/points/targets are per-level dicts for a single image; the
        caller (total_loss.py) loops over the batch and accumulates."""
        cls_losses, box_losses, ctr_losses, n_pos_total = [], [], [], 0

        for lvl, pred in levels.items():
            t = targets[lvl]
            labels, box_t, ctr_t = t["labels"], t["box_targets"], t["centerness_targets"]
            pos_mask = labels >= 0
            n_pos = pos_mask.sum().clamp(min=1)
            n_pos_total += pos_mask.sum().item()

            cls_logits = pred["cls_logits"].permute(1, 2, 0).reshape(-1, self.num_classes)
            one_hot = torch.zeros_like(cls_logits)
            if pos_mask.any():
                one_hot[pos_mask, labels[pos_mask]] = 1.0
            cls_losses.append(sigmoid_focal_loss(cls_logits, one_hot,
                                                  self.focal_alpha, self.focal_gamma) / n_pos)

            if pos_mask.any():
                box_reg = pred["box_reg"].permute(1, 2, 0).reshape(-1, 4)
                pts = points[lvl]
                x, y = pts[pos_mask, 0], pts[pos_mask, 1]
                l, t_, r, b = box_reg[pos_mask].unbind(-1)
                pred_boxes = torch.stack([x - l, y - t_, x + r, y + b], dim=1)
                l_t, t_t, r_t, b_t = box_t[pos_mask].unbind(-1)
                tgt_boxes = torch.stack([x - l_t, y - t_t, x + r_t, y + b_t], dim=1)
                box_losses.append(ciou_loss(pred_boxes, tgt_boxes).sum() / n_pos)

                ctr_logits = pred["centerness"].permute(1, 2, 0).reshape(-1)[pos_mask]
                ctr_losses.append(F.binary_cross_entropy_with_logits(
                    ctr_logits, ctr_t[pos_mask], reduction="sum") / n_pos)

        zero = next(iter(levels.values()))["cls_logits"].new_zeros(())
        return {
            "cls_loss": torch.stack(cls_losses).sum() if cls_losses else zero,
            "box_loss": torch.stack(box_losses).sum() if box_losses else zero,
            "centerness_loss": torch.stack(ctr_losses).sum() if ctr_losses else zero,
            "num_positive": n_pos_total,
        }
