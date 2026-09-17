"""Instance mask loss: BCE + Dice, computed only for matched positive instances.

  L_bce  = -[y log(p) + (1-y) log(1-p)]                         (pixel-wise)
  L_dice = 1 - (2 * sum(p*y) + eps) / (sum(p) + sum(y) + eps)     (per-instance)

Dice directly optimizes overlap and is far less sensitive than plain BCE to
the heavy foreground/background imbalance inside a small defect mask crop;
BCE is kept alongside it for stable, well-scaled gradients early in training
(pure Dice loss is known to be unstable when the prediction starts near 0).
Tversky loss (a generalization of Dice with asymmetric FP/FN weights) is
noted as an ablation candidate for severity-sensitive tuning -- see
DESIGN.md section 6 -- but is not the default.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """logits, targets: (N, H, W). Returns per-instance loss, shape (N,)."""
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    return 1 - (2 * intersection + eps) / (union + eps)


def tversky_loss(logits: torch.Tensor, targets: torch.Tensor,
                  alpha: float = 0.3, beta: float = 0.7, eps: float = 1.0) -> torch.Tensor:
    """alpha weights false positives, beta weights false negatives.
    beta > alpha biases towards recall -- useful when missed defects (false
    accepts) are costlier than over-segmentation."""
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    tp = (probs * targets).sum(dim=1)
    fp = (probs * (1 - targets)).sum(dim=1)
    fn = ((1 - probs) * targets).sum(dim=1)
    return 1 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)


class SegmentationLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, mask_logits: torch.Tensor, mask_targets: torch.Tensor) -> torch.Tensor:
        """Both inputs already resized to the same (N, out_size, out_size)."""
        if mask_logits.shape[0] == 0:
            return mask_logits.new_zeros(())
        bce = F.binary_cross_entropy_with_logits(mask_logits, mask_targets, reduction="mean")
        dice = dice_loss(mask_logits, mask_targets).mean()
        return self.bce_weight * bce + self.dice_weight * dice
