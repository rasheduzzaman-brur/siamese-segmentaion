"""Image-level classification losses (T-shirt validity, optional garment attributes).

Per-instance defect classification uses focal loss and lives in
detection_loss.py (it is a dense, per-location prediction, not a
whole-image classification, and shares the same tensors as the box head).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TshirtClassificationLoss(nn.Module):
    """Binary BCE for the valid_tshirt gate. Class imbalance here is usually
    mild (most captured frames are valid T-shirts) so plain BCE suffices;
    switch to focal if the false-accept rate on non-tshirt frames is high."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, targets.float())


class LabelSmoothingCE(nn.Module):
    """Used for any optional auxiliary closed-set classification head
    (e.g. GarmentAttributeHead) where label smoothing reduces
    over-confidence on a fairly small, clean, closed-set label space."""

    def __init__(self, num_classes: int, smoothing: float = 0.05):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return torch.sum(-true_dist * log_probs, dim=-1).mean()
