"""Linear-warmup + cosine-decay LR scheduler (per training step, not per epoch)."""
from __future__ import annotations

import math
import torch


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int,
                 total_steps: int, min_lr_ratio: float = 0.01, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            factor = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            factor = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))
        return [base_lr * factor for base_lr in self.base_lrs]
