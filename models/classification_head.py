"""Image-level classification heads.

Two distinct responsibilities, kept separate from the dense per-instance
defect classification that already lives in SegmentationHead.cls_pred:

  1. TshirtValidityHead  - "is this even a T-shirt, correctly framed/aligned?"
                           A cheap gate that runs before the expensive
                           segmentation heads are trusted (see DESIGN.md
                           section 14, industrial pipeline).
  2. GarmentAttributeHead - optional auxiliary head (style/color/size) used
                           only to sanity-check that the correct reference
                           was selected; not part of the core defect task.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TshirtValidityHead(nn.Module):
    """Binary "valid T-shirt, properly presented to the camera" classifier.

    Operates on the *inspected* branch only (a reference is not needed to
    tell whether a T-shirt is even present) using the P5 feature map.
    """

    def __init__(self, in_channels: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, p5_feat: torch.Tensor) -> torch.Tensor:
        x = self.pool(p5_feat).flatten(1)
        return self.mlp(x).squeeze(-1)  # (B,) logit


class GarmentAttributeHead(nn.Module):
    """Optional: predicts style/color/size from the inspected image, used
    only to flag a reference-selection mismatch (see DESIGN.md section 11).
    """

    def __init__(self, in_channels: int, num_styles: int, num_colors: int, num_sizes: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.shared = nn.Sequential(nn.Linear(in_channels, 256), nn.ReLU(inplace=True))
        self.style_fc = nn.Linear(256, num_styles)
        self.color_fc = nn.Linear(256, num_colors)
        self.size_fc = nn.Linear(256, num_sizes)

    def forward(self, p5_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.shared(self.pool(p5_feat).flatten(1))
        return {"style": self.style_fc(x), "color": self.color_fc(x), "size": self.size_fc(x)}
