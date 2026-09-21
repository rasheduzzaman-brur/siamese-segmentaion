"""Single-stage, anchor-free detection head.

Design: FCOS-style dense prediction (classification / box / centerness per
feature-map location). This avoids RoIAlign and a two-stage RPN->RoI
pipeline entirely, which keeps the graph simple and fast to export to
ONNX/OpenVINO (see DESIGN.md section 15).

Per FPN location we predict:
  - cls_logits   : (num_classes,)  defect-class logits (focal loss, no explicit background)
  - box_reg      : (4,)            distances (l, t, r, b) to the box edges, in stride units
  - centerness   : (1,)            "how close to the instance center" quality score
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDES = {"p3": 8, "p4": 16, "p5": 32, "p6": 64, "p7": 128}
# object-size range assigned to each level (FCOS-style), in input-image pixels
SCALE_RANGES = {
    "p3": (0, 64), "p4": (64, 128), "p5": (128, 256), "p6": (256, 512), "p7": (512, 1e8),
}


class ConvTower(nn.Module):
    def __init__(self, channels: int, num_convs: int = 4):
        super().__init__()
        layers = []
        for _ in range(num_convs):
            layers += [nn.Conv2d(channels, channels, 3, padding=1),
                       nn.GroupNorm(32, channels), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Scale(nn.Module):
    """Learnable per-level scalar for box regression (FCOS trick: levels
    otherwise fight over the same regression weights)."""

    def __init__(self, init: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class DetectionHead(nn.Module):
    def __init__(self, num_classes: int, channels: int = 256, num_convs: int = 4):
        super().__init__()
        self.num_classes = num_classes

        self.cls_tower = ConvTower(channels, num_convs)
        self.box_tower = ConvTower(channels, num_convs)

        self.cls_pred = nn.Conv2d(channels, num_classes, 3, padding=1)
        self.box_pred = nn.Conv2d(channels, 4, 3, padding=1)
        self.centerness_pred = nn.Conv2d(channels, 1, 3, padding=1)

        self.scales = nn.ModuleDict({lvl: Scale(1.0) for lvl in STRIDES})

        self._init_weights()

    def _init_weights(self):
        for m in [self.cls_tower, self.box_tower, self.cls_pred, self.box_pred,
                  self.centerness_pred]:
            for layer in m.modules() if isinstance(m, nn.Module) else [m]:
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    nn.init.constant_(layer.bias, 0)
        # focal-loss-friendly prior: rare positives at init (Lin et al. 2017, RetinaNet)
        prior = 0.01
        nn.init.constant_(self.cls_pred.bias, -torch.log(torch.tensor((1 - prior) / prior)).item())

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        per_level = {}
        for lvl, feat in features.items():
            cls_feat = self.cls_tower(feat)
            box_feat = self.box_tower(feat)
            box_reg = F.relu(self.scales[lvl](self.box_pred(box_feat)))
            per_level[lvl] = {
                "cls_logits": self.cls_pred(cls_feat),
                "box_reg": box_reg * STRIDES[lvl],
                "centerness": self.centerness_pred(box_feat),
            }
        return {"levels": per_level}


def generate_points(shapes: Dict[str, Tuple[int, int]], device) -> Dict[str, torch.Tensor]:
    """Grid of (x, y) locations, in input-image pixel coordinates, per level."""
    points = {}
    for lvl, (h, w) in shapes.items():
        stride = STRIDES[lvl]
        ys, xs = torch.meshgrid(
            torch.arange(h, device=device) * stride + stride // 2,
            torch.arange(w, device=device) * stride + stride // 2,
            indexing="ij",
        )
        points[lvl] = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1).float()
    return points
