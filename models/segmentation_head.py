"""Single-stage, anchor-free instance segmentation head.

Design: FCOS-style dense prediction (classification / box / centerness per
feature-map location) + a YOLACT/CondInst-style prototype mask branch. This
avoids RoIAlign and a two-stage RPN->RoI pipeline entirely, which keeps the
graph simple and fast to export to ONNX/OpenVINO (see DESIGN.md section 15).

Per FPN location we predict:
  - cls_logits   : (num_classes,)  defect-class logits (focal loss, no explicit background)
  - box_reg      : (4,)            distances (l, t, r, b) to the box edges, in stride units
  - centerness   : (1,)            "how close to the instance center" quality score
  - mask_coeff   : (k,)            coefficients over the shared prototypes

A global prototype branch (on the highest-resolution level, P3) produces k
prototype maps shared by every instance; an instance's final mask is the
sigmoid of the coefficient-weighted sum of prototypes, cropped to its box.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

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


class ProtoNet(nn.Module):
    """Produces k shared prototype masks from the highest-resolution level."""

    def __init__(self, in_channels: int, proto_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, proto_channels, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, p3_feat: torch.Tensor) -> torch.Tensor:
        return self.net(p3_feat)  # (B, k, 2*H3, 2*W3) -> stride 4 relative to input


class SegmentationHead(nn.Module):
    def __init__(self, num_classes: int, channels: int = 256, num_convs: int = 4,
                 proto_channels: int = 32):
        super().__init__()
        self.num_classes = num_classes
        self.proto_channels = proto_channels

        self.cls_tower = ConvTower(channels, num_convs)
        self.box_tower = ConvTower(channels, num_convs)

        self.cls_pred = nn.Conv2d(channels, num_classes, 3, padding=1)
        self.box_pred = nn.Conv2d(channels, 4, 3, padding=1)
        self.centerness_pred = nn.Conv2d(channels, 1, 3, padding=1)
        self.mask_coeff_pred = nn.Conv2d(channels, proto_channels, 3, padding=1)

        self.scales = nn.ModuleDict({lvl: Scale(1.0) for lvl in STRIDES})
        self.proto_net = ProtoNet(channels, proto_channels)

        self._init_weights()

    def _init_weights(self):
        for m in [self.cls_tower, self.box_tower, self.cls_pred, self.box_pred,
                  self.centerness_pred, self.mask_coeff_pred]:
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
                "mask_coeff": self.mask_coeff_pred(box_feat),
            }
        prototypes = self.proto_net(features["p3"])
        return {"levels": per_level, "prototypes": prototypes}


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


def assemble_instance_masks(prototypes: torch.Tensor, coeffs: torch.Tensor,
                             boxes: torch.Tensor, out_size: int = 28) -> torch.Tensor:
    """Combine prototypes -> per-instance mask, cropped to its box.

    prototypes: (k, Hp, Wp) for one image (proto stride = 4)
    coeffs:     (N, k)      per-instance coefficients
    boxes:      (N, 4)      xyxy, in input-image pixel coordinates
    returns:    (N, out_size, out_size) mask logits
    """
    k, hp, wp = prototypes.shape
    n = coeffs.shape[0]
    if n == 0:
        return prototypes.new_zeros((0, out_size, out_size))

    full_mask = torch.einsum("nk,khw->nhw", coeffs, prototypes)  # (N, Hp, Wp)
    return roi_crop_resize(full_mask, boxes, out_size, source_stride=4.0)


def roi_crop_resize(maps: torch.Tensor, boxes: torch.Tensor, out_size: int,
                     source_stride: float = 1.0) -> torch.Tensor:
    """Differentiable per-instance RoI crop + resize via affine_grid/grid_sample
    (ONNX/OpenVINO-exportable, unlike torchvision.ops.roi_align which needs a
    custom op in some export paths).

    maps:  (N, H, W) one map per instance (e.g. assembled proto mask, or a
           full-resolution GT mask already indexed per instance)
    boxes: (N, 4) xyxy in the *same coordinate space as `maps` scaled by
           source_stride* -- i.e. image-pixel boxes with source_stride=1 for
           full-res GT masks, or proto-stride boxes for prototype maps.
    """
    n, h, w = maps.shape
    if n == 0:
        return maps.new_zeros((0, out_size, out_size))

    boxes_scaled = boxes / source_stride
    theta = torch.zeros(n, 2, 3, device=boxes.device, dtype=boxes.dtype)
    x1, y1, x2, y2 = boxes_scaled.unbind(dim=1)
    theta[:, 0, 0] = (x2 - x1) / w
    theta[:, 0, 2] = (x1 + x2) / w - 1
    theta[:, 1, 1] = (y2 - y1) / h
    theta[:, 1, 2] = (y1 + y2) / h - 1

    grid = F.affine_grid(theta, size=(n, 1, out_size, out_size), align_corners=False)
    cropped = F.grid_sample(maps.unsqueeze(1), grid, align_corners=False)
    return cropped.squeeze(1)
