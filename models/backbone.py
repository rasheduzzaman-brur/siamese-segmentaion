"""Shared-weight CNN backbones used by both Siamese branches.

Only the backbone feature maps (C3, C4, C5 -> strides 8/16/32) are exposed;
the FPN lives in siamese_encoder.py so the backbone stays swappable.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torchvision


_RESNET_OUT_CHANNELS = {
    "resnet18": [128, 256, 512],
    "resnet34": [128, 256, 512],
    "resnet50": [512, 1024, 2048],
    "resnet101": [512, 1024, 2048],
}


class ResNetBackbone(nn.Module):
    """torchvision ResNet truncated to stages C3/C4/C5, ONNX/OpenVINO-friendly."""

    def __init__(self, name: str = "resnet50", pretrained: bool = True, frozen_stages: int = 1):
        super().__init__()
        if name not in _RESNET_OUT_CHANNELS:
            raise ValueError(f"Unsupported resnet backbone: {name}")
        weights = "IMAGENET1K_V2" if pretrained else None
        net = getattr(torchvision.models, name)(weights=weights)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1  # stride 4  (C2, not returned but needed internally)
        self.layer2 = net.layer2  # stride 8  (C3)
        self.layer3 = net.layer3  # stride 16 (C4)
        self.layer4 = net.layer4  # stride 32 (C5)

        self.out_channels: List[int] = _RESNET_OUT_CHANNELS[name]
        self._freeze_stages(frozen_stages)

    def _freeze_stages(self, frozen_stages: int) -> None:
        if frozen_stages <= 0:
            return
        modules = [self.stem]
        if frozen_stages >= 2:
            modules.append(self.layer1)
        for m in modules:
            for p in m.parameters():
                p.requires_grad = False
            m.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # keep frozen stages (and their BN running stats) in eval mode
        for p in self.stem.parameters():
            if not p.requires_grad:
                self.stem.eval()
                break
        return self

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return {"c3": c3, "c4": c4, "c5": c5}


class MobileNetV3Backbone(nn.Module):
    """Lightweight alternative for CPU / edge-constrained OpenVINO deployment."""

    def __init__(self, pretrained: bool = True, **_ignored):
        super().__init__()
        weights = "IMAGENET1K_V2" if pretrained else None
        net = torchvision.models.mobilenet_v3_large(weights=weights).features
        # feature indices producing strides 8 / 16 / 32 for mobilenet_v3_large
        self.stage_a = net[:7]     # -> stride 8,  40 channels
        self.stage_b = net[7:13]  # -> stride 16, 112 channels
        self.stage_c = net[13:]  # -> stride 32, 960 channels
        self.out_channels = [40, 112, 960]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        c3 = self.stage_a(x)
        c4 = self.stage_b(c3)
        c5 = self.stage_c(c4)
        return {"c3": c3, "c4": c4, "c5": c5}


def build_backbone(cfg: dict) -> nn.Module:
    name = cfg["name"]
    if name.startswith("resnet"):
        return ResNetBackbone(name, pretrained=cfg.get("pretrained", True),
                               frozen_stages=cfg.get("frozen_stages", 1))
    if name == "mobilenet_v3_large":
        return MobileNetV3Backbone(pretrained=cfg.get("pretrained", True))
    raise ValueError(f"Unknown backbone '{name}'. See build_backbone() for supported names.")
