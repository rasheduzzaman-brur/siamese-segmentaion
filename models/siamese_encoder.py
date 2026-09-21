"""FPN neck + Siamese (weight-shared) two-branch encoder.

Both the reference image and the inspected image are passed through the
*same* backbone+FPN instance (true weight sharing, not two copies), which is
what makes the comparison in feature space meaningful and halves the
parameter count vs. two independent towers.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import build_backbone


class FPN(nn.Module):
    """Standard top-down FPN (Lin et al. 2017) producing P3-P7."""

    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        c3_ch, c4_ch, c5_ch = in_channels
        self.lateral_c3 = nn.Conv2d(c3_ch, out_channels, 1)
        self.lateral_c4 = nn.Conv2d(c4_ch, out_channels, 1)
        self.lateral_c5 = nn.Conv2d(c5_ch, out_channels, 1)

        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # extra strided levels for large-object / global context (RetinaNet/FCOS style)
        self.p6 = nn.Conv2d(c5_ch, out_channels, 3, stride=2, padding=1)
        self.p7 = nn.Sequential(nn.ReLU(inplace=True),
                                 nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1))

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        c3, c4, c5 = feats["c3"], feats["c4"], feats["c5"]

        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")

        p3 = self.smooth_p3(p3)
        p4 = self.smooth_p4(p4)
        p5 = self.smooth_p5(p5)
        p6 = self.p6(c5)
        p7 = self.p7(p6)
        return {"p3": p3, "p4": p4, "p5": p5, "p6": p6, "p7": p7}


class GeMPooling(nn.Module):
    """Generalized-mean pooling -> compact embedding vector per image.

    Used only for the global Siamese embedding branch (unknown-defect
    detection + auxiliary metric-learning loss), not for the dense
    detection head.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1).pow(1.0 / self.p)
        return x.flatten(1)


class SiameseEncoder(nn.Module):
    """Runs backbone+FPN once per branch with fully shared weights.

    Returns per-level feature maps for both branches plus an L2-normalized
    global embedding per branch (from P5) for metric-learning / anomaly
    scoring.
    """

    def __init__(self, backbone_cfg: dict, fpn_out_channels: int = 256, embed_dim: int = 256):
        super().__init__()
        self.backbone = build_backbone(backbone_cfg)
        self.fpn = FPN(self.backbone.out_channels, fpn_out_channels)
        self.gem = GeMPooling()
        self.embed_proj = nn.Linear(fpn_out_channels, embed_dim)

    def _encode_one(self, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        feats = self.fpn(self.backbone(x))
        pooled = self.gem(feats["p5"])
        embedding = F.normalize(self.embed_proj(pooled), dim=-1)
        return feats, embedding

    def forward(self, reference: torch.Tensor, inspected: torch.Tensor):
        ref_feats, ref_embed = self._encode_one(reference)
        inp_feats, inp_embed = self._encode_one(inspected)
        return {
            "reference": ref_feats, "inspected": inp_feats,
            "reference_embedding": ref_embed, "inspected_embedding": inp_embed,
        }
