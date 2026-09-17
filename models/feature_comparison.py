"""Multi-scale feature comparison between reference and inspected branches.

Implements several strategies (see DESIGN.md section 2) behind one interface
so the comparison mode is a config switch, not a code change:

  - "abs_diff"            : |F_in - F_ref|
  - "subtract"            : F_in - F_ref (signed)
  - "concat"              : conv1x1(cat[F_in, F_ref])
  - "cosine"              : per-pixel cosine similarity map (1 channel), broadcast
  - "concat_diff_residual": F_in + conv1x1(cat[F_in, F_ref, |F_in-F_ref|, cos_map])
                            <- recommended default, see DESIGN.md
  - "cross_attention"     : single-head cross-attention at the coarsest level only
                            (P6/P7), too expensive to run at every level/pixel.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cosine_map(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    a_n = F.normalize(a, dim=1, eps=eps)
    b_n = F.normalize(b, dim=1, eps=eps)
    return (a_n * b_n).sum(dim=1, keepdim=True)  # (B,1,H,W) in [-1, 1]


class LevelFusion(nn.Module):
    """Fusion block applied independently at every FPN level (weights are
    level-specific since defect scale statistics differ across levels, but
    shared across the reference/inspected pair by construction)."""

    def __init__(self, channels: int, use_cosine_channel: bool = True):
        super().__init__()
        self.use_cosine_channel = use_cosine_channel
        in_ch = channels * 3 + (1 if use_cosine_channel else 0)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch, channels, kernel_size=1),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, f_in: torch.Tensor, f_ref: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(f_in - f_ref)
        parts = [f_in, f_ref, diff]
        if self.use_cosine_channel:
            parts.append(_cosine_map(f_in, f_ref))
        combined = self.reduce(torch.cat(parts, dim=1))
        return f_in + combined  # residual: degrades gracefully w/o a reference


class CrossAttentionFusion(nn.Module):
    """Lightweight single-head cross-attention, coarsest level only (P6/P7).

    Cost scales with (H*W)^2 so this is intentionally restricted to the
    smallest feature maps -- see DESIGN.md for the accuracy/cost trade-off.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, f_in: torch.Tensor, f_ref: torch.Tensor) -> torch.Tensor:
        b, c, h, w = f_in.shape
        q = self.q(f_in).flatten(2).transpose(1, 2)      # (B, HW, C)
        k = self.k(f_ref).flatten(2)                       # (B, C, HW)
        v = self.v(f_ref).flatten(2).transpose(1, 2)      # (B, HW, C)
        attn = torch.softmax((q @ k) * self.scale, dim=-1)  # (B, HW, HW)
        out = (attn @ v).transpose(1, 2).reshape(b, c, h, w)
        return f_in + self.out(out)


class FeatureComparison(nn.Module):
    """Applies the configured comparison mode at every FPN level."""

    LEVELS = ("p3", "p4", "p5", "p6", "p7")
    ATTENTION_LEVELS = ("p6", "p7")  # only fuse w/ cross-attention here

    def __init__(self, channels: int = 256, mode: str = "concat_diff_residual",
                 use_cosine_channel: bool = True):
        super().__init__()
        self.mode = mode
        if mode in ("concat", "abs_diff", "subtract", "concat_diff_residual"):
            self.fusions = nn.ModuleDict({lvl: LevelFusion(channels, use_cosine_channel)
                                          for lvl in self.LEVELS})
        elif mode == "cross_attention":
            self.fusions = nn.ModuleDict({lvl: LevelFusion(channels, use_cosine_channel)
                                          for lvl in self.LEVELS if lvl not in self.ATTENTION_LEVELS})
            self.attn_fusions = nn.ModuleDict({lvl: CrossAttentionFusion(channels)
                                               for lvl in self.ATTENTION_LEVELS})
        elif mode == "cosine":
            self.fusions = None
        else:
            raise ValueError(f"Unknown comparison mode: {mode}")

    def forward(self, ref_feats: Dict[str, torch.Tensor],
                inp_feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for lvl in self.LEVELS:
            f_in, f_ref = inp_feats[lvl], ref_feats[lvl]
            if self.mode == "cosine":
                out[lvl] = f_in * _cosine_map(f_in, f_ref).sigmoid()
            elif self.mode == "abs_diff":
                out[lvl] = torch.abs(f_in - f_ref)
            elif self.mode == "subtract":
                out[lvl] = f_in - f_ref
            elif self.mode == "cross_attention" and lvl in self.ATTENTION_LEVELS:
                out[lvl] = self.attn_fusions[lvl](f_in, f_ref)
            else:
                out[lvl] = self.fusions[lvl](f_in, f_ref)
        return out
