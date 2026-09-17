"""Siamese metric-learning losses over the global (reference, inspected)
embedding pair produced by SiameseEncoder.

label convention: y = 0 -> "matching / no-defect" pair, y = 1 -> "defective /
mismatched" pair (same convention as classic contrastive loss, Hadsell et
al. 2006).

  Contrastive:      L = (1-y) * d^2 + y * max(0, margin - d)^2
  Cosine Embedding:  L = (1-y) * (1 - cos) + y * max(0, cos - margin)
  Triplet:          L = max(0, d(a,p) - d(a,n) + margin)
  InfoNCE:          L = -log( exp(sim(a,p)/T) / sum_i exp(sim(a,i)/T) )

Recommended default: Contrastive loss on L2-normalized embeddings (see
DESIGN.md section 6) -- it maps directly onto the deployed anomaly-scoring
mechanism (thresholding a distance), unlike triplet/InfoNCE which need
mined negatives per anchor and are harder to calibrate to one operating
threshold.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        d = F.pairwise_distance(emb_a, emb_b, p=2)
        loss = (1 - y) * d.pow(2) + y * F.relu(self.margin - d).pow(2)
        return loss.mean()


class CosineEmbeddingLossWrapper(nn.Module):
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.loss = nn.CosineEmbeddingLoss(margin=margin)

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # nn.CosineEmbeddingLoss expects target in {+1 (similar), -1 (dissimilar)}
        target = 1 - 2 * y.float()
        return self.loss(emb_a, emb_b, target)


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        return self.loss(anchor, positive, negative)


class InfoNCELoss(nn.Module):
    """In-batch negatives InfoNCE; requires a batch of (reference, matching
    inspected) pairs so other batch entries serve as negatives."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        logits = emb_a @ emb_b.t() / self.temperature
        labels = torch.arange(emb_a.shape[0], device=emb_a.device)
        return F.cross_entropy(logits, labels)


def build_siamese_loss(cfg: dict) -> nn.Module:
    kind = cfg["type"]
    if kind == "contrastive":
        return ContrastiveLoss(margin=cfg["margin"])
    if kind == "cosine_embedding":
        return CosineEmbeddingLossWrapper(margin=cfg.get("margin", 0.3))
    if kind == "triplet":
        return TripletLoss(margin=cfg.get("margin", 0.3))
    if kind == "infonce":
        return InfoNCELoss(temperature=cfg.get("temperature", 0.07))
    raise ValueError(f"Unknown siamese loss type: {kind}")
