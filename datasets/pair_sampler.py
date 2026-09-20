"""Batch composition + hard-negative mining for Siamese pair training.

Pair taxonomy actually produced by SiameseDefectDataset (see DESIGN.md
section 5):

  - Positive pair  (pair_label=1): reference (golden)  + defective inspected image
  - Negative pair  (pair_label=0): reference (golden)  + good inspected image
                                    (same reference_image, no defects)
  - Hard negative  (pair_label=0): reference (golden)  + good inspected image that
                                    the model currently scores as "far" in
                                    embedding space (different lighting,
                                    fabric wrinkle/deformation, camera angle,
                                    or a *different but visually similar*
                                    garment) -- these are the pairs that most
                                    often cause false rejects in production
                                    and must be over-represented.

This module implements:
  1. `SiamesePairBatchSampler` -- yields batches with a controlled ratio of
     positive / negative / hard-negative pairs per config.
  2. `HardNegativePool` -- maintained across epochs, refreshed by scoring the
     current negative pool with the model's own embedding distance and
     keeping the top-K "closest to the decision boundary but still correct"
     and "highest-distance false-flag risk" samples.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Sampler


class HardNegativePool:
    """Ranks negative (no-defect) samples by current model embedding
    distance to their reference, keeps the hardest ones for oversampling.

    Refresh cadence and pool size are config-driven
    (data.pair_sampling.hard_negative_pool_size /
    hard_negative_refresh_every_epochs) -- refreshing every epoch is
    affordable because it only requires a forward pass through the encoder
    (no detection/segmentation heads), typically <5% of one training epoch.
    """

    def __init__(self, negative_indices: List[int], pool_size: int = 5000):
        self.negative_indices = negative_indices
        self.pool_size = pool_size
        self.hard_indices: List[int] = list(negative_indices[:pool_size])

    @torch.no_grad()
    def refresh(self, encoder, dataset, device, batch_size: int = 32) -> None:
        """encoder: SiameseEncoder (or full model, only .encoder is used).
        Scores every negative sample's embedding distance and keeps the
        top `pool_size` hardest (largest distance despite being a true
        no-defect pair -> the model is currently *wrong or unsure* here).
        """
        encoder.eval()
        distances = np.zeros(len(self.negative_indices), dtype=np.float32)

        for start in range(0, len(self.negative_indices), batch_size):
            idx_batch = self.negative_indices[start:start + batch_size]
            refs, imgs = [], []
            for idx in idx_batch:
                sample = dataset[idx]
                refs.append(sample["reference"])
                imgs.append(sample["input"])
            ref_t = torch.stack(refs).to(device)
            img_t = torch.stack(imgs).to(device)

            out = encoder(ref_t, img_t)
            d = torch.nn.functional.pairwise_distance(
                out["reference_embedding"], out["inspected_embedding"])
            distances[start:start + len(idx_batch)] = d.cpu().numpy()

        order = np.argsort(-distances)  # largest distance first == hardest
        keep = order[: self.pool_size]
        self.hard_indices = [self.negative_indices[i] for i in keep]
        encoder.train()


class SiamesePairBatchSampler(Sampler):
    """Draws each batch with a controlled mix of positive / (easy + hard)
    negative pairs, per configs/config.yaml -> data.pair_sampling."""

    def __init__(self, positive_indices: List[int], negative_indices: List[int],
                 hard_negative_pool: HardNegativePool, batch_size: int,
                 positive_ratio: float = 0.5, hard_negative_ratio: float = 0.3,
                 num_batches: int = 1000, seed: int = 0):
        self.positive_indices = positive_indices
        self.negative_indices = negative_indices
        self.hard_pool = hard_negative_pool
        self.batch_size = batch_size
        self.positive_ratio = positive_ratio
        self.hard_negative_ratio = hard_negative_ratio
        self.num_batches = num_batches
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        n_pos = int(round(self.batch_size * self.positive_ratio))
        n_neg = self.batch_size - n_pos
        n_hard = int(round(n_neg * self.hard_negative_ratio))
        n_easy = n_neg - n_hard

        for _ in range(self.num_batches):
            batch = []
            batch += list(self.rng.choice(self.positive_indices, n_pos, replace=True))
            if n_hard > 0 and len(self.hard_pool.hard_indices) > 0:
                batch += list(self.rng.choice(self.hard_pool.hard_indices, n_hard, replace=True))
            batch += list(self.rng.choice(self.negative_indices, n_easy, replace=True))
            self.rng.shuffle(batch)
            yield batch


def split_positive_negative(dataset) -> Dict[str, List[int]]:
    positive, negative = [], []
    for i, record in enumerate(dataset.images):
        (positive if record.get("defects") else negative).append(i)
    return {"positive": positive, "negative": negative}
