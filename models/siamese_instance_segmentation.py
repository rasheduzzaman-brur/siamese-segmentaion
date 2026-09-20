"""Top-level model: Siamese encoder -> feature comparison -> detection /
segmentation / classification heads.

This is the "recommended final architecture" from DESIGN.md section 2/17:
a single-stage (FCOS + prototype-mask) Siamese instance segmentation
network. It intentionally avoids a two-stage RPN/RoIAlign pipeline for
export simplicity and latency (see DESIGN.md section 15/18).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torchvision

from models.siamese_encoder import SiameseEncoder
from models.feature_comparison import FeatureComparison
from models.segmentation_head import (
    SegmentationHead, STRIDES, SCALE_RANGES, generate_points, assemble_instance_masks,
)


def decode_boxes(points: torch.Tensor, box_reg: torch.Tensor) -> torch.Tensor:
    """points: (N,2) xy: box_reg: (N,4) ltrb distances -> (N,4) xyxy boxes."""
    x, y = points[:, 0], points[:, 1]
    l, t, r, b = box_reg.unbind(dim=1)
    return torch.stack([x - l, y - t, x + r, y + b], dim=1)


class SiameseInstanceSegmentation(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        mcfg = cfg["model"]
        self.num_classes = len(cfg["data"]["defect_classes"])
        fpn_ch = mcfg["fpn"]["out_channels"]

        self.encoder = SiameseEncoder(mcfg["backbone"], fpn_ch, mcfg["embedding"]["dim"])
        self.comparison = FeatureComparison(
            fpn_ch, mode=mcfg["comparison"]["mode"],
            use_cosine_channel=mcfg["comparison"]["use_cosine_channel"],
        )
        self.seg_head = SegmentationHead(
            self.num_classes, fpn_ch,
            num_convs=mcfg["seg_head"]["num_convs"],
            proto_channels=mcfg["seg_head"]["proto_channels"],
        )
        self.mask_out_size = mcfg["seg_head"]["mask_out_size"]

    def forward(self, reference: torch.Tensor, inspected: torch.Tensor) -> Dict:
        enc = self.encoder(reference, inspected)
        fused = self.comparison(enc["reference"], enc["inspected"])
        head_out = self.seg_head(fused)

        shapes = {lvl: tuple(f.shape[-2:]) for lvl, f in fused.items()}
        points = generate_points(shapes, inspected.device)

        return {
            "levels": head_out["levels"],
            "prototypes": head_out["prototypes"],
            "points": points,
            "reference_embedding": enc["reference_embedding"],
            "inspected_embedding": enc["inspected_embedding"],
        }

    # ------------------------------------------------------------------
    # Training-time target assignment (FCOS-style)
    # ------------------------------------------------------------------
    @staticmethod
    def assign_targets(points: Dict[str, torch.Tensor], gt_boxes: torch.Tensor,
                        gt_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Assign each feature-map location to a GT instance (or background).

        gt_boxes:  (M, 4) xyxy in image pixels, single image
        gt_labels: (M,)   class indices
        Returns per-level dict of {labels, box_targets, centerness_targets, matched_gt_idx}
        with background labels == -1.
        """
        targets = {}
        for lvl, pts in points.items():
            n = pts.shape[0]
            if gt_boxes.numel() == 0:
                targets[lvl] = {
                    "labels": pts.new_full((n,), -1, dtype=torch.long),
                    "box_targets": pts.new_zeros((n, 4)),
                    "centerness_targets": pts.new_zeros((n,)),
                    "matched_gt_idx": pts.new_full((n,), -1, dtype=torch.long),
                }
                continue

            x, y = pts[:, 0:1], pts[:, 1:2]                       # (N,1)
            x1, y1, x2, y2 = gt_boxes.unbind(dim=1)                  # (M,)
            l = x - x1[None, :]; t = y - y1[None, :]
            r = x2[None, :] - x; b = y2[None, :] - y
            reg_targets = torch.stack([l, t, r, b], dim=-1)         # (N, M, 4)

            inside_box = reg_targets.min(dim=-1).values > 0
            max_reg = reg_targets.max(dim=-1).values
            lo, hi = SCALE_RANGES[lvl]
            inside_scale = (max_reg >= lo) & (max_reg < hi)

            areas = (x2 - x1) * (y2 - y1)                            # (M,)
            areas = areas[None, :].expand(n, -1).clone()
            areas[~(inside_box & inside_scale)] = float("inf")

            min_area, matched_gt_idx = areas.min(dim=1)              # (N,)
            is_positive = torch.isfinite(min_area)

            labels = pts.new_full((n,), -1, dtype=torch.long)
            labels[is_positive] = gt_labels[matched_gt_idx[is_positive]]

            box_targets = reg_targets[torch.arange(n, device=pts.device), matched_gt_idx]
            box_targets[~is_positive] = 0

            lr = box_targets[:, [0, 2]]
            tb = box_targets[:, [1, 3]]
            centerness = torch.sqrt(
                (lr.min(dim=1).values / lr.max(dim=1).values.clamp(min=1e-6)) *
                (tb.min(dim=1).values / tb.max(dim=1).values.clamp(min=1e-6))
            )
            centerness[~is_positive] = 0

            targets[lvl] = {
                "labels": labels, "box_targets": box_targets,
                "centerness_targets": centerness,
                "matched_gt_idx": torch.where(is_positive, matched_gt_idx, -1),
            }
        return targets

    # ------------------------------------------------------------------
    # Inference-time decode + NMS + mask assembly
    # ------------------------------------------------------------------
    @torch.no_grad()
    def postprocess(self, outputs: Dict, image_size, score_thresh: float = 0.5,
                     nms_iou: float = 0.5, max_dets: int = 50) -> List[Dict]:
        """Single-image postprocess (call per-item for a batch)."""
        h, w = image_size
        all_boxes, all_scores, all_labels, all_coeffs = [], [], [], []

        for lvl, pred in outputs["levels"].items():
            cls_logits = pred["cls_logits"][0]           # (C, Hl, Wl)
            box_reg = pred["box_reg"][0]                   # (4, Hl, Wl)
            centerness = pred["centerness"][0]              # (1, Hl, Wl)
            coeff = pred["mask_coeff"][0]                    # (K, Hl, Wl)
            c = cls_logits.shape[0]

            cls_scores = cls_logits.permute(1, 2, 0).reshape(-1, c).sigmoid()
            ctr_scores = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            box_reg = box_reg.permute(1, 2, 0).reshape(-1, 4)
            coeff = coeff.permute(1, 2, 0).reshape(-1, coeff.shape[0])

            scores, labels = cls_scores.max(dim=1)
            scores = scores * ctr_scores
            keep = scores > score_thresh
            if keep.sum() == 0:
                continue

            boxes = decode_boxes(outputs["points"][lvl][keep], box_reg[keep])
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)

            all_boxes.append(boxes)
            all_scores.append(scores[keep])
            all_labels.append(labels[keep])
            all_coeffs.append(coeff[keep])

        if not all_boxes:
            return []

        boxes = torch.cat(all_boxes)
        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        coeffs = torch.cat(all_coeffs)

        keep = torchvision.ops.batched_nms(boxes, scores, labels, nms_iou)[:max_dets]
        boxes, scores, labels, coeffs = boxes[keep], scores[keep], labels[keep], coeffs[keep]

        prototypes = outputs["prototypes"][0]  # (K, Hp, Wp)
        mask_logits = assemble_instance_masks(prototypes, coeffs, boxes, self.mask_out_size)

        results = []
        for i in range(boxes.shape[0]):
            results.append({
                "bbox": boxes[i].tolist(),
                "class_id": int(labels[i].item()),
                "confidence": float(scores[i].item()),
                "mask_logits": mask_logits[i],  # caller resizes to bbox + pastes on full image
            })
        return results
