"""Sliding-window inference over aligned full-resolution (reference, input)
pairs, for small defects on large images (data.crop / inference.tile in the
config, e.g. configs/pcb.yaml).

Every tile has the same fixed size (edge tiles are shifted inward, and images
smaller than a tile are zero-padded), so the model always sees one input shape
-- matching training crops and keeping a fixed-shape OpenVINO export valid.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
import torchvision


def tile_starts(length: int, tile: int, overlap: int) -> List[int]:
    if length <= tile:
        return [0]
    stride = max(1, tile - overlap)
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


@torch.no_grad()
def predict_tiled(model, reference: torch.Tensor, inspected: torch.Tensor, tile: int,
                  overlap: int, score_thresh: float, nms_iou: float, max_dets: int,
                  batch_size: int = 8, return_embedding_distance: bool = False):
    """reference / inspected: (1, 3, H, W) or (3, H, W) float tensors on the
    model's device, same size. Returns model.postprocess-style dicts with
    boxes in full-image pixel coordinates; with return_embedding_distance,
    returns (results, max per-tile reference/input embedding distance)."""
    if reference.dim() == 3:
        reference, inspected = reference[None], inspected[None]
    if reference.shape != inspected.shape:
        raise ValueError(f"tiled inference needs same-size aligned pairs, got "
                         f"{tuple(reference.shape)} vs {tuple(inspected.shape)}")
    _, _, h, w = inspected.shape
    pad_h, pad_w = max(0, tile - h), max(0, tile - w)
    if pad_h or pad_w:
        reference = F.pad(reference, (0, pad_w, 0, pad_h))
        inspected = F.pad(inspected, (0, pad_w, 0, pad_h))

    windows = [(x, y) for y in tile_starts(h + pad_h, tile, overlap)
               for x in tile_starts(w + pad_w, tile, overlap)]

    boxes, scores, labels = [], [], []
    max_dist = 0.0
    for i in range(0, len(windows), batch_size):
        chunk = windows[i:i + batch_size]
        ref_b = torch.cat([reference[..., y:y + tile, x:x + tile] for x, y in chunk])
        inp_b = torch.cat([inspected[..., y:y + tile, x:x + tile] for x, y in chunk])
        out = model(ref_b, inp_b)
        max_dist = max(max_dist, F.pairwise_distance(
            out["reference_embedding"], out["inspected_embedding"]).max().item())
        for b, (x, y) in enumerate(chunk):
            single = {"levels": {lvl: {k: v[b:b + 1] for k, v in pred.items()}
                                 for lvl, pred in out["levels"].items()},
                      "points": out["points"]}
            for r in model.postprocess(single, (tile, tile), score_thresh=score_thresh,
                                       nms_iou=nms_iou, max_dets=max_dets):
                x1, y1, x2, y2 = r["bbox"]
                boxes.append([x1 + x, y1 + y, x2 + x, y2 + y])
                scores.append(r["confidence"])
                labels.append(r["class_id"])

    if not boxes:
        return ([], max_dist) if return_embedding_distance else []
    boxes_t = torch.tensor(boxes)
    boxes_t[:, [0, 2]] = boxes_t[:, [0, 2]].clamp(0, w)
    boxes_t[:, [1, 3]] = boxes_t[:, [1, 3]].clamp(0, h)
    scores_t, labels_t = torch.tensor(scores), torch.tensor(labels)
    keep = torchvision.ops.batched_nms(boxes_t, scores_t, labels_t, nms_iou)[:max_dets]
    results = [{"bbox": boxes_t[k].tolist(), "class_id": int(labels_t[k]),
                "confidence": float(scores_t[k])} for k in keep.tolist()]
    return (results, max_dist) if return_embedding_distance else results
