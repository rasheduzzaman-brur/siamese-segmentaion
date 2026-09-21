"""Evaluation metrics for every task head (see DESIGN.md section 10).

  - Defect classification      : per-class precision/recall/F1 + confusion matrix
  - Object detection            : box IoU, AP50, AP75, box mAP (COCO-style)
  - Industrial decision metrics: false reject rate (FRR), false accept rate (FAR),
                                  defect-recall (garment-level), latency/FPS
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def box_iou(box_a: List[float], box_b: List[float], eps: float = 1e-6) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / (union + eps)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """101-point interpolated AP, COCO convention."""
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[0.0], precisions, [0.0]])
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        idx = np.searchsorted(recalls, r)
        ap += precisions[min(idx, len(precisions) - 1)] / 101
    return float(ap)


def match_predictions_to_gt(pred_boxes: List[List[float]], pred_labels: List[int],
                             pred_scores: List[float], gt_boxes: List[List[float]],
                             gt_labels: List[int], iou_thresh: float) -> Dict:
    """Greedy one-to-one matching by descending score, per class. Returns
    per-prediction TP/FP flags for mAP accumulation."""
    order = np.argsort(-np.array(pred_scores)) if pred_scores else np.array([], dtype=int)
    matched_gt = set()
    tp = np.zeros(len(pred_scores))
    fp = np.zeros(len(pred_scores))

    for rank, i in enumerate(order):
        best_iou, best_j = 0.0, -1
        for j, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
            if j in matched_gt or gl != pred_labels[i]:
                continue
            iou = box_iou(pred_boxes[i], gb)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh and best_j >= 0:
            tp[rank] = 1
            matched_gt.add(best_j)
        else:
            fp[rank] = 1
    return {"tp": tp, "fp": fp, "num_gt": len(gt_boxes)}


def box_map(all_matches: List[Dict]) -> float:
    tp = np.concatenate([m["tp"] for m in all_matches]) if all_matches else np.array([])
    fp = np.concatenate([m["fp"] for m in all_matches]) if all_matches else np.array([])
    num_gt = sum(m["num_gt"] for m in all_matches)
    if num_gt == 0 or len(tp) == 0:
        return 0.0
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    recalls = tp_cum / num_gt
    precisions = tp_cum / np.clip(tp_cum + fp_cum, 1e-9, None)
    return compute_ap(recalls, precisions)


def classification_prf1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict:
    precisions, recalls, f1s = [], [], []
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1
    for c in range(num_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precisions.append(precision); recalls.append(recall); f1s.append(f1)
    return {"per_class_precision": precisions, "per_class_recall": recalls,
            "per_class_f1": f1s, "confusion_matrix": confusion}


def false_reject_rate(pred_defective: np.ndarray, gt_defective: np.ndarray) -> float:
    """Fraction of GOOD garments incorrectly flagged as defective (over-rejection)."""
    good = gt_defective == 0
    if good.sum() == 0:
        return 0.0
    return float((pred_defective[good] == 1).sum() / good.sum())


def false_accept_rate(pred_defective: np.ndarray, gt_defective: np.ndarray) -> float:
    """Fraction of DEFECTIVE garments incorrectly passed as good (escapes to customer)."""
    bad = gt_defective == 1
    if bad.sum() == 0:
        return 0.0
    return float((pred_defective[bad] == 0).sum() / bad.sum())


def defect_recall(pred_defective: np.ndarray, gt_defective: np.ndarray) -> float:
    return 1.0 - false_accept_rate(pred_defective, gt_defective)
