"""Evaluate a trained Siamese PCB checkpoint on the full validation split.

    python -m scripts.evaluate_pcb --config configs/pcb.yaml \
        --weights runs/pcb_siamese_v1/best.pt --out runs/pcb_siamese_v1/eval

Every val image is run at full resolution with sliding-window tiling
(inference.tile), the same path training/validate.py uses. Reports:
  - AP50 / AP75 (pooled), mAP50 (mean of per-class AP50), AP50 per class
  - per-class precision / recall / F1 at the config score threshold, IoU 0.5
  - image-level defect detection rate (any GT box found on the board)
Writes metrics.json, predictions.json and --num-vis visualizations to --out.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset import SiameseDefectDataset, siamese_collate_fn
from datasets.transforms import SiamesePairTransform
from evaluation.metrics import box_map, match_predictions_to_gt
from inference.predict import load_model
from inference.tiling import predict_tiled


def build_val_loader(cfg: dict, split: str, workers: int) -> DataLoader:
    transform = SiamesePairTransform(tuple(cfg["data"]["image_size"]), train=False,
                                     crop=cfg["data"].get("crop"), augment=cfg["data"].get("augment"))
    dataset = SiameseDefectDataset(split, cfg["data"]["images_dir"], cfg["data"]["defect_classes"],
                                   transform=transform, train=False,
                                   reference_root=cfg["data"]["reference_dir"])
    return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=siamese_collate_fn,
                      num_workers=workers)


def draw(image_rgb: np.ndarray, preds, gt_boxes, gt_labels, names) -> np.ndarray:
    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for b, l in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, f"GT {names[l]}", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    for p in preds:
        x1, y1, x2, y2 = map(int, p["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(img, f"{names[p['class_id']]} {p['confidence']:.2f}", (x1, y2 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return img


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pcb.yaml")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default=None, help="annotation json (default: data.val_split)")
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--score-threshold", type=float, default=None,
                    help="override inference.score_threshold")
    ap.add_argument("--num-vis", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N images (smoke test)")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    icfg, tcfg = cfg["inference"], cfg["inference"]["tile"]
    thr = args.score_threshold if args.score_threshold is not None else icfg["score_threshold"]
    names = cfg["data"]["defect_classes"]
    nc = len(names)
    os.makedirs(os.path.join(args.out, "vis"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, args.weights, device)
    ckpt = torch.load(args.weights, map_location="cpu")
    print(f"device={device} weights={args.weights} epoch={ckpt.get('epoch')} "
          f"train-best-metric={ckpt.get('metric')} score_thr={thr}", flush=True)

    loader = build_val_loader(cfg, args.split or cfg["data"]["val_split"], args.workers)
    file_names = [im["file_name"] for im in loader.dataset.images] \
        if hasattr(loader.dataset, "images") else None

    m50, m75 = [], []
    pc50 = [[] for _ in range(nc)]
    tp_c, fp_c, gt_c = np.zeros(nc), np.zeros(nc), np.zeros(nc)
    boards_with_defect = boards_hit = 0
    all_preds, t0 = [], time.time()

    for i, batch in enumerate(tqdm(loader, desc="eval")):
        if args.limit and i >= args.limit:
            break
        ref, inp = batch["reference"][0].to(device), batch["input"][0].to(device)
        target = batch["targets"][0]
        preds = predict_tiled(model, ref, inp, tcfg["size"], tcfg["overlap"], thr,
                              icfg["nms_iou_threshold"], icfg["max_detections"],
                              batch_size=tcfg.get("batch_size", 8))
        gb, gl = target["boxes"].tolist(), target["labels"].tolist()
        pb = [p["bbox"] for p in preds]
        pl = [p["class_id"] for p in preds]
        ps = [p["confidence"] for p in preds]

        m50.append(match_predictions_to_gt(pb, pl, ps, gb, gl, 0.5))
        m75.append(match_predictions_to_gt(pb, pl, ps, gb, gl, 0.75))
        for c in range(nc):
            idx = [k for k, l in enumerate(pl) if l == c]
            g = [b for b, l in zip(gb, gl) if l == c]
            m = match_predictions_to_gt([pb[k] for k in idx], [c] * len(idx), [ps[k] for k in idx],
                                        g, [c] * len(g), 0.5)
            pc50[c].append(m)
            tp_c[c] += m["tp"].sum()
            fp_c[c] += m["fp"].sum()
            gt_c[c] += len(g)
        if gb:
            boards_with_defect += 1
            boards_hit += int(m50[-1]["tp"].sum() > 0)

        name = file_names[i] if file_names else str(i)
        all_preds.append({"file_name": name, "gt": [{"bbox": b, "class": names[l]} for b, l in zip(gb, gl)],
                          "pred": [{"bbox": p["bbox"], "class": names[p["class_id"]],
                                    "score": p["confidence"]} for p in preds]})
        if i < args.num_vis:
            img = (inp.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            vis = draw(img, preds, gb, gl, names)
            cv2.imwrite(os.path.join(args.out, "vis", os.path.basename(name).rsplit(".", 1)[0] + ".jpg"), vis)

    per_class = {}
    for c, n in enumerate(names):
        p = tp_c[c] / max(1, tp_c[c] + fp_c[c])
        r = tp_c[c] / max(1, gt_c[c])
        per_class[n] = {"AP50": box_map(pc50[c]), "precision": p, "recall": r,
                        "f1": 2 * p * r / max(1e-9, p + r), "tp": int(tp_c[c]),
                        "fp": int(fp_c[c]), "gt": int(gt_c[c])}
    TP, FP, GT = tp_c.sum(), fp_c.sum(), gt_c.sum()
    P, R = TP / max(1, TP + FP), TP / max(1, GT)
    ap50, ap75 = box_map(m50), box_map(m75)
    metrics = {
        "num_images": len(all_preds), "num_gt_boxes": int(GT), "score_threshold": thr,
        "AP50": ap50, "AP75": ap75, "box_mAP": (ap50 + ap75) / 2,
        "mAP50": float(np.mean([v["AP50"] for v in per_class.values()])),
        "precision": P, "recall": R, "f1": 2 * P * R / max(1e-9, P + R),
        "board_detection_rate": boards_hit / max(1, boards_with_defect),
        "seconds_per_image": (time.time() - t0) / max(1, len(all_preds)),
        "checkpoint_epoch": ckpt.get("epoch"), "per_class": per_class,
    }

    print("\n=== validation results ===")
    for k in ("num_images", "num_gt_boxes", "AP50", "AP75", "box_mAP", "mAP50",
              "precision", "recall", "f1", "board_detection_rate", "seconds_per_image"):
        v = metrics[k]
        print(f"{k:22s} {v:.4f}" if isinstance(v, float) else f"{k:22s} {v}")
    print(f"\n{'class':18s} {'AP50':>6s} {'P':>6s} {'R':>6s} {'F1':>6s} {'TP':>5s} {'FP':>5s} {'GT':>5s}")
    for n, v in per_class.items():
        print(f"{n:18s} {v['AP50']:6.3f} {v['precision']:6.3f} {v['recall']:6.3f} {v['f1']:6.3f} "
              f"{v['tp']:5d} {v['fp']:5d} {v['gt']:5d}")

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.out, "predictions.json"), "w") as f:
        json.dump(all_preds, f)
    print(f"\nwrote {args.out}/metrics.json, predictions.json, vis/")


if __name__ == "__main__":
    main()
