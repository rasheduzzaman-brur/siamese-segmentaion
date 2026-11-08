"""Single-image inference entrypoint.

    python -m inference.predict --config configs/config.yaml --weights runs/.../best.pt \
        --reference path/to/golden.png --input path/to/captured.png [--save-vis out.jpg]

Produces the structured instance list described in the design brief:

    Instance 1:
        Class = Print Crack
        Confidence = 0.96
        Bounding Box = (...)

Boxes are reported in the *original* input-image pixel coordinates.

Two modes, matching training (see datasets/transforms.py):
  - letterbox (default): reference and input are each letterboxed to
    data.image_size, exactly like the eval transform used in training.
  - tiled (inference.tile set, e.g. configs/pcb.yaml): sliding-window over the
    full-resolution aligned pair (inference/tiling.py). Reference and input
    must be the same size.

Also reports the Siamese embedding distance (used for unknown/unseen-defect
flagging, see DESIGN.md section 12; in tiled mode the max over tiles). This
module assumes the input image was already detected, aligned (ORB+RANSAC) and
reference-selected upstream -- see DESIGN.md section 13 for where this fits in
the full hybrid pipeline.
"""
from __future__ import annotations

import argparse
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from inference.tiling import predict_tiled
from models.siamese_detector import SiameseDefectDetector


def load_model(cfg: dict, weights_path: str, device: torch.device) -> SiameseDefectDetector:
    model = SiameseDefectDetector(cfg).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt.get("ema_model", ckpt.get("model", ckpt)))
    model.eval()
    return model


def to_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).float()[None] / 255.0


def letterbox(image_bgr: np.ndarray, size) -> Tuple[torch.Tensor, float, Tuple[int, int]]:
    """Same geometry as A.LongestMaxSize + A.PadIfNeeded (centered pad) used in
    training. Returns (tensor, scale, (pad_x, pad_y))."""
    h, w = size
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    ih, iw = image.shape[:2]
    scale = max(h, w) / max(ih, iw)
    nh, nw = int(round(ih * scale)), int(round(iw * scale))
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA)
    pad_y, pad_x = max(0, h - nh) // 2, max(0, w - nw) // 2
    canvas = np.zeros((max(h, nh), max(w, nw), 3), dtype=np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = image
    return to_tensor(canvas), scale, (pad_x, pad_y)


@torch.no_grad()
def predict(model: SiameseDefectDetector, reference_bgr: np.ndarray, input_bgr: np.ndarray,
            cfg: dict, device: torch.device) -> Dict:
    icfg = cfg["inference"]
    tile_cfg = icfg.get("tile")

    if tile_cfg:
        if reference_bgr.shape[:2] != input_bgr.shape[:2]:
            raise ValueError("tiled inference needs a reference aligned to (same size as) the "
                             f"input, got {reference_bgr.shape[:2]} vs {input_bgr.shape[:2]}")
        ref_t = to_tensor(cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB)).to(device)
        inp_t = to_tensor(cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)).to(device)
        results, embedding_distance = predict_tiled(
            model, ref_t, inp_t, tile_cfg["size"], tile_cfg["overlap"],
            icfg["score_threshold"], icfg["nms_iou_threshold"], icfg["max_detections"],
            batch_size=tile_cfg.get("batch_size", 8), return_embedding_distance=True)
    else:
        size = tuple(cfg["data"]["image_size"])
        ref_t, _, _ = letterbox(reference_bgr, size)
        inp_t, scale, (pad_x, pad_y) = letterbox(input_bgr, size)
        outputs = model(ref_t.to(device), inp_t.to(device))
        results = model.postprocess(outputs, inp_t.shape[-2:],
                                    score_thresh=icfg["score_threshold"],
                                    nms_iou=icfg["nms_iou_threshold"],
                                    max_dets=icfg["max_detections"])
        ih, iw = input_bgr.shape[:2]
        for r in results:  # letterboxed -> original pixel coordinates
            x1, y1, x2, y2 = r["bbox"]
            r["bbox"] = [min(max((x1 - pad_x) / scale, 0), iw), min(max((y1 - pad_y) / scale, 0), ih),
                         min(max((x2 - pad_x) / scale, 0), iw), min(max((y2 - pad_y) / scale, 0), ih)]
        embedding_distance = F.pairwise_distance(
            outputs["reference_embedding"], outputs["inspected_embedding"]).item()

    class_names = cfg["data"]["defect_classes"]
    instances = [{"class": class_names[r["class_id"]],
                  "confidence": round(r["confidence"], 4),
                  "bounding_box": [round(v, 1) for v in r["bbox"]]} for r in results]

    is_anomalous_unseen = (embedding_distance > icfg["anomaly_embedding_threshold"]
                           and len(instances) == 0)
    return {
        "embedding_distance": embedding_distance,
        "unknown_defect_suspected": is_anomalous_unseen,
        "instances": instances,
    }


def draw(image_bgr: np.ndarray, instances) -> np.ndarray:
    vis = image_bgr.copy()
    thick = max(2, round(max(vis.shape[:2]) / 800))
    for inst in instances:
        x1, y1, x2, y2 = map(int, inst["bounding_box"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), thick)
        cv2.putText(vis, f"{inst['class']} {inst['confidence']:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * thick, (0, 0, 255), thick)
    return vis


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--save-vis", help="write the input image with predicted boxes drawn")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, args.weights, device)

    reference_bgr = cv2.imread(args.reference)
    input_bgr = cv2.imread(args.input)
    if reference_bgr is None or input_bgr is None:
        raise FileNotFoundError(f"could not read {args.reference if reference_bgr is None else args.input}")
    result = predict(model, reference_bgr, input_bgr, cfg, device)

    for inst in result["instances"]:
        print(inst)
    print(f"embedding_distance={result['embedding_distance']:.3f} "
          f"unknown_defect_suspected={result['unknown_defect_suspected']}")
    if args.save_vis:
        cv2.imwrite(args.save_vis, draw(input_bgr, result["instances"]))
        print(f"wrote {args.save_vis}")
