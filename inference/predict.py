"""Single-image inference entrypoint.

    python -m inference.predict --config configs/config.yaml --weights runs/.../best.pt \
        --reference path/to/golden.png --input path/to/captured.png

Produces the structured instance list described in the design brief:

    Instance 1:
        Class = Print Crack
        Confidence = 0.96
        Bounding Box = (...)
        Mask = (...)

Also reports the Siamese embedding distance (used for unknown/unseen-defect
flagging, see DESIGN.md section 12). This module assumes the input image was
already T-shirt-detected, aligned (ORB+RANSAC) and reference-selected
upstream -- see DESIGN.md section 13 for where this fits in the full hybrid
pipeline.
"""
from __future__ import annotations

import argparse
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from models.siamese_instance_segmentation import SiameseInstanceSegmentation


def load_model(cfg: dict, weights_path: str, device: torch.device) -> SiameseInstanceSegmentation:
    model = SiameseInstanceSegmentation(cfg).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt.get("ema_model", ckpt.get("model", ckpt)))
    model.eval()
    return model


def preprocess(image_bgr: np.ndarray, size) -> torch.Tensor:
    h, w = size
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)


def paste_mask_on_image(mask_logits: torch.Tensor, box: List[float], image_size) -> np.ndarray:
    h, w = image_size
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((h, w), dtype=np.uint8)
    mask_prob = torch.sigmoid(mask_logits).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(mask_prob, size=(y2 - y1, x2 - x1), mode="bilinear", align_corners=False)
    full = np.zeros((h, w), dtype=np.uint8)
    full[y1:y2, x1:x2] = (resized.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    return full


@torch.no_grad()
def predict(model: SiameseInstanceSegmentation, reference_bgr: np.ndarray, input_bgr: np.ndarray,
            cfg: dict, device: torch.device) -> Dict:
    size = tuple(cfg["data"]["image_size"])
    ref_t = preprocess(reference_bgr, size).to(device)
    inp_t = preprocess(input_bgr, size).to(device)

    outputs = model(ref_t, inp_t)
    results = model.postprocess(
        outputs, size,
        score_thresh=cfg["inference"]["score_threshold"],
        nms_iou=cfg["inference"]["nms_iou_threshold"],
        max_dets=cfg["inference"]["max_detections"],
    )

    class_names = cfg["data"]["defect_classes"]
    instances = []
    for r in results:
        instances.append({
            "class": class_names[r["class_id"]],
            "confidence": round(r["confidence"], 4),
            "bounding_box": [round(v, 1) for v in r["bbox"]],
            "mask": paste_mask_on_image(r["mask_logits"], r["bbox"], size),
        })

    embedding_distance = F.pairwise_distance(
        outputs["reference_embedding"], outputs["inspected_embedding"]).item()
    is_anomalous_unseen = (embedding_distance > cfg["inference"]["anomaly_embedding_threshold"]
                            and len(instances) == 0)

    return {
        "embedding_distance": embedding_distance,
        "unknown_defect_suspected": is_anomalous_unseen,
        "instances": instances,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, args.weights, device)

    reference_bgr = cv2.imread(args.reference)
    input_bgr = cv2.imread(args.input)
    result = predict(model, reference_bgr, input_bgr, cfg, device)

    for inst in result["instances"]:
        inst_display = {k: v for k, v in inst.items() if k != "mask"}
        print(inst_display)
    print(f"embedding_distance={result['embedding_distance']:.3f} "
          f"unknown_defect_suspected={result['unknown_defect_suspected']}")
