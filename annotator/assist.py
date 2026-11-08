"""Model-assisted pre-labelling with a trained checkpoint (boxes only -- the
detector has no mask output). Wraps inference/predict.py and maps its boxes
from the model's resized input (cfg data.image_size, plain resize, see
inference.predict.preprocess) back to original-image pixels."""
from __future__ import annotations

import threading
from typing import Dict, List

import cv2


class ModelAssistant:
    def __init__(self, cfg: Dict, weights_path: str):
        self.cfg = cfg
        self.weights_path = weights_path
        self._model = None
        self._device = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            import torch
            from inference.predict import load_model
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = load_model(self.cfg, self.weights_path, self._device)
        return self._model

    def propose(self, image_path: str, reference_path: str, score_threshold: float = None) -> List[Dict]:
        from inference.predict import predict
        inp = cv2.imread(image_path, cv2.IMREAD_COLOR)
        ref = cv2.imread(reference_path, cv2.IMREAD_COLOR)
        if inp is None or ref is None:
            raise FileNotFoundError(image_path if inp is None else reference_path)
        cfg = self.cfg
        if score_threshold is not None:
            cfg = {**cfg, "inference": {**cfg["inference"], "score_threshold": score_threshold}}
        with self._lock:
            result = predict(self._load(), ref, inp, cfg, self._device)

        size_h, size_w = cfg["data"]["image_size"]
        H, W = inp.shape[:2]
        sx, sy = W / size_w, H / size_h
        class_names = cfg["data"]["defect_classes"]
        proposals = []
        for inst in result["instances"]:
            x0, y0, x1, y1 = inst["bounding_box"]
            proposals.append({
                "category_id": class_names.index(inst["class"]) + 1,
                "shape": "box", "source": "model", "score": inst["confidence"],
                "bbox": [x0 * sx, y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy],
            })
        return proposals
