"""Label Studio ML backend: pre-labels Siamese pairs with a trained checkpoint
(boxes only -- the detector has no mask output).

    python -m labelstudio.ml_backend --weights runs/siamese_defect_v1/best.pt [--port 9090]

then connect it in the project (Settings -> Model -> Connect model, URL
http://localhost:9090) or with `setup_project.py --ml-backend`. Images are
read straight from disk using the task's data.file_name / data.reference_image,
so run it on the same machine as the data.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import yaml
from fastapi import Body, FastAPI

from annotator.assist import ModelAssistant
from annotator.store import image_size
from labelstudio.config import INSPECTED
from labelstudio.tasks import _rid


def create_app(assistant: ModelAssistant, images_dir: str, reference_dir: str,
               classes: List[str], model_version: str) -> FastAPI:
    app = FastAPI(title="Siamese ML backend")

    @app.get("/")
    @app.get("/health")
    def health():
        return {"status": "UP", "model_class": "SiameseDefectDetector"}

    @app.post("/setup")
    def setup(_: Dict = Body(None)):
        return {"model_version": model_version}

    @app.post("/webhook")
    def webhook(_: Dict = Body(None)):
        return {}

    @app.post("/predict")
    def predict(payload: Dict = Body(...)):
        results = []
        for task in payload.get("tasks", []):
            d = task["data"]
            img_path = os.path.join(images_dir, d["file_name"])
            W, H = image_size(img_path)
            proposals = assistant.propose(img_path, os.path.join(reference_dir, d["reference_image"]))
            items = []
            for p in proposals:
                x, y, w, h = p["bbox"]
                items.append({
                    "id": _rid(), "from_name": "box", "to_name": INSPECTED, "type": "rectanglelabels",
                    "original_width": W, "original_height": H, "image_rotation": 0, "score": p["score"],
                    "value": {"x": x * 100 / W, "y": y * 100 / H, "width": w * 100 / W,
                              "height": h * 100 / H, "rotation": 0,
                              "rectanglelabels": [classes[p["category_id"] - 1]]},
                })
            score = sum(i["score"] for i in items) / len(items) if items else 0.0
            results.append({"result": items, "score": score, "model_version": model_version})
        return {"results": results}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--images", help="inspected images root (default: data.images_dir)")
    parser.add_argument("--references", help="reference images root (default: data.reference_dir)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9090)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data = cfg["data"]
    app = create_app(ModelAssistant(cfg, args.weights), args.images or data["images_dir"],
                     args.references or data["reference_dir"], data["defect_classes"],
                     model_version=os.path.basename(args.weights))
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
