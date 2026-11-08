"""FastAPI app: JSON API over ProjectStore + the static single-page UI."""
from __future__ import annotations

import os
from typing import Dict, Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from annotator import export as export_mod
from annotator.store import ProjectStore

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def _safe_path(root: str, rel: str) -> str:
    root = os.path.realpath(root)
    path = os.path.realpath(os.path.join(root, rel))
    if os.path.commonpath([root, path]) != root or not os.path.isfile(path):
        raise HTTPException(404, f"not found: {rel}")
    return path


def create_app(store: ProjectStore, export_defaults: Dict, assistant=None) -> FastAPI:
    app = FastAPI(title="Siamese annotator")

    @app.get("/api/project")
    def project():
        return {**store.summary(), "assist_available": assistant is not None,
                "export_defaults": export_defaults}

    @app.get("/api/image/{image_id}")
    def get_image(image_id: int):
        try:
            img = store.get_image(image_id)
        except KeyError:
            raise HTTPException(404, "no such image")
        return {"image": img, "annotations": store.annotations_for(image_id)}

    @app.put("/api/image/{image_id}")
    def put_image(image_id: int, payload: Dict = Body(...)):
        try:
            return store.update_image(image_id, payload)
        except KeyError:
            raise HTTPException(404, "no such image")
        except (ValueError, KeyError, TypeError) as e:
            raise HTTPException(400, str(e))

    @app.post("/api/rescan")
    def rescan():
        return {"added": store.scan()}

    @app.post("/api/assist/{image_id}")
    def assist(image_id: int, payload: Optional[Dict] = Body(None)):
        if assistant is None:
            raise HTTPException(400, "start the annotator with --weights to enable model assist")
        img = store.get_image(image_id)
        if not img.get("reference_image"):
            raise HTTPException(400, "image has no reference image")
        thr = (payload or {}).get("score_threshold")
        proposals = assistant.propose(_safe_path(store.images_dir, img["file_name"]),
                                      _safe_path(store.reference_dir, img["reference_image"]),
                                      score_threshold=thr)
        return {"proposals": proposals}

    @app.post("/api/export")
    def do_export(payload: Optional[Dict] = Body(None)):
        opts = {**export_defaults, **(payload or {})}
        with store.lock:
            return export_mod.export(
                store.data, opts["annotations_dir"],
                val_ratio=float(opts["val_ratio"]), test_ratio=float(opts["test_ratio"]),
                seed=int(opts["seed"]), include_unreviewed=bool(opts.get("include_unreviewed")),
                masks_dir=opts["masks_dir"] if opts.get("semantic_masks") else None,
                copy_to=opts.get("copy_to") or None,
                images_dir=store.images_dir, reference_dir=store.reference_dir)

    @app.get("/files/images/{rel:path}")
    def image_file(rel: str):
        return FileResponse(_safe_path(store.images_dir, rel))

    @app.get("/files/references/{rel:path}")
    def reference_file(rel: str):
        return FileResponse(_safe_path(store.reference_dir, rel))

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
