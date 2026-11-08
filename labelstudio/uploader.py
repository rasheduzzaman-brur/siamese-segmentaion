"""Upload page for raw images + the template (golden reference) image.

    python -m labelstudio.uploader --token TOKEN [--project 1] [--port 8090]
    # open http://127.0.0.1:8090

The template is saved to data.reference_dir and becomes the current template;
every raw image uploaded afterwards is saved under data.images_dir/raw/<date>/,
paired with the current template, and added to the Label Studio project as a
task right away. Without --project a new Siamese project is created
(labelstudio/setup_project.py) and its id printed for next time.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import threading
from typing import Dict, Optional

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from annotator.store import IMAGE_EXTS, list_image_files
from labelstudio.setup_project import LabelStudioClient, create_project
from labelstudio.tasks import make_task

MAX_BYTES = 100 * 1024 * 1024
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def safe_name(name: str) -> str:
    base, ext = os.path.splitext(os.path.basename(name.replace("\\", "/")))
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._") or "image"
    ext = ext.lower()
    if ext not in IMAGE_EXTS:
        raise HTTPException(400, f"unsupported file type {ext or '(none)'}; use {sorted(IMAGE_EXTS)}")
    return base[:120] + ext


def save_unique(directory: str, name: str, body: bytes) -> tuple:
    """Write `body` as directory/name, adding _1, _2 ... on a name clash.
    Returns (file name, already_existed) -- identical content is not duplicated."""
    os.makedirs(directory, exist_ok=True)
    base, ext = os.path.splitext(name)
    digest = hashlib.sha1(body).hexdigest()
    for k in range(10000):
        candidate = name if k == 0 else f"{base}_{k}{ext}"
        path = os.path.join(directory, candidate)
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(body)
            return candidate, False
        with open(path, "rb") as f:
            if hashlib.sha1(f.read()).hexdigest() == digest:
                return candidate, True
    raise HTTPException(409, "too many files with this name")


def check_image(body: bytes):
    if not body:
        raise HTTPException(400, "empty file")
    if len(body) > MAX_BYTES:
        raise HTTPException(413, "file too large")
    img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "not a readable image")
    return img.shape[1], img.shape[0]


class Uploader:
    def __init__(self, ls: LabelStudioClient, ls_url: str, project_id: int, images_dir: str,
                 reference_dir: str, document_root: str, state_path: str):
        self.ls, self.ls_url, self.project_id = ls, ls_url, project_id
        self.images_dir, self.reference_dir, self.document_root = images_dir, reference_dir, document_root
        self.state_path = state_path
        self.lock = threading.Lock()
        self.state = {}
        if os.path.isfile(state_path):
            with open(state_path) as f:
                self.state = json.load(f)
        if not self.template or not os.path.isfile(os.path.join(reference_dir, self.template)):
            # default: the newest image already in the reference dir, if any
            refs = list_image_files(reference_dir)
            refs.sort(key=lambda r: os.path.getmtime(os.path.join(reference_dir, r)))
            self.state["template"] = refs[-1] if refs else None
            self._save_state()

    @property
    def template(self) -> Optional[str]:
        return self.state.get("template")

    def _save_state(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.state_path)), exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=1)

    def set_template(self, name: str, body: Optional[bytes] = None) -> Dict:
        with self.lock:
            if body is not None:
                w, h = check_image(body)
                name, existed = save_unique(self.reference_dir, safe_name(name), body)
            elif name not in list_image_files(self.reference_dir):
                raise HTTPException(404, f"no template {name!r}")
            self.state["template"] = name
            self._save_state()
            return self.info()

    def add_raw(self, name: str, body: bytes) -> Dict:
        if not self.template:
            raise HTTPException(400, "upload the template image first")
        check_image(body)
        batch = os.path.join("raw", datetime.date.today().isoformat())
        with self.lock:
            saved, existed = save_unique(os.path.join(self.images_dir, batch), safe_name(name), body)
            rel = f"{batch}/{saved}".replace(os.sep, "/")
            if existed:
                return {"file_name": rel, "status": "duplicate", "reference_image": self.template}
            task = make_task(rel, self.template, self.images_dir, self.reference_dir, self.document_root)
            try:
                self.ls.post(f"/api/projects/{self.project_id}/import", [task])
            except Exception as e:  # keep the file; it can be re-imported with tasks.py
                raise HTTPException(502, f"saved {rel}, but adding it to Label Studio failed: {e}")
            return {"file_name": rel, "status": "added", "reference_image": self.template}

    def info(self) -> Dict:
        project = self.ls.get(f"/api/projects/{self.project_id}")
        return {"template": self.template, "templates": list_image_files(self.reference_dir),
                "project_id": self.project_id, "project_title": project.get("title"),
                "task_count": project.get("task_number"),
                "labeled_count": project.get("num_tasks_with_annotations"),
                "label_studio_url": f"{self.ls_url}/projects/{self.project_id}/data"}


def create_app(up: Uploader) -> FastAPI:
    app = FastAPI(title="Siamese uploader")

    @app.get("/api/info")
    def info():
        return up.info()

    @app.post("/api/template")
    async def upload_template(request: Request, name: str):
        return up.set_template(name, await request.body())

    @app.put("/api/template")
    def choose_template(name: str):
        return up.set_template(name)

    @app.get("/api/template/{name:path}")
    def template_file(name: str):
        if name not in list_image_files(up.reference_dir):
            raise HTTPException(404, "no such template")
        return FileResponse(os.path.join(up.reference_dir, name))

    @app.post("/api/raw")
    async def upload_raw(request: Request, name: str):
        return up.add_raw(name, await request.body())

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "upload.html"))

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080"))
    parser.add_argument("--token", default=os.environ.get("LABEL_STUDIO_API_KEY"))
    parser.add_argument("--project", type=int, help="Label Studio project id (default: create one)")
    parser.add_argument("--title", default="Siamese defects", help="title when creating a project")
    parser.add_argument("--ml-backend", help="ML backend URL to connect when creating a project")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--images", help="inspected images root (default: data.images_dir)")
    parser.add_argument("--references", help="template images root (default: data.reference_dir)")
    parser.add_argument("--document-root", default="data")
    parser.add_argument("--state", default="data/annotations/uploader_state.json",
                        help="remembers the current template")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    if not args.token:
        parser.error("--token (or LABEL_STUDIO_API_KEY) is required")

    with open(args.config) as f:
        data = yaml.safe_load(f)["data"]
    images_dir = args.images or data["images_dir"]
    reference_dir = args.references or data["reference_dir"]
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(reference_dir, exist_ok=True)

    ls = LabelStudioClient(args.url, args.token)
    pid = args.project
    if pid is None:
        pid = create_project(ls, args.title, data["defect_classes"], images_dir, reference_dir, args.ml_backend)
        print(f"use --project {pid} next time")
    up = Uploader(ls, args.url.rstrip("/"), pid, images_dir, reference_dir, args.document_root, args.state)

    import uvicorn
    print(f"upload page: http://{args.host}:{args.port}  (Label Studio project {pid})")
    uvicorn.run(create_app(up), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
