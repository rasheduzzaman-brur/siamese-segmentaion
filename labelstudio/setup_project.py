"""Create a Siamese annotation project in a running Label Studio.

    labelstudio/start.sh                                  # in one terminal
    python -m labelstudio.setup_project --token TOKEN \\
        [--import data/prepared/annotations/train.json] [--ml-backend http://localhost:9090]

TOKEN: Label Studio -> Account & Settings -> Personal Access Token (or a
legacy access token). Steps: create the project with the labelling template
from labelstudio/config.py, register a local-files storage for the document
root (Label Studio only serves /data/local-files/ paths covered by one),
import one task per inspected image (labelstudio/tasks.py), and optionally
connect the model-assist backend (labelstudio/ml_backend.py).
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from typing import Optional

import yaml

from labelstudio.config import build_labeling_config
from labelstudio.tasks import build_tasks


class LabelStudioClient:
    def __init__(self, url: str, token: str):
        self.url = url.rstrip("/")
        if token.count(".") == 2:  # personal access token (JWT refresh token)
            access = self._request("POST", "/api/token/refresh", {"refresh": token}, auth=None)["access"]
            self.auth = f"Bearer {access}"
        else:                      # legacy token
            self.auth = f"Token {token}"

    def _request(self, method: str, path: str, body=None, auth: Optional[str] = "default"):
        headers = {"Content-Type": "application/json"}
        if auth == "default":
            headers["Authorization"] = self.auth
        req = urllib.request.Request(self.url + path, method=method, headers=headers,
                                     data=None if body is None else json.dumps(body).encode())
        try:
            with urllib.request.urlopen(req) as res:
                raw = res.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{method} {path} -> {e.code}: {e.read().decode()[:500]}") from None
        return json.loads(raw) if raw else None

    def get(self, path):
        return self._request("GET", path)

    def post(self, path, body):
        return self._request("POST", path, body)

    def patch(self, path, body):
        return self._request("PATCH", path, body)


def create_project(ls: LabelStudioClient, title: str, classes, images_dir: str, reference_dir: str,
                   ml_backend: Optional[str] = None) -> int:
    """Empty Siamese project: template, local-files storages, optional ML backend."""
    project = ls.post("/api/projects", {
        "title": title, "label_config": build_labeling_config(classes),
        "description": "Siamese pairs: label defects on the inspected image (right) against "
                       "the reference (left). Submit with no regions, or choose 'good', for a "
                       "defect-free image.",
        "show_collab_predictions": True,
    })
    pid = project["id"]
    print(f"created project {pid}")
    # Label Studio serves /data/local-files/ only under a registered storage
    # path, which must be a subdirectory of the document root -- not the root
    for storage_title, path in (("inspected images", images_dir), ("reference images", reference_dir)):
        ls.post("/api/storages/localfiles", {
            "project": pid, "title": storage_title, "path": os.path.abspath(path),
            "use_blob_urls": True, "regex_filter": r".*\.(jpe?g|png|bmp|tiff?|webp)$",
        })
    if ml_backend:
        ls.post("/api/ml", {"url": ml_backend, "project": pid, "title": "siamese-detector"})
        ls.patch(f"/api/projects/{pid}", {"evaluate_predictions_automatically": True})
    return pid


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080"))
    parser.add_argument("--token", default=os.environ.get("LABEL_STUDIO_API_KEY"))
    parser.add_argument("--title", default="Siamese defects")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--images", help="inspected images root (default: data.images_dir)")
    parser.add_argument("--references", help="reference images root (default: data.reference_dir)")
    parser.add_argument("--document-root", default="data",
                        help="same directory as LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT in start.sh")
    parser.add_argument("--default-reference")
    parser.add_argument("--reference-map", help="json {file_name: reference_image}")
    parser.add_argument("--import", dest="imports", action="append", default=[],
                        help="existing COCO(+reference_image) json / annotator project.json (repeatable)")
    parser.add_argument("--as-annotations", action="store_true",
                        help="import existing labels as finished annotations, not pre-annotations")
    parser.add_argument("--ml-backend", help="URL of labelstudio/ml_backend.py, e.g. http://localhost:9090")
    args = parser.parse_args()
    if not args.token:
        parser.error("--token (or LABEL_STUDIO_API_KEY) is required")

    with open(args.config) as f:
        data = yaml.safe_load(f)["data"]
    images_dir = args.images or data["images_dir"]
    reference_dir = args.references or data["reference_dir"]
    classes = data["defect_classes"]

    tasks = build_tasks(images_dir, reference_dir, args.document_root, classes,
                        args.default_reference, args.reference_map, args.imports, args.as_annotations)

    ls = LabelStudioClient(args.url, args.token)
    pid = create_project(ls, args.title, classes, images_dir, reference_dir, args.ml_backend)
    ls.post(f"/api/projects/{pid}/import", tasks)

    n_pre = sum(1 for t in tasks if "predictions" in t or "annotations" in t)
    print(f"project {pid} '{args.title}': {len(tasks)} tasks ({n_pre} with imported labels)\n"
          f"open {args.url}/projects/{pid}/data")


if __name__ == "__main__":
    main()
