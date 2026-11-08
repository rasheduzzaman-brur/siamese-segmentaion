"""Pair references and defect images uploaded through Label Studio's own Import.

Label Studio's direct file upload only works for a labelling interface with a
single image, so uploading uses two projects:

    "Siamese inbox"  drag references AND defect images in here (Import button)
    "Siamese pairs"  where you label: each defect image next to its reference

This worker watches the inbox. Every reference it sees is remembered (and
copied into data.reference_dir/ls_upload/); every defect image is moved to the
pairs project as a task showing it next to its reference, then removed from
the inbox. Pairing is by file name:

    shirt01_ref.png        reference of group "shirt01"  (also -ref, .ref,
                           _reference, _template, _golden; any case)
    shirt01_front.jpg      defect image -> paired with shirt01_ref.png
    shirt01-2.jpg          (group name followed by _ - . or space)

If only one reference exists, every defect image pairs with it. A defect
image whose reference hasn't arrived yet waits in the inbox (its "pairing"
column says why) and moves as soon as the reference is uploaded.

    python -m labelstudio.pairing --token TOKEN
    # first run creates both projects and remembers them in
    # data/annotations/ls_pairing.json; keep it running while uploading
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

import yaml

from labelstudio.config import build_labeling_config
from labelstudio.setup_project import LabelStudioClient

REF_MARKER = re.compile(r"^(?P<group>.+?)[ ._-]+(ref|reference|template|golden)$", re.IGNORECASE)
UPLOAD_PREFIX = re.compile(r"^[0-9a-f]{8}-")
INBOX_CONFIG = """<View>
  <Header value="Inbox: upload references (e.g. shirt01_ref.png) and defect images (e.g. shirt01_front.jpg). They move to the pairs project automatically: label there, not here." size="6"/>
  <Image name="image" value="$image" zoom="true"/>
  <Choices name="inbox" toName="image" visibleWhen="no-region-selected">
    <Choice value="waiting for pairing"/>
  </Choices>
</View>
"""


def original_name(url: str) -> str:
    """'/data/upload/3/8f2a91c0-shirt01_ref.png' -> 'shirt01_ref.png'."""
    return UPLOAD_PREFIX.sub("", os.path.basename(unquote(urlparse(url).path)))


def reference_group(name: str) -> Optional[str]:
    """Group name if `name` is a reference image, else None."""
    m = REF_MARKER.match(os.path.splitext(name)[0])
    return m.group("group").lower() if m else None


def match_group(name: str, groups: List[str]) -> Optional[str]:
    """Longest reference group that prefixes the defect image's name; the only
    group if there is exactly one."""
    stem = os.path.splitext(name)[0].lower()
    hits = [g for g in groups if stem == g or re.match(re.escape(g) + r"[ ._-]", stem)]
    if hits:
        return max(hits, key=len)
    return groups[0] if len(groups) == 1 else None


class Pairer:
    def __init__(self, ls: LabelStudioClient, inbox: int, pairs: int, ls_data_dir: str,
                 reference_dir: str, state: Dict, state_path: str):
        self.ls, self.inbox, self.pairs = ls, inbox, pairs
        self.ls_data_dir, self.reference_dir = ls_data_dir, reference_dir
        self.state, self.state_path = state, state_path
        self.state.setdefault("references", {})
        self.noted: Dict[int, str] = {}  # inbox task id -> pairing note already written

    def save_state(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.state_path)), exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=1)

    def _copy_reference(self, url: str, name: str) -> str:
        """Copy an uploaded reference out of Label Studio's media dir into
        reference_dir; returns its path relative to reference_dir."""
        src = os.path.join(self.ls_data_dir, "media", "upload", unquote(urlparse(url).path)[len("/data/upload/"):])
        if not os.path.isfile(src):
            raise FileNotFoundError(f"uploaded reference not found at {src} -- is --ls-data-dir "
                                    f"Label Studio's base data dir?")
        with open(src, "rb") as f:
            digest = hashlib.sha1(f.read()).hexdigest()
        base, ext = os.path.splitext(name)
        for k in range(1000):
            rel = f"ls_upload/{self.inbox}/{name if k == 0 else f'{base}_{k}{ext}'}"
            dst = os.path.join(self.reference_dir, rel)
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                return rel
            with open(dst, "rb") as f:
                if hashlib.sha1(f.read()).hexdigest() == digest:
                    return rel
        raise RuntimeError(f"too many references named {name}")

    def _inbox_tasks(self) -> List[Dict]:
        tasks, page = [], 1
        while True:
            res = self.ls.get(f"/api/tasks?project={self.inbox}&page={page}&page_size=100")
            batch = res["tasks"] if isinstance(res, dict) else res
            tasks.extend(batch)
            if len(batch) < 100:
                return tasks
            page += 1

    def _note(self, task: Dict, url: str, note: str):
        if self.noted.get(task["id"]) != note:
            self.ls.patch(f"/api/tasks/{task['id']}", {"data": {"image": url, "pairing": note}})
            self.noted[task["id"]] = note

    def step(self) -> Tuple[int, int]:
        """One pass over the inbox -> (references registered, images paired)."""
        tasks = self._inbox_tasks()
        refs = self.state["references"]
        n_refs = 0
        # references first, so images uploaded in the same batch find them
        for t in sorted(tasks, key=lambda t: reference_group(original_name(self._url(t))) is None):
            url = self._url(t)
            if url is None:
                continue
            name = original_name(url)
            group = reference_group(name)
            if group is None:
                continue
            refs[group] = {"url": url, "reference_image": self._copy_reference(url, name), "name": name}
            self.save_state()
            self.ls._request("DELETE", f"/api/tasks/{t['id']}/")
            n_refs += 1
            print(f"reference '{group}': {name}")

        paired, done = [], []
        for t in tasks:
            url = self._url(t)
            if url is None:
                continue
            name = original_name(url)
            if reference_group(name) is not None:
                continue
            group = match_group(name, sorted(refs))
            if group is None:
                stem = os.path.splitext(name)[0]
                hint = (f"no matching reference: upload {stem}_ref.png, or rename this image "
                        f"<reference name>_{stem}") if refs else "waiting for a reference upload (e.g. shirt01_ref.png)"
                self._note(t, url, hint)
                continue
            ref = refs[group]
            paired.append({"data": {"image": url, "reference": ref["url"], "image_name": name,
                                    "reference_image": ref["reference_image"], "pair_group": group}})
            done.append(t["id"])
            print(f"paired {name} -> {ref['name']}")
        if paired:
            self.ls.post(f"/api/projects/{self.pairs}/import", paired)
            for tid in done:
                self.ls._request("DELETE", f"/api/tasks/{tid}/")
                self.noted.pop(tid, None)
        return n_refs, len(paired)

    @staticmethod
    def _url(task: Dict) -> Optional[str]:
        url = task.get("data", {}).get("image")
        return url if isinstance(url, str) and url.startswith("/data/upload/") else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080"))
    parser.add_argument("--token", default=os.environ.get("LABEL_STUDIO_API_KEY"))
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--references", help="where references are copied (default: data.reference_dir)")
    parser.add_argument("--ls-data-dir", default=os.environ.get("LABEL_STUDIO_BASE_DATA_DIR", "data/label-studio"),
                        help="Label Studio base data dir (labelstudio/start.sh: data/label-studio)")
    parser.add_argument("--state", default="data/annotations/ls_pairing.json",
                        help="remembers the two project ids and the references seen")
    parser.add_argument("--title", default="Siamese", help="project name prefix when creating them")
    parser.add_argument("--interval", type=float, default=2.0, help="seconds between inbox checks")
    parser.add_argument("--once", action="store_true", help="process the inbox once and exit")
    args = parser.parse_args()
    if not args.token:
        parser.error("--token (or LABEL_STUDIO_API_KEY) is required")

    with open(args.config) as f:
        data = yaml.safe_load(f)["data"]
    state = {}
    if os.path.isfile(args.state):
        with open(args.state) as f:
            state = json.load(f)
    ls = LabelStudioClient(args.url, args.token)
    if "inbox" not in state:
        state["inbox"] = ls.post("/api/projects", {"title": f"{args.title} inbox",
                                                   "label_config": INBOX_CONFIG})["id"]
        state["pairs"] = ls.post("/api/projects", {
            "title": f"{args.title} pairs", "show_collab_predictions": True,
            "label_config": build_labeling_config(data["defect_classes"])})["id"]
    pairer = Pairer(ls, state["inbox"], state["pairs"], args.ls_data_dir,
                    args.references or data["reference_dir"], state, args.state)
    pairer.save_state()
    base = args.url.rstrip("/")
    print(f"upload into: {base}/projects/{state['inbox']}  (Siamese inbox)\n"
          f"label in:    {base}/projects/{state['pairs']}  (Siamese pairs)\n"
          f"{len(state['references'])} reference(s) known; watching the inbox (Ctrl+C to stop)")
    while True:
        try:
            pairer.step()
        except Exception as e:  # keep watching; Label Studio may be restarting
            print(f"error: {e}")
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
