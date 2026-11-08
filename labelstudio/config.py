"""Label Studio labelling template for Siamese pairs.

Both images of a pair are shown side by side; regions are drawn only on the
inspected image. Each defect class appears in three label groups (one per
tool) with the same colour:

    box   RectangleLabels  detection                     hotkeys 1..9
    poly  PolygonLabels    segmentation                  hotkeys q..o
    brush BrushLabels      segmentation (paint/erase)    hotkeys z..m

`pair` (defect / good) marks a reviewed image with no defects as a negative
pair (hotkeys d / g); an annotation submitted with no regions is treated as
good as well.

    # one template for everything, images uploaded with Label Studio's Import button:
    python -m labelstudio.config --template path/to/golden.png --output siamese_template.xml
    # per-task pairing (tasks from tasks.py / setup_project.py / uploader.py):
    python -m labelstudio.config > labeling_config.xml

Paste the XML into Project -> Settings -> Labeling Interface -> Code.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
from typing import Dict, List, Optional
from xml.sax.saxutils import quoteattr

import cv2
import yaml

# same palette as annotator/static/app.js so both tools colour classes alike
PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
           "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990"]
TOOLS = [  # (control tag, name, title, hotkeys)
    ("RectangleLabels", "box", "Box (detection)", "123456789"),
    ("PolygonLabels", "poly", "Polygon (segmentation)", "qwertyuio"),
    ("BrushLabels", "brush", "Brush (segmentation)", "zxcvbnm"),
]
FROM_NAMES = {name: tag for tag, name, _, _ in TOOLS}
INSPECTED, REFERENCE, PAIR = "inspected", "reference", "pair"


def build_labeling_config(classes: List[str], template: Optional[Dict[str, str]] = None) -> str:
    """template=None: per-task pairing, the task data carries $reference
    (tasks.py / uploader.py). template={"name", "src"}: one fixed template
    embedded in the interface, so plain image uploads through Label Studio's
    own Import button work (each task only has $image)."""
    if template:
        header = f"Template (golden reference): {template['name']}"
        reference_value = template["src"]
    else:
        header = "$image_name  ↔  reference: $reference_image"
        reference_value = "$reference"
    groups = []
    for tag, name, title, keys in TOOLS:
        labels = "\n".join(
            f'        <Label value={quoteattr(c)} background="{PALETTE[i % len(PALETTE)]}"'
            + (f' hotkey="{keys[i]}"' if i < len(keys) else "") + "/>"
            for i, c in enumerate(classes))
        extra = ' strokeWidth="2"' if tag == "RectangleLabels" else ""
        groups.append(f'''      <Header value="{title}" size="6"/>
      <{tag} name="{name}" toName="{INSPECTED}"{extra}>
{labels}
      </{tag}>''')
    groups_xml = "\n".join(groups)
    return f'''<View>
  <Header value={quoteattr(header)} size="6"/>
  <Choices name="{PAIR}" toName="{INSPECTED}" choice="single" showInline="true">
    <Choice value="defect" hotkey="d"/>
    <Choice value="good" hotkey="g" hint="Reviewed, no defects: exported as a negative pair"/>
  </Choices>
  <View style="display: flex; gap: 16px; align-items: flex-start">
    <View style="flex: 1 1 0; min-width: 0">
      <Header value="Reference (golden)" size="5"/>
      <Image name="{REFERENCE}" value={quoteattr(reference_value)} zoom="true" zoomControl="true"/>
    </View>
    <View style="flex: 1 1 0; min-width: 0">
      <Header value="Inspected: label defects here" size="5"/>
      <Image name="{INSPECTED}" value="$image" zoom="true" zoomControl="true"
             brightnessControl="true" contrastControl="true" rotateControl="false"/>
    </View>
    <View style="width: 220px">
{groups_xml}
    </View>
  </View>
</View>
'''


TEMPLATE_STATE = "data/annotations/siamese_template.json"


def embed_template(path: str, reference_dir: str, max_side: int = 2048) -> Dict[str, str]:
    """Copy the template into reference_dir (so the export can pair with it)
    and return it as a display-sized JPEG data URI for the interface."""
    name = os.path.basename(path)
    dst = os.path.join(reference_dir, name)
    if os.path.abspath(path) != os.path.abspath(dst):
        os.makedirs(reference_dir, exist_ok=True)
        if os.path.exists(dst) and open(dst, "rb").read() != open(path, "rb").read():
            raise SystemExit(f"{dst} exists with different content; rename the template")
        shutil.copy2(path, dst)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read template image {path}")
    scale = min(1.0, max_side / max(img.shape[:2]))
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {"name": name, "src": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--template", help="golden reference image to embed: every image uploaded "
                                           "through Label Studio's Import is paired with it")
    parser.add_argument("--output", help="write the XML here instead of stdout")
    args = parser.parse_args()
    with open(args.config) as f:
        data = yaml.safe_load(f)["data"]
    template = embed_template(args.template, data["reference_dir"]) if args.template else None
    xml = build_labeling_config(data["defect_classes"], template)
    if args.output:
        with open(args.output, "w") as f:
            f.write(xml)
    else:
        print(xml, end="")
    if template:
        # export_coco.py reads this to pair uploaded images with the template
        os.makedirs(os.path.dirname(TEMPLATE_STATE), exist_ok=True)
        with open(TEMPLATE_STATE, "w") as f:
            json.dump({"reference_image": template["name"]}, f)
        print(f"template {template['name']} -> {data['reference_dir']}"
              + (f"; labelling interface written to {args.output}" if args.output else ""),
              file=sys.stderr)


if __name__ == "__main__":
    main()
