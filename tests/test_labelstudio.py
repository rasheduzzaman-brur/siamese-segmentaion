"""Label Studio integration: template, RLE codec, tasks and export conversion."""
import json
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest

from labelstudio import rle
from labelstudio.config import build_labeling_config
from labelstudio.export_coco import ls_to_project
from labelstudio.tasks import build_tasks

CLASSES = ["crack", "ghosting", "spot"]


def test_labeling_config_is_valid_xml_with_all_tools():
    root = ET.fromstring(build_labeling_config(CLASSES))
    names = {el.get("name"): el.tag for el in root.iter() if el.get("name")}
    assert names["reference"] == "Image" and names["inspected"] == "Image"
    for tool in ("box", "poly", "brush"):
        el = root.find(f".//*[@name='{tool}']")
        assert el.get("toName") == "inspected"
        assert [l.get("value") for l in el] == CLASSES
    hotkeys = [el.get("hotkey") for el in root.iter() if el.get("hotkey")]
    assert len(hotkeys) == len(set(hotkeys))


def test_rle_roundtrip():
    rng = np.random.default_rng(0)
    mask = rng.random((37, 53)) > 0.6
    mask[5:30, 10:40] = True
    assert (rle.decode(rle.encode(mask), 53, 37) == mask).all()


@pytest.fixture
def data(tmp_path):
    images, refs = tmp_path / "data" / "images", tmp_path / "data" / "reference"
    (images / "train").mkdir(parents=True)
    refs.mkdir(parents=True)
    for i in range(3):
        cv2.imwrite(str(images / "train" / f"img{i}.jpg"), np.zeros((100, 200, 3), np.uint8))
    cv2.imwrite(str(refs / "golden.png"), np.zeros((100, 200, 3), np.uint8))
    return tmp_path, images, refs


def test_build_tasks_with_imported_labels(data):
    tmp, images, refs = data
    coco = {"categories": [{"id": 1, "name": "crack"}, {"id": 3, "name": "spot"}],
            "images": [{"id": 0, "file_name": "train/img0.jpg", "width": 200, "height": 100,
                        "reference_image": "golden.png"},
                       {"id": 1, "file_name": "train/img1.jpg", "width": 200, "height": 100,
                        "reference_image": "golden.png"}],
            "annotations": [{"id": 1, "image_id": 0, "category_id": 3, "bbox": [20, 10, 40, 50]}]}
    (tmp / "c.json").write_text(json.dumps(coco))
    tasks = build_tasks(str(images), str(refs), str(tmp / "data"), CLASSES, imports=[str(tmp / "c.json")])
    assert len(tasks) == 3
    t0 = tasks[0]
    assert t0["data"]["image"] == "/data/local-files/?d=images/train/img0.jpg"
    assert t0["data"]["reference"] == "/data/local-files/?d=reference/golden.png"
    box = next(r for r in t0["predictions"][0]["result"] if r["type"] == "rectanglelabels")
    assert box["value"]["x"] == 10 and box["value"]["height"] == 50 and box["value"]["rectanglelabels"] == ["spot"]
    assert tasks[1]["predictions"][0]["result"][0]["value"]["choices"] == ["good"]
    assert "predictions" not in tasks[2]


def test_ls_export_to_project(data):
    tmp, images, _ = data
    mask = np.zeros((100, 200), bool)
    mask[20:40, 100:150] = True
    common = {"to_name": "inspected", "original_width": 200, "original_height": 100}
    ls = [
        {"id": 1, "data": {"file_name": "train/img0.jpg", "reference_image": "golden.png"},
         "annotations": [
             {"id": 1, "was_cancelled": False, "updated_at": "2026-01-01", "result": [
                 {"from_name": "box", "type": "rectanglelabels", **common,
                  "value": {"x": 10, "y": 10, "width": 20, "height": 30, "rotation": 0, "rectanglelabels": ["crack"]}},
                 {"from_name": "poly", "type": "polygonlabels", **common,
                  "value": {"points": [[50, 50], [60, 50], [60, 80]], "polygonlabels": ["spot"]}},
                 {"from_name": "brush", "type": "brushlabels", **common,
                  "value": {"format": "rle", "rle": rle.encode(mask), "brushlabels": ["ghosting"]}}]},
             {"id": 2, "was_cancelled": True, "updated_at": "2026-02-01", "result": []}]},
        {"id": 2, "data": {"file_name": "train/img1.jpg", "reference_image": "golden.png"},
         "annotations": [{"id": 3, "result": []}]},
        {"id": 3, "data": {"file_name": "train/img2.jpg", "reference_image": "golden.png"}, "annotations": []},
    ]
    project, stats = ls_to_project(ls, CLASSES, str(images))
    assert stats == {"tasks": 3, "labeled": 1, "good": 1, "unlabeled": 1, "regions": 3, "skipped_regions": 0}
    by_cat = {a["category_id"]: a for a in project["annotations"]}
    assert by_cat[1]["bbox"] == [20, 10, 40, 30] and by_cat[1]["segmentation"] == []
    assert by_cat[3]["bbox"] == [100, 50, 20, 30]
    x, y, w, h = by_cat[2]["bbox"]
    assert (x, y) == (100, 20) and abs(w - 49) <= 1 and abs(h - 19) <= 1


def test_uploader_file_safety(tmp_path):
    from fastapi import HTTPException
    from labelstudio.uploader import safe_name, save_unique
    assert safe_name("../../etc/My Photo (1).JPG") == "My_Photo_1.jpg"
    assert safe_name("C:\\x\\..\\a.png") == "a.png"
    with pytest.raises(HTTPException):
        safe_name("evil.sh")
    assert save_unique(str(tmp_path), "a.png", b"one") == ("a.png", False)
    assert save_unique(str(tmp_path), "a.png", b"one") == ("a.png", True)      # same bytes: duplicate
    assert save_unique(str(tmp_path), "a.png", b"two") == ("a_1.png", False)   # name clash: renamed


def test_native_upload_tasks_use_embedded_template(data, tmp_path):
    from labelstudio.config import embed_template
    _, images, refs = data
    tpl = embed_template(str(refs / "golden.png"), str(refs))
    root = ET.fromstring(build_labeling_config(CLASSES, tpl))
    assert root.find(".//Image[@name='reference']").get("value").startswith("data:image/jpeg;base64,")

    media = tmp_path / "ls" / "media" / "upload" / "5"
    media.mkdir(parents=True)
    cv2.imwrite(str(media / "ab12-shirt.jpg"), np.zeros((100, 200, 3), np.uint8))
    ls = [{"id": 1, "data": {"image": "/data/upload/5/ab12-shirt.jpg"}, "annotations": [{"id": 1, "result": [
        {"from_name": "box", "type": "rectanglelabels", "to_name": "inspected", "original_width": 200,
         "original_height": 100, "value": {"x": 0, "y": 0, "width": 50, "height": 50, "rectanglelabels": ["crack"]}}]}]}]
    project, _ = ls_to_project(ls, CLASSES, str(images), ls_data_dir=str(tmp_path / "ls"),
                               reference_image="golden.png")
    (img,) = project["images"]
    assert img["file_name"] == "ls_upload/5/ab12-shirt.jpg" and img["reference_image"] == "golden.png"
    assert (images / "ls_upload" / "5" / "ab12-shirt.jpg").is_file()
    with pytest.raises(ValueError, match="reference"):
        ls_to_project(ls, CLASSES, str(images), ls_data_dir=str(tmp_path / "ls"))


def test_pairing_names():
    from labelstudio.pairing import match_group, original_name, reference_group
    assert original_name("/data/upload/3/8f2a91c0-shirt01_ref.png") == "shirt01_ref.png"
    assert reference_group("shirt01_ref.png") == "shirt01"
    assert reference_group("Shirt02-Template.PNG") == "shirt02"
    assert reference_group("polo.v2 golden.jpg") == "polo.v2"
    assert reference_group("shirt01_front.jpg") is None
    assert reference_group("ref.png") is None
    groups = ["shirt01", "shirt01_blue", "shirt02"]
    assert match_group("shirt01_front.jpg", groups) == "shirt01"
    assert match_group("shirt01_blue-3.jpg", groups) == "shirt01_blue"   # longest prefix wins
    assert match_group("shirt010.jpg", groups) is None                    # needs a separator
    assert match_group("other.jpg", groups) is None
    assert match_group("anything.jpg", ["only"]) == "only"                # single reference
