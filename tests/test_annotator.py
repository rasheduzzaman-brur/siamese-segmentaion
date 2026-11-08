"""Annotator geometry + store/export round-trip through the real dataset loader."""
import base64
import json
import os

import cv2
import numpy as np
import pytest

from annotator import geometry
from annotator.export import export
from annotator.store import ProjectStore

CLASSES = ["crack", "ghosting", "spot"]


def test_clip_bbox():
    assert geometry.clip_bbox([-10, -5, 30, 20], 100, 100) == [0, 0, 20, 15]
    assert geometry.clip_bbox([90, 90, 50, 50], 100, 100) == [90, 90, 10, 10]
    assert geometry.clip_bbox([200, 200, 10, 10], 100, 100) is None
    assert geometry.clip_bbox([10, 10, -5, -5], 100, 100) == [5, 5, 5, 5]


def test_mask_polygon_bbox_roundtrip():
    mask = np.zeros((120, 160), dtype=bool)
    mask[20:60, 30:90] = True
    mask[80:100, 100:150] = True
    polys = geometry.mask_to_polygons(mask)
    assert len(polys) == 2
    x, y, w, h = geometry.polygons_bbox(polys)
    assert (x, y) == (30, 20) and abs((x + w) - 149) <= 1 and abs((y + h) - 99) <= 1
    back = geometry.rasterize(polys, 160, 120).astype(bool)
    iou = (back & mask).sum() / (back | mask).sum()
    assert iou > 0.95


def _png_data_url(mask: np.ndarray) -> str:
    rgba = np.zeros(mask.shape + (4,), dtype=np.uint8)
    rgba[mask] = (0, 0, 255, 255)
    ok, buf = cv2.imencode(".png", rgba)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


@pytest.fixture
def project(tmp_path):
    images, refs = tmp_path / "images", tmp_path / "reference"
    (images / "train").mkdir(parents=True)
    refs.mkdir()
    for i in range(4):
        cv2.imwrite(str(images / "train" / f"img{i}.jpg"), np.full((100, 150, 3), 40 * i, np.uint8))
    cv2.imwrite(str(refs / "golden.png"), np.zeros((100, 150, 3), np.uint8))
    store = ProjectStore(str(tmp_path / "project.json"), str(images), str(refs), CLASSES)
    return store, tmp_path


def test_store_and_export_load_in_dataset(project):
    store, tmp = project
    ids = [img["id"] for img in store.data["images"]]
    assert len(ids) == 4 and all(i["reference_image"] == "golden.png" for i in store.data["images"])

    mask = np.zeros((100, 150), dtype=bool)
    mask[10:40, 20:70] = True
    store.update_image(ids[0], {"annotations": [
        {"category_id": 1, "shape": "box", "bbox": [5, 5, 20, 30]},
        {"category_id": 3, "shape": "polygon", "segmentation": [[60, 10, 90, 10, 90, 50, 60, 50]]},
        {"category_id": 2, "shape": "mask", "mask_png": _png_data_url(mask)},
        {"category_id": 1, "shape": "box", "bbox": [0, 0, 10, 10], "source": "model", "score": 0.7},
    ]})
    store.update_image(ids[1], {"status": "good", "annotations": []})
    store.update_image(ids[2], {"annotations": [{"category_id": 2, "shape": "box", "bbox": [1, 1, 5, 5]}]})

    anns = store.annotations_for(ids[0])
    poly = next(a for a in anns if a["shape"] == "polygon")
    assert poly["bbox"] == [60, 10, 30, 40]
    m = next(a for a in anns if a["shape"] == "mask")
    assert m["segmentation"] and abs(m["bbox"][0] - 20) <= 1 and abs(m["bbox"][1] - 10) <= 1

    # reload from disk
    store2 = ProjectStore(store.path, store.images_dir, store.reference_dir, CLASSES)
    assert [i["status"] for i in store2.data["images"]] == ["labeled", "good", "labeled", "unlabeled"]

    out = tmp / "annotations"
    res = export(store2.data, str(out), val_ratio=0.0, test_ratio=0.0, seed=0,
                 masks_dir=str(tmp / "masks"))
    assert res["images"] == 3 and res["annotations"] == 4 and res["unreviewed_model_skipped"] == 1
    sem = cv2.imread(str(tmp / "masks" / "train" / "img0.png"), cv2.IMREAD_UNCHANGED)
    assert sem[20, 40] == 2 and sem[30, 75] == 3 and sem[90, 140] == 0

    coco = json.load(open(out / "train.json"))
    assert not any("status" in i for i in coco["images"])
    assert not any("shape" in a or "source" in a for a in coco["annotations"])

    torch = pytest.importorskip("torch")
    from datasets.dataset import SiameseDefectDataset
    ds = SiameseDefectDataset(str(out / "train.json"), store.images_dir, CLASSES,
                              reference_root=store.reference_dir)
    by_id = {ds.images[k]["id"]: ds[k]["target"] for k in range(len(ds))}
    t0 = by_id[ids[0]]
    assert t0["pair_label"] == 1 and sorted(t0["labels"].tolist()) == [0, 1, 2]
    assert torch.allclose(t0["boxes"][0], torch.tensor([5., 5., 25., 35.]))
    assert by_id[ids[1]]["pair_label"] == 0 and len(by_id[ids[1]]["boxes"]) == 0


def test_import_existing_coco(project, tmp_path):
    store, _ = project
    coco = {"categories": [{"id": 0, "name": "gdd"}, {"id": 1, "name": "crack"}, {"id": 3, "name": "spot"}],
            "images": [{"id": 7, "file_name": "train/img3.jpg", "reference_image": "golden.png"}],
            "annotations": [{"id": 1, "image_id": 7, "category_id": 3, "bbox": [1, 2, 3, 4],
                             "segmentation": [], "area": 12}]}
    p = tmp_path / "in.json"
    p.write_text(json.dumps(coco))
    assert store.import_coco(str(p)) == {"images": 1, "annotations": 1, "skipped": 0}
    img = next(i for i in store.data["images"] if i["file_name"] == "train/img3.jpg")
    (a,) = store.annotations_for(img["id"])
    assert img["status"] == "labeled" and a["category_id"] == 3 and a["shape"] == "box"
