"""Shape helpers shared by the store, the export and the tests.

All coordinates are in original-image pixels; boxes are COCO xywh unless the
name says otherwise. Polygons use the COCO flat layout [x1, y1, x2, y2, ...].
"""
from __future__ import annotations

import base64
from typing import List, Optional, Sequence

import cv2
import numpy as np

Polygon = List[float]


def clip_bbox(bbox: Sequence[float], width: int, height: int) -> Optional[List[float]]:
    """Clip an xywh box to the image; None if nothing of it is left."""
    x, y, w, h = (float(v) for v in bbox)
    if w < 0:
        x, w = x + w, -w
    if h < 0:
        y, h = y + h, -h
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(width), x + w), min(float(height), y + h)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)]


def clip_polygon(poly: Sequence[float], width: int, height: int) -> Optional[Polygon]:
    pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return None
    pts[:, 0] = pts[:, 0].clip(0, width)
    pts[:, 1] = pts[:, 1].clip(0, height)
    if polygon_area(pts.reshape(-1).tolist()) < 1:
        return None
    return [round(float(v), 2) for v in pts.reshape(-1)]


def polygon_area(poly: Sequence[float]) -> float:
    pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)


def polygons_bbox(polys: Sequence[Sequence[float]]) -> List[float]:
    pts = np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in polys])
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return [round(float(x0), 2), round(float(y0), 2),
            round(float(x1 - x0), 2), round(float(y1 - y0), 2)]


def decode_mask_png(data_url: str) -> np.ndarray:
    """Browser canvas.toDataURL() PNG -> boolean HxW mask (alpha > 0)."""
    b64 = data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url
    buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("mask is not a decodable PNG")
    if img.ndim == 3 and img.shape[2] == 4:
        return img[:, :, 3] > 0
    if img.ndim == 3:
        return img.max(axis=2) > 0
    return img > 0


def mask_to_polygons(mask: np.ndarray, epsilon: float = 1.0, min_area: float = 4.0) -> List[Polygon]:
    """Outer contours of a binary mask as COCO polygons. Holes are not kept
    (COCO polygon segmentation has no hole semantics)."""
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        approx = cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        polys.append([float(v) for v in approx.reshape(-1)])
    return polys


def rasterize(polys: Sequence[Sequence[float]], width: int, height: int,
              value: int = 1, out: Optional[np.ndarray] = None) -> np.ndarray:
    if out is None:
        out = np.zeros((height, width), dtype=np.uint8)
    pts = [np.round(np.asarray(p).reshape(-1, 2)).astype(np.int32) for p in polys]
    if pts:
        cv2.fillPoly(out, pts, int(value))
    return out
