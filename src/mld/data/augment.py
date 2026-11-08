"""Synthetic augmentations aimed at the assignment's failure modes.

Boxes are (cls, x1, y1, x2, y2) in pixels. All functions return (new_img, new_boxes)
and never touch files on disk.
"""
import random
from typing import List, Tuple

import cv2
import numpy as np

from ..config import CLASS_IDS

Box = Tuple[int, float, float, float, float]
FILLER = CLASS_IDS["filler"]

CAPTIONS = [
    "CC-0. In Public Domain. Digitization by eGangotri. Funding MIDF",
    "CSU-Sringeri Campus Rare Book and Manuscript Collection, Sringeri",
    "Digitized by the National Mission for Manuscripts",
    "Public Domain - Scanned Copy - Folio {n}",
    "Library Collection, Manuscript No. {n}",
    "Property of the State Oriental Library",
]


def _border_color(img: np.ndarray) -> Tuple[int, int, int]:
    edge = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    return tuple(int(v) for v in np.median(edge, axis=0))


def add_caption_strips(img: np.ndarray, boxes: List[Box], rng: random.Random):
    """Add English caption strips (top and/or bottom) as NEW rows and label them filler."""
    h, w = img.shape[:2]
    which = rng.choice(["top", "bottom", "both"])
    out, new = img, list(boxes)
    for side in (["top", "bottom"] if which == "both" else [which]):
        text = rng.choice(CAPTIONS).format(n=rng.randint(1, 999))
        strip_h = max(24, int(h * rng.uniform(0.04, 0.07)))
        bg = rng.choice([_border_color(out), (255, 255, 255), (235, 235, 235), (160, 120, 90)])
        strip = np.full((strip_h, out.shape[1], 3), bg, np.uint8)
        scale = strip_h * 0.55 / 22.0
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        while tw > out.shape[1] * 0.9 and scale > 0.3:
            scale *= 0.9
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        lum = int(np.mean(bg))
        col = (0, 0, 0) if lum > 110 else (255, 255, 255)
        x0 = (out.shape[1] - tw) // 2 + rng.randint(-int(0.05 * out.shape[1]), int(0.05 * out.shape[1]))
        x0 = max(2, min(out.shape[1] - tw - 2, x0))
        y0 = (strip_h + th) // 2
        cv2.putText(strip, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)
        box_s = (FILLER, max(0, x0 - 3), max(0, y0 - th - 3), min(out.shape[1] - 1, x0 + tw + 3), min(strip_h - 1, y0 + bl + 3))
        if side == "top":
            out = np.vstack([strip, out])
            new = [(c, a, b + strip_h, d, e + strip_h) for c, a, b, d, e in new]
            new.append(box_s)
        else:
            y_off = out.shape[0]
            out = np.vstack([out, strip])
            new.append((box_s[0], box_s[1], box_s[2] + y_off, box_s[3], box_s[4] + y_off))
    return out, new


def _overlaps(b, boxes, thr=0.0):
    for _, x1, y1, x2, y2 in boxes:
        if min(b[2], x2) - max(b[0], x1) > thr and min(b[3], y2) - max(b[1], y1) > thr:
            return True
    return False


def add_pencil_text(img: np.ndarray, boxes: List[Box], rng: random.Random):
    """Faint grey pencil-like scribble in a margin, labelled filler."""
    h, w = img.shape[:2]
    for _ in range(8):
        bw, bh = int(w * rng.uniform(0.04, 0.12)), int(h * rng.uniform(0.02, 0.08))
        x1 = rng.randint(0, max(0, w - bw - 1))
        y1 = rng.randint(0, max(0, h - bh - 1))
        cand = (x1, y1, x1 + bw, y1 + bh)
        if not _overlaps(cand, boxes):
            break
    else:
        return img, boxes
    layer = np.zeros(img.shape[:2], np.uint8)
    n_chars = max(3, bw // max(6, bh // 2))
    cw = bw / n_chars
    for i in range(n_chars):
        pts = [(int(x1 + i * cw + rng.uniform(0, cw)), int(y1 + rng.uniform(0, bh))) for _ in range(rng.randint(3, 6))]
        cv2.polylines(layer, [np.array(pts, np.int32)], False, 255, rng.randint(1, 2), cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (3, 3), 0).astype(np.float32) / 255.0 * rng.uniform(0.25, 0.5)
    grey = np.full_like(img, rng.randint(90, 140), np.uint8)
    out = (img * (1 - layer[..., None]) + grey * layer[..., None]).astype(np.uint8)
    return out, boxes + [(FILLER, x1, y1, x1 + bw, y1 + bh)]


def add_hard_negatives(img: np.ndarray, boxes: List[Box], rng: random.Random):
    """Binding hole, ruled line(s) and dust. Deliberately NOT labelled."""
    out = img.copy()
    h, w = out.shape[:2]
    m = min(h, w)
    if rng.random() < 0.6:
        r = int(m * rng.uniform(0.02, 0.04))
        cv2.circle(out, (int(w * rng.uniform(0.25, 0.75)), int(h * rng.uniform(0.3, 0.7))), r, (40, 35, 30), -1, cv2.LINE_AA)
    if rng.random() < 0.5:
        y = int(h * rng.uniform(0.3, 0.7))
        cv2.line(out, (int(w * 0.05), y), (int(w * 0.95), y), (60, 60, 60), 1, cv2.LINE_AA)
    if rng.random() < 0.5:
        for x in (int(w * rng.uniform(0.02, 0.06)), int(w * rng.uniform(0.94, 0.98))):
            cv2.line(out, (x, 0), (x, h - 1), (50, 40, 60), 1, cv2.LINE_AA)
    for _ in range(rng.randint(0, 6)):
        cv2.circle(out, (rng.randint(0, w - 1), rng.randint(0, h - 1)), rng.randint(1, 3), (50, 50, 50), -1)
    return out, boxes


def pad_background(img: np.ndarray, boxes: List[Box], rng: random.Random):
    """Surround the page by a solid background (photographed palm-leaf look)."""
    h, w = img.shape[:2]
    t, b = int(h * rng.uniform(0.03, 0.2)), int(h * rng.uniform(0.03, 0.2))
    l, r = int(w * rng.uniform(0.03, 0.15)), int(w * rng.uniform(0.03, 0.15))
    col = rng.choice([(150, 110, 80), (40, 40, 40), (90, 110, 130), (60, 80, 50)])
    out = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=col)
    return out, [(c, x1 + l, y1 + t, x2 + l, y2 + t) for c, x1, y1, x2, y2 in boxes]


def wide_crop(img: np.ndarray, boxes: List[Box], rng: random.Random):
    """Crop a horizontal band (aspect 2:1-3:1). Rejected if it would cut a box partially."""
    h, w = img.shape[:2]
    target = rng.uniform(2.0, 3.0)
    ch = int(w / target)
    if ch >= h * 0.9:
        return img, boxes
    for _ in range(10):
        y0 = rng.randint(0, h - ch)
        new, ok = [], True
        for c, x1, y1, x2, y2 in boxes:
            ny1, ny2 = max(y1, y0), min(y2, y0 + ch)
            kept = max(0, ny2 - ny1) / max(1, y2 - y1)
            if 0.02 < kept < 0.9:
                ok = False
                break
            if kept >= 0.9:
                new.append((c, x1, ny1 - y0, x2, ny2 - y0))
        if ok and new:
            return img[y0:y0 + ch], new
    return img, boxes
