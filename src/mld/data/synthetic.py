"""Tiny synthetic page generator, used ONLY for offline smoke tests of the train/infer pipeline."""
import random
from typing import List, Tuple

import cv2
import numpy as np

from ..config import CLASS_IDS


def _line_block(img, x1, y1, x2, y2, rng, line_h=22, ink=(30, 30, 30)):
    y = y1
    while y + line_h <= y2:
        x = x1
        while x < x2 - 20:
            wl = rng.randint(15, 70)
            cv2.rectangle(img, (x, y + 4), (min(x2, x + wl), y + line_h - 4), ink, -1)
            x += wl + rng.randint(6, 14)
        y += line_h + 6


def make_page(rng: random.Random, w=900, h=640) -> Tuple[np.ndarray, List[tuple]]:
    img = np.full((h, w, 3), rng.choice([(215, 225, 235), (190, 210, 225), (235, 235, 235)]), np.uint8)
    boxes = []
    mx1, mx2 = int(w * 0.18), int(w * 0.82)
    my1, my2 = int(h * 0.18), int(h * 0.84)
    _line_block(img, mx1, my1, mx2, my2, rng)
    boxes.append((CLASS_IDS["main_text"], mx1, my1, mx2, my2))
    if rng.random() < 0.8:
        _line_block(img, mx1 + 80, 20, mx2 - 80, 20 + 26, rng)
        boxes.append((CLASS_IDS["header"], mx1 + 80, 20, mx2 - 80, 46))
    if rng.random() < 0.8:
        _line_block(img, mx1 + 160, my2 + 20, mx2 - 160, my2 + 46, rng)
        boxes.append((CLASS_IDS["footer"], mx1 + 160, my2 + 20, mx2 - 160, my2 + 46))
    if rng.random() < 0.7:
        x1, x2 = (10, mx1 - 15) if rng.random() < 0.5 else (mx2 + 15, w - 10)
        y1 = rng.randint(my1, my2 - 150)
        _line_block(img, x1, y1, x2, y1 + 120, rng, ink=(60, 60, 110))
        boxes.append((CLASS_IDS["side_text"], x1, y1, x2, y1 + 120))
    return img, boxes
