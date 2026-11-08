"""Draw annotated images."""
from typing import List

import cv2
import numpy as np

from .config import COLORS_BGR
from .postprocess import Detection


def draw(img_bgr: np.ndarray, dets: List[Detection]) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]
    t = max(2, int(round(min(h, w) / 300)))
    fs = max(0.5, min(h, w) / 900)
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        col = COLORS_BGR[d.label]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, t)
        txt = f"{d.label} {d.confidence:.2f}"
        (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, t - 1))
        ty1 = y1 - th - bl - 2
        if ty1 < 0:  # keep the label inside the page
            ty1 = y1 + 1
        ty2 = ty1 + th + bl + 2
        tx2 = min(w - 1, x1 + tw + 4)
        cv2.rectangle(out, (x1, ty1), (tx2, ty2), col, -1)
        cv2.putText(out, txt, (x1 + 2, ty2 - bl - 1), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (255, 255, 255), max(1, t - 1), cv2.LINE_AA)
    return out
