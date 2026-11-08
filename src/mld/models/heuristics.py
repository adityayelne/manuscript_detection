"""Training-free fallback detector (classical CV).

Pipeline: ink mask (darker than local background) -> remove ruled lines and filled blobs
(binding holes, stains) -> merge ink into text blocks -> label blocks by position.
It is a baseline, NOT a replacement for the trained YOLO model. Scores are fixed rule
confidences, not calibrated probabilities.
"""
from typing import List

import cv2
import numpy as np

from ..config import CLASS_IDS
from ..postprocess import Detection, finalize

NAME = "heuristic-v1"


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    k = max(15, int(0.03 * min(h, w)) | 1)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    diff = cv2.subtract(bg, gray)
    otsu, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(22, min(90, otsu))
    return (diff > thr).astype(np.uint8)


def _remove_rules_and_blobs(ink: np.ndarray) -> np.ndarray:
    h, w = ink.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(0.12 * w)), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, int(0.12 * h))))
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, hk) | cv2.morphologyEx(ink, cv2.MORPH_OPEN, vk)
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
    out = ink & (1 - lines)
    # residual (slightly skewed / broken) rules: columns/rows that are inked over a large fraction
    vc = np.convolve((out.sum(0) > 0.35 * h).astype(np.uint8), np.ones(7, np.uint8), "same") > 0
    out[:, vc] = 0
    hr = np.convolve((out.sum(1) > 0.35 * w).astype(np.uint8), np.ones(7, np.uint8), "same") > 0
    out[hr, :] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    m = min(h, w)
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        fill = a / float(bw * bh)
        if fill > 0.6 and bw > 0.025 * m and bh > 0.025 * m and 0.6 < bw / bh < 1.6:
            out[lab == i] = 0                      # binding hole / dark stain
        elif a < 4:
            out[lab == i] = 0                      # speckle
    return out


def _split_caption_strips(ink: np.ndarray):
    """Find thin, wide, centred single text lines hugging the top/bottom edge (digitisation
    captions, printed English notes) via the row profile. Returns (ink_without_strips, boxes)."""
    h, w = ink.shape
    on = ink.sum(1) >= max(3, 0.006 * w)
    bands, st = [], None
    for y, v in enumerate(on):
        if v and st is None:
            st = y
        if not v and st is not None:
            bands.append([st, y - 1]); st = None
    if st is not None:
        bands.append([st, h - 1])
    bands = [b for b in bands if b[1] - b[0] + 1 >= max(3, 0.01 * h)]   # drop specks / leaf edges
    merged = []
    for b in bands:                       # join bands separated by tiny gaps (descenders)
        if merged and b[0] - merged[-1][1] <= max(2, 0.004 * h):
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    strips = []
    cand = [(merged[0], merged[1:2]), (merged[-1], merged[-2:-1])] if len(merged) >= 2 else []
    for b, nb in cand:
        bh = b[1] - b[0] + 1
        cy = (b[0] + b[1]) / 2
        if not nb or bh >= 0.05 * h or not (cy < 0.15 * h or cy > 0.85 * h):
            continue
        if max(m[1] - m[0] + 1 for m in merged) < 3 * bh:   # page text must be clearly taller
            continue
        cols = np.where(ink[b[0]:b[1] + 1].sum(0) > 0)[0]
        if len(cols) == 0:
            continue
        splits = np.where(np.diff(cols) > 0.03 * w)[0]
        groups = np.split(cols, splits + 1)
        g = max(groups, key=len)
        x1, x2 = int(g.min()), int(g.max())
        if (x2 - x1) > 0.2 * w and abs((x1 + x2) / 2 - w / 2) < 0.2 * w:
            strips.append((x1, max(0, b[0] - 2), x2, min(h - 1, b[1] + 3)))
    out = ink.copy()
    for x1, y1, x2, y2 in strips:
        out[max(0, y1 - 2):y2 + 3, :] = 0
    return out, strips


def _blocks(ink: np.ndarray):
    h, w = ink.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    hs = [st[i][3] for i in range(1, n) if st[i][4] > 20]
    med_h = float(np.median(hs)) if hs else 0.02 * h
    kx = max(3, int(0.04 * w))
    ky = int(np.clip(med_h, 0.012 * h, 0.05 * h))
    dil = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(dil, connectivity=8)
    blocks = []
    min_ink = max(30, int(0.0002 * h * w))
    for i in range(1, n):
        x, y, bw, bh, _ = st[i]
        sub = (lab[y:y + bh, x:x + bw] == i) & (ink[y:y + bh, x:x + bw] > 0)
        cnt = int(sub.sum())
        if cnt < min_ink:
            continue
        rr = np.where(sub.sum(1) >= max(3, 0.004 * bw))[0]   # ignore sparse speckle at the edges
        cc = np.where(sub.sum(0) >= max(3, 0.008 * bh))[0]
        if len(rr) == 0 or len(cc) == 0:
            continue
        blocks.append(dict(x1=x + cc.min(), y1=y + rr.min(), x2=x + cc.max(), y2=y + rr.max(), ink=cnt))
    return blocks


def _union(a, b):
    return dict(x1=min(a["x1"], b["x1"]), y1=min(a["y1"], b["y1"]),
                x2=max(a["x2"], b["x2"]), y2=max(a["y2"], b["y2"]), ink=a["ink"] + b["ink"])


def detect(img_bgr: np.ndarray, caption_as_filler: bool = True, max_side: int = 1400) -> List[Detection]:
    H, W = img_bgr.shape[:2]
    s = min(1.0, max_side / max(H, W))
    small = cv2.resize(img_bgr, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_AREA) if s < 1 else img_bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    ink = _remove_rules_and_blobs(_ink_mask(gray))
    dets: List[Detection] = []
    if caption_as_filler:
        ink, strips = _split_caption_strips(ink)
        dets += [Detection(CLASS_IDS["filler"], 0.55, b) for b in strips]
    blocks = _blocks(ink)
    if not blocks:
        return finalize([Detection(d.class_id, d.confidence, tuple(v / s for v in d.bbox)) for d in dets], W, H)

    rest = []
    # 1) caption / watermark strips: thin, wide, centred, hugging the top or bottom edge -> filler
    biggest = max(blocks, key=lambda b: b["ink"])
    for b in blocks:
        bw, bh = b["x2"] - b["x1"], b["y2"] - b["y1"]
        cy = (b["y1"] + b["y2"]) / 2
        cx = (b["x1"] + b["x2"]) / 2
        is_strip = (b is not biggest and bh < 0.09 * h and bw > 0.2 * w
                    and abs(cx - w / 2) < 0.2 * w and (cy < 0.12 * h or cy > 0.88 * h))
        if caption_as_filler and is_strip:
            dets.append(Detection(CLASS_IDS["filler"], 0.55, (b["x1"], b["y1"], b["x2"], b["y2"])))
        else:
            rest.append(b)
    if not rest:
        return finalize([Detection(d.class_id, d.confidence, tuple(v / s for v in d.bbox)) for d in dets], W, H)

    # 2) main text = block with the most ink, absorbing neighbours (e.g. split by a binding hole)
    main = max(rest, key=lambda b: b["ink"])
    others = [b for b in rest if b is not main]
    changed = True
    while changed:
        changed = False
        for b in list(others):
            xo = min(main["x2"], b["x2"]) - max(main["x1"], b["x1"])
            yo = min(main["y2"], b["y2"]) - max(main["y1"], b["y1"])
            gap_x = -xo if xo < 0 else 0
            gap_y = -yo if yo < 0 else 0
            bw, bh = b["x2"] - b["x1"] + 1, b["y2"] - b["y1"] + 1
            inside_x = xo >= 0.5 * bw
            inside_y = yo >= 0.5 * bh
            if (inside_x and gap_y < 0.05 * h) or (inside_y and gap_x < 0.04 * w and bw > 0.08 * w):
                main = _union(main, b)
                others.remove(b)
                changed = True
    dets.append(Detection(CLASS_IDS["main_text"], 0.60, (main["x1"], main["y1"], main["x2"], main["y2"])))

    # 3) everything else by position relative to the main block
    for b in others:
        cx, cy = (b["x1"] + b["x2"]) / 2, (b["y1"] + b["y2"]) / 2
        if b["x2"] <= main["x1"] or b["x1"] >= main["x2"]:
            cls = "side_text"
        elif b["y2"] <= main["y1"] + 0.02 * h:
            cls = "header"
        elif b["y1"] >= main["y2"] - 0.02 * h:
            cls = "footer"
        else:
            cls = "header" if cy < h / 2 else "footer"
        dets.append(Detection(CLASS_IDS[cls], 0.50, (b["x1"], b["y1"], b["x2"], b["y2"])))

    scaled = [Detection(d.class_id, d.confidence, tuple(v / s for v in d.bbox)) for d in dets]
    return finalize(scaled, W, H)
