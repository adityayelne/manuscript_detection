#!/usr/bin/env python
"""Offline end-to-end smoke test of train -> weights -> CLI on SYNTHETIC pages (no internet needed).
The resulting weights are meaningless for real manuscripts; this only proves the pipeline runs."""
import random
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mld.config import CLASS_NAMES  # noqa: E402
from mld.data import augment as A  # noqa: E402
from mld.data.synthetic import make_page  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data/processed/synthetic")
rng = random.Random(0)
for split, n in (("train", 48), ("val", 12)):
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img, boxes = make_page(rng)
        if rng.random() < 0.5:
            img, boxes = A.add_caption_strips(img, boxes, rng)
        h, w = img.shape[:2]
        cv2.imwrite(str(out / "images" / split / f"{i}.jpg"), img)
        (out / "labels" / split / f"{i}.txt").write_text("\n".join(
            f"{c} {(x1+x2)/2/w:.5f} {(y1+y2)/2/h:.5f} {(x2-x1)/w:.5f} {(y2-y1)/h:.5f}" for c, x1, y1, x2, y2 in boxes))
(out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/val\nnames:\n" +
                               "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)))
print("synthetic dataset ->", out / "data.yaml")
