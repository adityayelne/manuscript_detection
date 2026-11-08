#!/usr/bin/env python
"""Draw YOLO bounding boxes and labels onto sample images for visual verification."""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mld.config import CLASS_NAMES, COLORS_BGR  # noqa: E402


def draw_yolo_boxes(img: np.ndarray, label_path: Path) -> np.ndarray:
    h, w = img.shape[:2]
    canvas = img.copy()
    if not label_path.exists():
        return canvas

    lines = label_path.read_text().strip().splitlines()
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
        x1 = max(0, int((cx - bw / 2) * w))
        y1 = max(0, int((cy - bh / 2) * h))
        x2 = min(w - 1, int((cx + bw / 2) * w))
        y2 = min(h - 1, int((cy + bh / 2) * h))

        cname = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else f"cls_{cid}"
        color = COLORS_BGR.get(cname, (255, 255, 255))

        # Draw bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # Label tag
        label_text = f"{cname}"
        font_scale = 0.6
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        tag_y1 = max(0, y1 - th - baseline - 4)
        tag_y2 = y1
        tag_x2 = min(w, x1 + tw + 6)
        cv2.rectangle(canvas, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
        # Choose text color (black for light, white for dark)
        text_color = (0, 0, 0) if (color[0] * 0.114 + color[1] * 0.587 + color[2] * 0.299) > 130 else (255, 255, 255)
        cv2.putText(canvas, label_text, (x1 + 3, tag_y2 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

    return canvas


def main():
    ap = argparse.ArgumentParser(description="Generate visual previews of YOLO labeled images")
    ap.add_argument("--yolo-dir", default=str(ROOT / "data/processed/yolo"))
    ap.add_argument("--out-dir", default=str(ROOT / "data/processed/preview"))
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    img_dir = Path(a.yolo_dir) / "images" / "train"
    lbl_dir = Path(a.yolo_dir) / "labels" / "train"
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists():
        sys.exit(f"Images directory not found: {img_dir}")

    all_images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if not all_images:
        sys.exit(f"No images found in {img_dir}")

    # Separate augmented vs normal images
    aug_images = [p for p in all_images if "_aug" in p.stem]
    norm_images = [p for p in all_images if "_aug" not in p.stem]

    rng = random.Random(a.seed)
    # Pick a mix of augmented (which include caption strips) and non-augmented
    num_aug = min(len(aug_images), a.num_samples // 2)
    num_norm = min(len(norm_images), a.num_samples - num_aug)
    selected = rng.sample(aug_images, num_aug) + rng.sample(norm_images, num_norm)
    if len(selected) < a.num_samples and len(all_images) >= a.num_samples:
        remaining = [p for p in all_images if p not in selected]
        selected += rng.sample(remaining, a.num_samples - len(selected))

    print(f"Generating {len(selected)} preview images ({num_aug} augmented, {num_norm} normal) into {out_dir}...")
    for img_path in selected:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        annotated = draw_yolo_boxes(img, lbl_path)
        out_path = out_dir / f"preview_{img_path.name}"
        cv2.imwrite(str(out_path), annotated)
        print(f"  [preview] {out_path.name} (from {img_path.name})")

    print(f"Preview generation complete. Saved {len(selected)} previews to {out_dir}")


if __name__ == "__main__":
    main()

