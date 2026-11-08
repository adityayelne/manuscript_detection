#!/usr/bin/env python
"""Build a 5-class YOLO detection dataset from CATMuS (+ optional HORAE / custom labels).

  python scripts/build_dataset.py                      # CATMuS from the HF hub / local cache
  python scripts/build_dataset.py --catmus-dir data/raw/catmus
  python scripts/build_dataset.py --horae-dir data/raw/horae_lsv2 --horae-filler-ids 3,4,5
  python scripts/build_dataset.py --custom-dir data/raw/indic_custom   # your own labelled pages

Output: data/processed/yolo/{images,labels}/{train,val,test} + data.yaml
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mld.config import CLASS_NAMES  # noqa: E402
from mld.data import augment as A  # noqa: E402
from mld.data.mapping import catmus_class  # noqa: E402
from mld.io import read_image  # noqa: E402


def write_sample(out: Path, split: str, name: str, img: np.ndarray, boxes, stats: Counter):
    h, w = img.shape[:2]
    lines = []
    for c, x1, y1, x2, y2 in boxes:
        x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        lines.append(f"{int(c)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        stats[(split, CLASS_NAMES[int(c)])] += 1
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))


def shrink(img, boxes, max_side):
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s == 1.0:
        return img, boxes
    img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img, [(c, x1 * s, y1 * s, x2 * s, y2 * s) for c, x1, y1, x2, y2 in boxes]


def augment_variant(img, boxes, rng):
    if rng.random() < 0.25:
        img, boxes = A.wide_crop(img, boxes, rng)
    if rng.random() < 0.30:
        img, boxes = A.pad_background(img, boxes, rng)
    if rng.random() < 0.40:
        img, boxes = A.add_hard_negatives(img, boxes, rng)
    if rng.random() < 0.15:
        img, boxes = A.add_pencil_text(img, boxes, rng)
    if rng.random() < 0.60:           # English caption strips: the key failure mode in the assignment samples
        img, boxes = A.add_caption_strips(img, boxes, rng)
    return img, boxes


def _objects(rec, cat_names):
    objs = rec["objects"]
    if isinstance(objs, dict):  # HF Sequence-of-dict layout: dict of lists
        keys = list(objs.keys())
        objs = [dict(zip(keys, vals)) for vals in zip(*[objs[k] for k in keys])]
    for o in objs:
        cat = o["category"]
        if isinstance(cat, int) and cat_names:
            cat = cat_names[cat]
        yield cat, o.get("type", "block"), o["bbox"]


def build_catmus(a, out, stats, rng):
    import datasets
    from PIL import Image as PILImage
    catmus_path = a.catmus_dir if a.catmus_dir else "CATMuS/medieval-segmentation"
    ds = datasets.load_dataset(catmus_path)
    print("CATMuS splits:", {k: len(v) for k, v in ds.items()}, "columns:", ds[list(ds.keys())[0]].column_names)
    split_map = {"train": "train", "dev": "val", "validation": "val", "val": "val", "test": "test"}
    dropped = Counter()
    for sp, data in ds.items():
        if sp not in split_map:
            continue
        tgt = split_map[sp]
        img_col = next((c for c, f in data.features.items() if getattr(f, "_type", "") == "Image" or str(type(f)).lower().endswith("image'>")), None) or "image"
        if img_col in data.features and isinstance(data.features[img_col], datasets.Image):
            data = data.cast_column(img_col, datasets.Image(decode=False))
        try:
            cat_names = data.features["objects"].feature["category"].names
        except Exception:
            try:
                cat_names = data.features["objects"]["category"].feature.names
            except Exception:
                cat_names = None
        count_written = 0
        for i, rec in enumerate(data):
            if a.limit and count_written >= a.limit:
                break
            img_data = rec[img_col]
            if isinstance(img_data, dict) and "path" in img_data:
                img_path = Path(img_data["path"]) if img_data.get("path") else None
                if not img_path or not img_path.exists():
                    continue
                pil = PILImage.open(img_path).convert("RGB")
            elif hasattr(img_data, "convert"):
                pil = img_data.convert("RGB")
            else:
                continue
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            H = img.shape[0]
            boxes = []
            for cat, typ, bb in _objects(rec, cat_names):
                if typ != "block":
                    continue
                c = catmus_class(cat, bb, H)
                if c is None:
                    dropped[str(cat)] += 1
                    continue
                boxes.append((c, *bb))
            if not boxes:
                continue
            img, boxes = shrink(img, boxes, a.max_side)
            name = f"catmus_{tgt}_{i:05d}"
            write_sample(out, tgt, name, img, boxes, stats)
            count_written += 1
            if tgt == "train" and rng.random() < a.aug_prob:
                for k in range(a.aug_copies):
                    aimg, aboxes = augment_variant(img, boxes, rng)
                    write_sample(out, tgt, f"{name}_aug{k}", aimg, aboxes, stats)
    print("dropped CATMuS categories:", dict(dropped))


def build_horae(a, out, stats, rng):
    ids = {int(x) for x in a.horae_filler_ids.split(",") if x.strip()}
    root = Path(a.horae_dir)
    imgs = {p.stem: p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    n = 0
    for lab in root.rglob("*.txt"):
        if lab.stem not in imgs or lab.name.lower() == "readme.txt":
            continue
        img = read_image(imgs[lab.stem])
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = []
        for line in lab.read_text().splitlines():
            t = line.split()
            if len(t) < 5 or int(float(t[0])) not in ids:
                continue
            v = list(map(float, t[1:]))
            if len(v) == 4:
                cx, cy, bw, bh = v
                x1, y1, x2, y2 = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
            else:
                xs, ys = v[0::2], v[1::2]
                x1, x2, y1, y2 = min(xs) * w, max(xs) * w, min(ys) * h, max(ys) * h
            boxes.append((4, x1, y1, x2, y2))
        if boxes:
            img, boxes = shrink(img, boxes, a.max_side)
            write_sample(out, "train", f"horae_{lab.stem}", img, boxes, stats)
            n += 1
    print(f"HORAE: added {n} images (filler only)")


def build_custom(a, out, stats):
    root = Path(a.custom_dir)
    n = 0
    for ip in sorted((root / "images").glob("*")):
        lp = root / "labels" / (ip.stem + ".txt")
        img = read_image(ip)
        if img is None or not lp.exists():
            continue
        h, w = img.shape[:2]
        boxes = []
        for line in lp.read_text().splitlines():
            c, cx, cy, bw, bh = line.split()[:5]
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            boxes.append((int(c), cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
        img, boxes = shrink(img, boxes, a.max_side)
        for r in range(a.custom_repeat):          # upweight in-domain pages
            write_sample(out, "train", f"custom_{ip.stem}_r{r}", img, boxes, stats)
        n += 1
    print(f"custom: added {n} images x{a.custom_repeat}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/processed/yolo"))
    default_catmus = str(ROOT / "data/raw/catmus") if (ROOT / "data/raw/catmus").exists() else None
    ap.add_argument("--catmus-dir", default=default_catmus)
    ap.add_argument("--no-catmus", action="store_true")
    ap.add_argument("--horae-dir", default=None)
    ap.add_argument("--horae-filler-ids", default="", help="comma-separated HORAE class ids to map to filler")
    ap.add_argument("--custom-dir", default=None)
    ap.add_argument("--custom-repeat", type=int, default=3)
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--aug-prob", type=float, default=0.7)
    ap.add_argument("--aug-copies", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="debug: max images per split")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    stats = Counter()
    if not a.no_catmus:
        build_catmus(a, out, stats, rng)
    if a.horae_dir:
        if not a.horae_filler_ids:
            sys.exit("--horae-filler-ids is required: read data/raw/horae_lsv2/README.md and list the ids of "
                     "decoration / initials / rubrication / miniature classes")
        build_horae(a, out, stats, rng)
    if a.custom_dir:
        build_custom(a, out, stats)

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)))
    lines = ["# Dataset statistics (instances per split)\n", "| split | " + " | ".join(CLASS_NAMES) + " |",
             "|---|" + "---|" * len(CLASS_NAMES)]
    for sp in ("train", "val", "test"):
        lines.append(f"| {sp} | " + " | ".join(str(stats[(sp, c)]) for c in CLASS_NAMES) + " |")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "DATASET_STATS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("data.yaml ->", out / "data.yaml")


if __name__ == "__main__":
    main()
