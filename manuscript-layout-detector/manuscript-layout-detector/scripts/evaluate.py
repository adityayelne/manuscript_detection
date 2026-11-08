#!/usr/bin/env python
"""Evaluate weights on the test split and write docs/RESULTS.md (per-class P/R/mAP)."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "weights/best.pt"))
    ap.add_argument("--data", default=str(ROOT / "data/processed/yolo/data.yaml"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    from ultralytics import YOLO
    m = YOLO(a.weights)
    r = m.val(data=a.data, imgsz=a.imgsz, split=a.split, device=a.device, plots=False)
    rows = ["# Test-set results\n", f"split: `{a.split}`  |  weights: `{a.weights}`\n",
            "| class | precision | recall | mAP50 | mAP50-95 |", "|---|---|---|---|---|"]
    for i, k in enumerate(r.box.ap_class_index):
        rows.append(f"| {m.names[int(k)]} | {r.box.p[i]:.3f} | {r.box.r[i]:.3f} | {r.box.ap50[i]:.3f} | {r.box.ap[i]:.3f} |")
    rows.append(f"| **all** | {r.box.mp:.3f} | {r.box.mr:.3f} | {r.box.map50:.3f} | {r.box.map:.3f} |")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "RESULTS.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
