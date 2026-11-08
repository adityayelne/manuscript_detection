#!/usr/bin/env python
"""Train YOLO11 on the built dataset and copy the best checkpoint to weights/best.pt.

  python train.py --data data/processed/yolo/data.yaml --model yolo11s.pt --epochs 60 --imgsz 1280 --batch 8
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/processed/yolo/data.yaml"))
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="layout")
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.model)
    res = model.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, workers=a.workers,
        fraction=a.fraction,
        seed=a.seed, project=str(ROOT / "runs"), name=a.name, exist_ok=True, patience=20,
        fliplr=0.0, flipud=0.0, degrees=3.0, translate=0.1, scale=0.4, hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        mosaic=1.0, close_mosaic=10, cos_lr=True, plots=True,
    )
    best = Path(res.save_dir) / "weights" / "best.pt"
    (ROOT / "weights").mkdir(exist_ok=True)
    shutil.copy(best, ROOT / "weights" / "best.pt")
    print("best weights copied to weights/best.pt")


if __name__ == "__main__":
    main()
