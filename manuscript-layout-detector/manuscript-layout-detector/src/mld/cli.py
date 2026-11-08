"""Batch inference CLI."""
import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

from . import __version__
from .config import CLASS_NAMES, DEFAULT_WEIGHTS
from .io import discover_images, read_image, write_image, write_json
from .models.detector import build_detector
from .preprocess import preprocess
from .visualize import draw

log = logging.getLogger("mld")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Detect header/footer/main_text/side_text/filler regions in manuscript images.")
    p.add_argument("--input", required=True, help="image file or folder (searched recursively)")
    p.add_argument("--output", required=True, help="output folder (annotated/, json/, summary.json)")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="YOLO weights (default weights/best.pt)")
    p.add_argument("--backend", choices=["auto", "yolo", "heuristic"], default="auto")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default=None, help="e.g. cpu, 0")
    p.add_argument("--preprocess", choices=["none", "clahe", "auto"], default="auto")
    p.add_argument("--class-conf", default="", help="per-class thresholds, e.g. header=0.3,filler=0.2")
    p.add_argument("--no-json", action="store_true")
    p.add_argument("--no-annotated", action="store_true")
    return p.parse_args(argv)


def _class_conf(s: str) -> dict:
    out = {}
    for part in filter(None, (x.strip() for x in s.split(","))):
        k, v = part.split("=")
        if k not in CLASS_NAMES:
            raise SystemExit(f"unknown class in --class-conf: {k}")
        out[k] = float(v)
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    inp, out = Path(args.input), Path(args.output)
    if not inp.exists():
        log.error("input path does not exist: %s", inp)
        return 2
    if inp.is_dir() and inp.resolve() == out.resolve():
        log.error("output directory must differ from the input directory (inputs are never modified)")
        return 2
    images = discover_images(inp, exclude_dir=out)
    if not images:
        log.error("no images found in %s", inp)
        return 2
    root = inp if inp.is_dir() else inp.parent

    try:
        det = build_detector(Path(args.weights), args.backend, conf=args.conf, iou=args.iou,
                             imgsz=args.imgsz, device=args.device, class_conf=_class_conf(args.class_conf))
    except Exception as e:
        log.error("could not build detector: %s", e)
        return 3
    log.info("detector: %s (%s) | %d image(s)", det.name, det.backend, len(images))

    totals, failures, t_all = Counter(), [], time.time()
    for i, path in enumerate(images, 1):
        rel = path.relative_to(root)
        img = read_image(path)
        if img is None:
            log.warning("[%d/%d] unreadable, skipped: %s", i, len(images), path)
            failures.append(str(rel))
            continue
        H, W = img.shape[:2]
        t0 = time.time()
        try:
            dets = det.predict(preprocess(img, args.preprocess))
        except Exception as e:
            log.warning("[%d/%d] failed on %s: %s", i, len(images), path, e)
            failures.append(str(rel))
            continue
        ms = (time.time() - t0) * 1000
        for d in dets:
            totals[d.label] += 1
        record = {
            "image": str(rel).replace("\\", "/"),
            "image_size": {"width": W, "height": H},
            "model": {"name": det.name, "backend": det.backend, "weights": det.weights,
                      "version": __version__},
            "predictions": [{
                "label": d.label, "class_id": d.class_id, "confidence": round(d.confidence, 4),
                "bbox_xyxy": list(d.bbox),
                "bbox_xywh_norm": [round(((d.bbox[0] + d.bbox[2]) / 2) / W, 5), round(((d.bbox[1] + d.bbox[3]) / 2) / H, 5),
                                   round((d.bbox[2] - d.bbox[0]) / W, 5), round((d.bbox[3] - d.bbox[1]) / H, 5)],
            } for d in dets],
            "inference": {"conf": args.conf, "iou": args.iou, "imgsz": args.imgsz,
                          "preprocess": args.preprocess, "time_ms": round(ms, 1)},
        }
        stem = rel.with_suffix("")
        if not args.no_json:
            write_json(out / "json" / (str(stem) + ".json"), record)
        if not args.no_annotated:
            write_image(out / "annotated" / (str(stem) + ".png"), draw(img, dets))
        log.info("[%d/%d] %s -> %d region(s) in %.0f ms", i, len(images), rel, len(dets), ms)

    write_json(out / "summary.json", {
        "num_images": len(images), "num_failed": len(failures), "failed": failures,
        "region_counts": {c: totals.get(c, 0) for c in CLASS_NAMES},
        "detector": {"name": det.name, "backend": det.backend, "weights": det.weights},
        "total_time_s": round(time.time() - t_all, 2),
    })
    log.info("done. results in %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
