"""Detector backends: trained YOLO (primary) and classical heuristic (fallback)."""
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..config import CLASS_NAMES
from ..postprocess import Detection, finalize
from . import heuristics

log = logging.getLogger("mld")


class HeuristicDetector:
    name = heuristics.NAME
    backend = "heuristic"
    weights = None

    def __init__(self, class_conf: Optional[dict] = None, iou: float = 0.5, **_):
        self.class_conf, self.iou = class_conf, iou

    def predict(self, img_bgr: np.ndarray) -> List[Detection]:
        H, W = img_bgr.shape[:2]
        return finalize(heuristics.detect(img_bgr), W, H, self.iou, self.class_conf)


class YoloDetector:
    backend = "yolo"

    def __init__(self, weights: Path, conf=0.25, iou=0.5, imgsz=1280, device=None,
                 class_conf: Optional[dict] = None):
        from ultralytics import YOLO  # lazy: heavy import
        self.model = YOLO(str(weights))
        self.weights = str(weights)
        self.name = Path(weights).stem
        self.conf, self.iou, self.imgsz, self.device = conf, iou, imgsz, device
        self.class_conf = class_conf
        names = self.model.names
        names = names if isinstance(names, dict) else dict(enumerate(names))
        unknown = [n for n in names.values() if n not in CLASS_NAMES]
        if unknown:
            raise ValueError(f"weights contain classes outside the 5 target classes: {unknown}")
        self._map = {i: CLASS_NAMES.index(n) for i, n in names.items()}

    def predict(self, img_bgr: np.ndarray) -> List[Detection]:
        H, W = img_bgr.shape[:2]
        res = self.model.predict(img_bgr, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                 device=self.device, verbose=False)[0]
        dets = []
        if res.boxes is not None:
            for xyxy, c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                                  res.boxes.cls.cpu().numpy()):
                dets.append(Detection(self._map[int(k)], float(c), tuple(float(v) for v in xyxy)))
        return finalize(dets, W, H, self.iou, self.class_conf)


def build_detector(weights: Path, backend: str = "auto", **kw):
    """backend: auto (YOLO if weights load, else heuristic) | yolo | heuristic."""
    if backend == "heuristic":
        return HeuristicDetector(**kw)
    weights = Path(weights)
    if weights.exists():
        try:
            return YoloDetector(weights, **kw)
        except Exception as e:  # missing ultralytics, bad weights, wrong classes...
            if backend == "yolo":
                raise
            log.warning("Could not load YOLO weights (%s). Falling back to heuristic detector.", e)
    elif backend == "yolo":
        raise FileNotFoundError(f"weights not found: {weights}")
    else:
        log.warning("No weights at %s. Using the heuristic fallback (train a model for best results).", weights)
    return HeuristicDetector(**kw)
