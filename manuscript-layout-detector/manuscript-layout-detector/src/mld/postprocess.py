"""Box utilities: clipping to page bounds, IoU, NMS."""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .config import CLASS_NAMES


@dataclass
class Detection:
    class_id: int
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 in ORIGINAL image pixels

    @property
    def label(self) -> str:
        return CLASS_NAMES[self.class_id]


def clip_box(box: Sequence[float], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    """Clip to [0, W-1] x [0, H-1] and round to ints. Returns None if degenerate."""
    x1, y1, x2, y2 = box
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = int(max(0, min(width - 1, round(x1))))
    x2 = int(max(0, min(width - 1, round(x2))))
    y1 = int(max(0, min(height - 1, round(y1))))
    y2 = int(max(0, min(height - 1, round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def class_aware_nms(dets: List[Detection], iou_thr: float = 0.5) -> List[Detection]:
    keep: List[Detection] = []
    for d in sorted(dets, key=lambda x: -x.confidence):
        if all(k.class_id != d.class_id or iou(k.bbox, d.bbox) < iou_thr for k in keep):
            keep.append(d)
    return keep


def finalize(dets: List[Detection], width: int, height: int, iou_thr: float = 0.5,
             class_conf: Optional[dict] = None) -> List[Detection]:
    """Clip every box to the page, drop degenerate ones, apply per-class thresholds and NMS."""
    out: List[Detection] = []
    for d in dets:
        if class_conf and d.confidence < class_conf.get(d.label, 0.0):
            continue
        b = clip_box(d.bbox, width, height)
        if b is None:
            continue
        out.append(Detection(d.class_id, float(d.confidence), b))
    out = class_aware_nms(out, iou_thr)
    return sorted(out, key=lambda d: (d.bbox[1], d.bbox[0]))
