"""Read-only image discovery/loading and JSON writing. Inputs are never modified."""
import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .config import IMAGE_EXTS


def discover_images(path: Path, exclude_dir: Optional[Path] = None) -> List[Path]:
    path = Path(path)
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTS else []
    ex = exclude_dir.resolve() if exclude_dir else None
    files = []
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            if ex is not None and ex in p.resolve().parents:
                continue
            files.append(p)
    return files


def read_image(path: Path) -> Optional[np.ndarray]:
    """BGR uint8 image or None if unreadable. Opens the file read-only."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)  # also works for non-ASCII paths
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    try:
        from PIL import Image
        with Image.open(path) as im:
            return cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def write_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise IOError(f"could not encode {path}")
    buf.tofile(str(path))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
