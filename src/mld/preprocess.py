"""In-memory enhancement for robustness (faded ink, low contrast, uneven lighting).
Always works on a copy; the file on disk is never touched."""
import cv2
import numpy as np


def clahe_enhance(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def flatten_illumination(img_bgr: np.ndarray) -> np.ndarray:
    """Divide by a heavily blurred background estimate to remove shading/vignetting."""
    k = max(31, (min(img_bgr.shape[:2]) // 8) | 1)
    bg = cv2.GaussianBlur(img_bgr, (k, k), 0).astype(np.float32) + 1.0
    out = img_bgr.astype(np.float32) / bg * 160.0
    return np.clip(out, 0, 255).astype(np.uint8)


def preprocess(img_bgr: np.ndarray, mode: str = "auto") -> np.ndarray:
    """mode: none | clahe | auto (CLAHE only when the page is low contrast)."""
    if mode == "none":
        return img_bgr.copy()
    if mode == "clahe":
        return clahe_enhance(img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if float(gray.std()) < 40.0:
        return clahe_enhance(img_bgr)
    return img_bgr.copy()
