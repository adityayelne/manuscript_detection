"""Central constants. Class order defines YOLO class ids everywhere in the repo."""
from pathlib import Path

CLASS_NAMES = ["header", "footer", "main_text", "side_text", "filler"]
CLASS_IDS = {n: i for i, n in enumerate(CLASS_NAMES)}

# BGR colours for annotated images
COLORS_BGR = {
    "header": (0, 165, 255),     # orange
    "footer": (255, 0, 255),     # magenta
    "main_text": (0, 200, 0),    # green
    "side_text": (255, 120, 0),  # blue-ish
    "filler": (0, 0, 230),       # red
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "best.pt"
