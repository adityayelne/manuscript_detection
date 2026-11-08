"""Map source-dataset taxonomies onto the five target classes."""
from typing import Optional

from ..config import CLASS_IDS

# CATMuS Medieval Segmentation (SegmOnto zone types; line types are ignored by the caller).
CATMUS_MAP = {
    "runningtitlezone": "header",
    "quiremarkszone": "footer",
    "mainzone": "main_text",
    "margintextzone": "side_text",
    "dropcapitalzone": "filler",
    "graphiczone": "filler",
    "stampzone": "filler",
    "sealzone": "filler",
    "titlepagezone": "filler",
    # NumberingZone is resolved by position (see catmus_class)
    # Dropped: damagezone, digitizationartefactzone, musiczone, all *Line classes
}


def _norm(cat: str) -> str:
    return str(cat).split(":")[0].replace(" ", "").replace("_", "").lower()


def catmus_class(category: str, bbox_xyxy, page_height: float) -> Optional[int]:
    """Return the target class id for a CATMuS zone, or None if it is dropped."""
    c = _norm(category)
    if c == "numberingzone":  # page/folio numbers: header if in the top half, else footer
        cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
        return CLASS_IDS["header" if cy < page_height / 2 else "footer"]
    name = CATMUS_MAP.get(c)
    return CLASS_IDS[name] if name else None
