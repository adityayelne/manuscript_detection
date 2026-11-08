import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset
from mld.config import CLASS_NAMES, CLASS_IDS
from mld.data.mapping import catmus_class
from scripts.build_dataset import _objects

ds = load_dataset("data/raw/catmus")
print("Dataset splits:", ds.keys())
rec = ds["validation"][0]
print("Validation[0] keys:", list(rec.keys()))
print("Image:", rec["image"], "mode:", rec["image"].mode, "size:", rec["image"].size)
print("Objects type:", type(rec["objects"]))
objs = list(_objects(rec, None))
print(f"Total objects yielded: {len(objs)}")
for cat, typ, bb in objs[:5]:
    c = catmus_class(cat, bb, rec["image"].size[1])
    print(f"  cat={cat!r}, type={typ!r}, bbox={bb} -> mapped class: {c} ({CLASS_NAMES[c] if c is not None else 'DROPPED'})")
