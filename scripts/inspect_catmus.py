import datasets
from collections import Counter

print("Loading validation stream...")
ds = datasets.load_dataset("CATMuS/medieval-segmentation", split="validation", streaming=True)
rec = next(iter(ds))
print("Columns:", list(rec.keys()))
print("Image type:", type(rec["image"]), "Size:", rec["image"].size, "rec width/height:", rec["width"], rec["height"])
objs = rec["objects"]
print("Objects type:", type(objs))
print("Objects keys:", list(objs.keys()))
print("Sample object entries:")
for i in range(min(10, len(objs["category"]))):
    print(f"  [{i}] cat={objs['category'][i]!r}, type={objs['type'][i]!r}, bbox={objs['bbox'][i]}")

