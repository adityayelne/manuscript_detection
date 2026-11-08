import json
from collections import Counter
from pathlib import Path

for split in ["dev", "train", "test"]:
    p = Path(f"data/raw/catmus/data/{split}/metadata.jsonl")
    if not p.exists():
        print(f"Split {split}: NOT DOWNLOADED YET")
        continue
    cats = Counter()
    types = Counter()
    block_cats = Counter()
    line_cats = Counter()
    total_records = 0
    total_objs = 0
    sample_bbox = None
    sample_w_h = None

    with open(p, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            total_records += 1
            objs = d["objects"]
            w, h = d["width"], d["height"]
            if sample_bbox is None and objs["bbox"]:
                sample_bbox = objs["bbox"][0]
                sample_w_h = (w, h)
            for c, t in zip(objs["category"], objs["type"]):
                cats[c] += 1
                types[t] += 1
                total_objs += 1
                if t == "block":
                    block_cats[c] += 1
                elif t == "line":
                    line_cats[c] += 1

    print(f"\n=== Split {split} ===")
    print(f"Total records: {total_records}, Total objects: {total_objs}")
    print(f"Types: {dict(types)}")
    print(f"Sample bbox: {sample_bbox} on image W,H={sample_w_h}")
    print("Block categories:")
    for k, v in block_cats.most_common():
        print(f"  {k}: {v}")
    print("Line categories:")
    for k, v in line_cats.most_common():
        print(f"  {k}: {v}")

