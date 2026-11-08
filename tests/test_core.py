import hashlib
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mld.cli import main  # noqa: E402
from mld.config import CLASS_IDS, CLASS_NAMES  # noqa: E402
from mld.data import augment as A  # noqa: E402
from mld.data.mapping import catmus_class  # noqa: E402
from mld.data.synthetic import make_page  # noqa: E402
from mld.postprocess import Detection, clip_box, finalize  # noqa: E402

SAMPLES = ROOT / "data" / "samples"


def test_clip_box_inside_page():
    assert clip_box((-10, -5, 5000, 5000), 100, 50) == (0, 0, 99, 49)
    assert clip_box((10, 10, 10, 30), 100, 100) is None          # zero width
    assert clip_box((200, 10, 300, 30), 100, 100) is None        # fully outside


def test_finalize_drops_degenerate_and_clips():
    dets = [Detection(2, 0.9, (-5, -5, 120, 80)), Detection(0, 0.8, (50, 50, 50, 60))]
    out = finalize(dets, 100, 60)
    assert len(out) == 1 and out[0].bbox == (0, 0, 99, 59)


def test_catmus_mapping():
    assert catmus_class("MainZone", (0, 0, 10, 10), 100) == CLASS_IDS["main_text"]
    assert catmus_class("MarginTextZone", (0, 0, 10, 10), 100) == CLASS_IDS["side_text"]
    assert catmus_class("RunningTitleZone", (0, 0, 10, 10), 100) == CLASS_IDS["header"]
    assert catmus_class("QuireMarksZone", (0, 0, 10, 10), 100) == CLASS_IDS["footer"]
    assert catmus_class("DropCapitalZone", (0, 0, 10, 10), 100) == CLASS_IDS["filler"]
    assert catmus_class("GraphicZone", (0, 0, 10, 10), 100) == CLASS_IDS["filler"]
    assert catmus_class("StampZone", (0, 0, 10, 10), 100) == CLASS_IDS["filler"]
    assert catmus_class("SealZone", (0, 0, 10, 10), 100) == CLASS_IDS["filler"]
    assert catmus_class("TitlePageZone", (0, 0, 10, 10), 100) == CLASS_IDS["filler"]
    assert catmus_class("NumberingZone", (0, 5, 10, 15), 100) == CLASS_IDS["header"]
    assert catmus_class("NumberingZone", (0, 85, 10, 95), 100) == CLASS_IDS["footer"]
    assert catmus_class("DamageZone", (0, 0, 10, 10), 100) is None
    assert catmus_class("DigitizationArtefactZone", (0, 0, 10, 10), 100) is None
    assert catmus_class("MusicZone", (0, 0, 10, 10), 100) is None
    assert catmus_class("DefaultLine", (0, 0, 10, 10), 100) is None
    assert catmus_class("HeadingLine", (0, 0, 10, 10), 100) is None


def test_augment_boxes_stay_in_image():
    rng = random.Random(1)
    for _ in range(20):
        img, boxes = make_page(rng)
        for fn in (A.add_caption_strips, A.add_pencil_text, A.add_hard_negatives, A.pad_background, A.wide_crop):
            img2, b2 = fn(img, boxes, rng)
            h, w = img2.shape[:2]
            for c, x1, y1, x2, y2 in b2:
                assert 0 <= c < len(CLASS_NAMES)
                assert -1 <= x1 < x2 <= w + 1 and -1 <= y1 < y2 <= h + 1, (fn.__name__, (x1, y1, x2, y2), (w, h))


def _hash_dir(d: Path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob("*")) if p.is_file()}


def test_cli_heuristic_outputs_and_originals_untouched(tmp_path):
    before = _hash_dir(SAMPLES)
    out = tmp_path / "res"
    assert main(["--input", str(SAMPLES), "--output", str(out), "--backend", "heuristic"]) == 0
    assert _hash_dir(SAMPLES) == before                                  # originals unchanged
    summary = json.loads((out / "summary.json").read_text())
    assert summary["num_images"] == 2 and summary["num_failed"] == 0
    for js in (out / "json").glob("*.json"):
        d = json.loads(js.read_text())
        W, H = d["image_size"]["width"], d["image_size"]["height"]
        assert {"image", "image_size", "model", "predictions", "inference"} <= set(d)
        for p in d["predictions"]:
            assert p["label"] in CLASS_NAMES and 0 <= p["confidence"] <= 1
            x1, y1, x2, y2 = p["bbox_xyxy"]
            assert 0 <= x1 < x2 <= W - 1 and 0 <= y1 < y2 <= H - 1       # inside the page
    assert len(list((out / "annotated").glob("*.png"))) == 2


def test_cli_rejects_same_in_out_and_missing_input(tmp_path):
    assert main(["--input", str(SAMPLES), "--output", str(SAMPLES)]) == 2
    assert main(["--input", str(tmp_path / "nope"), "--output", str(tmp_path / "o")]) == 2


def test_cli_skips_unreadable_file(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    (d / "bad.png").write_bytes(b"not an image")
    cv2.imwrite(str(d / "ok.png"), cv2.imread(str(SAMPLES / "sample_paper.png")))
    assert main(["--input", str(d), "--output", str(tmp_path / "o"), "--backend", "heuristic"]) == 0
    s = json.loads((tmp_path / "o" / "summary.json").read_text())
    assert s["num_failed"] == 1
