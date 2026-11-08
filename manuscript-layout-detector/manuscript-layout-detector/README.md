# Manuscript Layout Region Detection

Detects and classifies five layout regions in historical manuscript images (palm-leaf and paper, any script):
`header`, `footer`, `main_text`, `side_text` (marginalia), `filler` (decoration, English text, pencil text).
Outputs bounding boxes, labels, confidence scores, annotated images and JSON. Batch CLI included.

## Status (read this first)

| Part | State |
|---|---|
| CLI, JSON/annotated output, clipping, batch handling, tests | Done and tested (7 tests pass) |
| **Heuristic fallback detector** (no training) | Done; checked visually on the two assignment samples |
| Dataset download / build / train / evaluate scripts | Written; **not run end to end**, because the authoring environment could not reach Hugging Face or Zenodo and had no GPU |
| Trained YOLO weights (`weights/best.pt`) | **Not included.** Train with the Colab notebook (~1 h on a T4) |

Until `weights/best.pt` exists, `inference.py` automatically uses the heuristic fallback and says so in the log and in
`results/summary.json`. The fallback is a baseline, not a substitute for the trained model.
The CATMuS record parsing in `scripts/build_dataset.py` was written from the dataset card's schema description and has
not been run against the real files, so expect to fix a column name or two on first run.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python inference.py --input ./data/samples --output ./results
```

Output tree:
```
results/
  annotated/<name>.png     boxes + label + score drawn on a copy of the page
  json/<name>.json         predictions for that page
  summary.json             batch totals, region counts, failures, detector used
```

Per-image JSON:
```json
{
  "image": "sample_paper.png",
  "image_size": {"width": 2047, "height": 774},
  "model": {"name": "heuristic-v1", "backend": "heuristic", "weights": null, "version": "1.0.0"},
  "predictions": [
    {"label": "filler", "class_id": 4, "confidence": 0.55,
     "bbox_xyxy": [589, 23, 1456, 51], "bbox_xywh_norm": [0.499, 0.048, 0.424, 0.036]}
  ],
  "inference": {"conf": 0.25, "iou": 0.5, "imgsz": 1280, "preprocess": "auto", "time_ms": 28.0}
}
```
Coordinates are integer pixels in the original image, always inside `[0, W-1] x [0, H-1]`.

CLI options: `--weights`, `--backend {auto,yolo,heuristic}`, `--conf`, `--iou`, `--imgsz`, `--device`,
`--preprocess {none,clahe,auto}`, `--class-conf header=0.3,filler=0.2`, `--no-json`, `--no-annotated`.
Input may be one image or a folder (recursive). Unreadable files are skipped and listed in `summary.json`.
Input files are opened read-only; preprocessing runs on an in-memory copy; the output folder must differ from the input folder.

## Train the real model

1. Open `notebooks/train_colab.ipynb` in Colab (GPU runtime), upload this repo as a zip, run all cells. Or locally:
```bash
python scripts/download_datasets.py --only catmus        # CATMuS, CC-BY 4.0, ~4.9 GB
python scripts/build_dataset.py --catmus-dir data/raw/catmus
python train.py --model yolo11s.pt --epochs 50 --imgsz 1280 --batch 8
python scripts/evaluate.py                                # writes docs/RESULTS.md
```
2. `train.py` copies the best checkpoint to `weights/best.pt`; `inference.py` picks it up automatically.
3. Strongly recommended: label 50-100 Indic pages (palm-leaf + paper) with `docs/LABELING_GUIDE.md`, put them in
   `data/raw/indic_custom/{images,labels}` (YOLO txt, 5 classes) and rebuild with `--custom-dir`. They are repeated 3x in training.

## Datasets and class mapping

Full links, licences and caveats: `docs/DATASETS.md`.

| Source | Target class |
|---|---|
| CATMuS RunningTitleZone | header |
| CATMuS NumberingZone | header if box centre is in the top half of the page, else footer |
| CATMuS QuireMarksZone | footer |
| CATMuS MainZone | main_text |
| CATMuS MarginTextZone | side_text |
| CATMuS DropCapitalZone, GraphicZone, StampZone, SealZone, TitlePageZone | filler |
| CATMuS DamageZone, DigitizationArtefactZone, MusicZone, all line classes | dropped |
| HORAE-LSv2 decoration / initials / rubrication / miniatures (optional) | filler |

No public dataset annotates English captions or pencil text, so these are **synthesised** during dataset build
(`src/mld/data/augment.py`):
- **English caption strips** added above/below the page, labelled `filler`. The two assignment samples carry such
  captions at the very top and bottom, which a model trained only on medieval data would call header/footer.
- **Pencil-like scribbles** in margins, labelled `filler`.
- **Hard negatives** (binding hole, ruled lines, dust) left unlabelled so they are not mistaken for text.
- **Palm-leaf look**: solid-colour background padding and wide 2:1-3:1 crops.

## Model rationale

YOLO11 (Ultralytics) bounding-box detector: the task needs boxes with class and confidence, regions are few and large,
and YOLO trains quickly from COCO weights and runs on CPU for inference. `imgsz=1280` keeps small caption/header regions visible
on wide pages. Horizontal flips are disabled because scripts are directional. Alternatives considered: oriented boxes
(useful for skewed pages, but the deliverable asks for axis-aligned boxes) and pixel segmentation (more labelling cost).

## Robustness

Implemented: CLAHE for low-contrast pages (`--preprocess auto`), illumination flattening helper (`src/mld/preprocess.py`),
HSV and small-rotation augmentation in training, hard-negative overlays, box clipping, per-file error isolation in batch mode.
Not implemented: explicit deskewing, bleed-through removal, damaged-edge modelling. Blur, stains and heavy skew will lower accuracy.

## Heuristic fallback (how it works, where it fails)

Ink mask (darker than local background) -> remove long ruled lines and filled blobs (holes, stains) -> detect thin, wide,
centred text lines hugging the top/bottom edge (captions) as `filler` -> merge ink into blocks -> largest block is `main_text`,
blocks beside it `side_text`, above/below it `header`/`footer`. Scores are fixed rule confidences, not probabilities.
It will fail on multi-column pages, pages with large illustrations, and any page whose top/bottom text is real header/footer
text rather than a caption (it treats thin centred edge lines as `filler`; disable with `caption_as_filler=False` in code).

## Limitations

- Training data is Latin-script medieval manuscripts; Indic palm-leaf pages are out of domain until you add labelled pages.
- Pencil text and English text are synthetic in training.
- HORAE and the other optional sets contain no header/footer/margin classes.
- Image licences differ by source (see `docs/DATASETS.md`); CC BY-NC sets are research/assignment use only.

## Layout

```
inference.py  train.py
src/mld/{cli,config,io,preprocess,postprocess,visualize}.py  src/mld/models/{detector,heuristics}.py
src/mld/data/{mapping,augment,synthetic}.py
scripts/{download_datasets,build_dataset,evaluate,smoke_train}.py
notebooks/train_colab.ipynb  docs/  tests/  data/samples/
```

Run tests: `python -m pytest -q`.
