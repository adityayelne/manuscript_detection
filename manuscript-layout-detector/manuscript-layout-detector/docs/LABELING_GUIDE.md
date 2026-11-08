# Labelling guide for your own Indic pages (YOLO format)

Class ids: 0 header, 1 footer, 2 main_text, 3 side_text, 4 filler.
Files: `data/raw/indic_custom/images/<name>.jpg` and `labels/<name>.txt`, one line per box: `class cx cy w h` (normalised 0-1).
Tools: Label Studio or Roboflow with YOLO export, or `labelImg` in YOLO mode.

Rules
1. One box per text **block**, not per line. The 8 lines of body text on a leaf are one `main_text` box.
2. Printed English caption strips (e.g. "CC-0. In Public Domain... eGangotri") at the top/bottom edge: `filler`, never header/footer.
3. Pencil notes and English notes written on the page: `filler`. Decorations, borders with ornament, seals, drop capitals: `filler`.
4. Folio numbers, running titles above the main block: `header`. Catchwords, signature marks, page numbers below: `footer`.
5. Marginal notes beside the main block: `side_text`.
6. Do not label the binding hole, plain ruled border lines, stains or the background.
7. Label at least 50 pages, mixing palm-leaf and paper, plus a few with real margins/headers so those classes are not empty.
