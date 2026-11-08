# Datasets

| Dataset | Use here | Access | Licence | Link |
|---|---|---|---|---|
| CATMuS Medieval Segmentation (1,705 images; train 1,336 / dev 191 / test 178) | Primary training data | Open | CC-BY 4.0 | https://huggingface.co/datasets/CATMuS/medieval-segmentation |
| HORAE-LSv2 (555 images, 22,964 annotations) | Optional extra `filler` | Open | Annotations CC BY 4.0; images keep institutional licences | https://zenodo.org/records/16919911 |
| Medieval Manuscript Layout (1,800 pages, pixel masks) | Optional extra `filler` (needs mask-to-box step, not implemented) | Open | CC BY-NC 4.0 | https://zenodo.org/records/20282994 |
| YOLO_historical / Tridis (about 2,510 images, YOLO OBB) | Optional, not used by the scripts | Open | MIT on the card; source licences vary | https://huggingface.co/datasets/magistermilitum/YOLO_historical |
| Indiscapes (508 Indic palm-leaf/paper images) | Best in-domain data; **not automated** | Google Form request | MIT on the GitHub repo | https://ihdia.iiit.ac.in/indiscapes/ , https://github.com/ihdia/indiscapes , form: https://forms.gle/yjPyug3zqVumacL37 |
| Cam datasets (CamMarg, CamStd, GW5056) | Not used: annotation download not found | - | not stated | https://www.mdpi.com/1999-4893/16/3/136 |

HORAE-LSv2: read the dataset README (downloaded with the data) for its class-id table, then pass the ids of decoration,
initials, rubrication and miniatures as `--horae-filler-ids` to `scripts/build_dataset.py`.

Indiscapes classes (Page Boundary, Hole, Decorator, Picture, Library Marker, Physical Degradation, Character Line Segment,
Boundary Line) have no header/footer/side-text equivalents. If obtained, use Decorator/Picture as `filler` and Hole/Boundary Line
as unlabelled hard negatives.

No dataset found annotates English captions or pencil text; see README for the synthetic substitutes.
