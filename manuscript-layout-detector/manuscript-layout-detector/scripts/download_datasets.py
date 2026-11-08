#!/usr/bin/env python
"""Download the open datasets used for training (run on a machine with internet access).

  python scripts/download_datasets.py            # CATMuS + HORAE-LSv2
  python scripts/download_datasets.py --only catmus
  python scripts/download_datasets.py --with-medieval-layout   # optional 1 GB pixel-mask set (CC BY-NC)

Idempotent: existing files with a matching MD5 are skipped. Writes data/raw/MANIFEST.json.
Indiscapes cannot be automated: it needs a Google Form access request (see docs/DATASETS.md).
"""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

HORAE = "https://zenodo.org/records/16919911/files/{}?download=1"
HORAE_FILES = {
    "images.zip": "9a6b73e3282040b45a12c17b922f9404",
    "labels.zip": "0ce2ed685f87c249c77ce2966f839900",
    "metadata.zip": "a4de7cc226b8aeda32f35b297499dde5",
    "README.md": "066c951402b61dd5027e63c15d73179a",
}
MEDLAYOUT = ("https://zenodo.org/records/20282994/files/Dataset.zip?download=1", "965ded7012f1be5da0d4e5641ccc2273")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, expect_md5: str) -> None:
    import requests
    from tqdm import tqdm
    if dest.exists() and md5(dest) == expect_md5:
        print(f"[skip] {dest.name} already downloaded and verified")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    got = md5(dest)
    if got != expect_md5:
        raise RuntimeError(f"MD5 mismatch for {dest.name}: {got} != {expect_md5}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["catmus", "horae"], default=None)
    ap.add_argument("--with-medieval-layout", action="store_true")
    a = ap.parse_args()
    manifest = {}

    if a.only in (None, "catmus"):
        from huggingface_hub import snapshot_download
        d = RAW / "catmus"
        snapshot_download(repo_id="CATMuS/medieval-segmentation", repo_type="dataset", local_dir=str(d))
        manifest["catmus"] = {"source": "https://huggingface.co/datasets/CATMuS/medieval-segmentation",
                              "license": "CC-BY-4.0", "path": str(d.relative_to(ROOT))}
        print("[ok] CATMuS ->", d)

    if a.only in (None, "horae"):
        d = RAW / "horae_lsv2"
        for name, m in HORAE_FILES.items():
            fetch(HORAE.format(name), d / name, m)
        for z in ("images.zip", "labels.zip"):
            tgt = d / z.replace(".zip", "")
            if not tgt.exists():
                with zipfile.ZipFile(d / z) as zf:
                    zf.extractall(tgt)
        manifest["horae_lsv2"] = {"source": "https://zenodo.org/records/16919911",
                                  "license": "annotations CC BY 4.0; images keep institutional licences",
                                  "path": str(d.relative_to(ROOT)), "md5": HORAE_FILES}
        print("[ok] HORAE-LSv2 ->", d, "(read its README.md for the class-id table)")

    if a.with_medieval_layout:
        d = RAW / "medieval_layout"
        fetch(MEDLAYOUT[0], d / "Dataset.zip", MEDLAYOUT[1])
        manifest["medieval_layout"] = {"source": "https://zenodo.org/records/20282994", "license": "CC BY-NC 4.0",
                                       "path": str(d.relative_to(ROOT))}

    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print("manifest written to", RAW / "MANIFEST.json")


if __name__ == "__main__":
    sys.exit(main())
