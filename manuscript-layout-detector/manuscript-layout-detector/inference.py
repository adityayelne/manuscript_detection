#!/usr/bin/env python
"""CLI entry point:  python inference.py --input ./data/test_images --output ./results"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mld.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
