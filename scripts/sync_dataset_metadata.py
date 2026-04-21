#!/usr/bin/env python3
"""Internal helper to refresh dataset metadata from raw EMD files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quantem.data.sync import sync_dataset_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()
    sync_dataset_metadata(args.dataset_dir)


if __name__ == "__main__":
    main()
