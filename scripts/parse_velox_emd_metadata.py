#!/usr/bin/env python3
"""Parse normalized metadata from a Velox/EMD file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quantem.data.metadata import parse_velox_emd_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("emd_files", nargs="+", type=Path)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()

    parsed = [parse_velox_emd_metadata(path) for path in args.emd_files]
    if len(parsed) == 1:
        print(json.dumps(parsed[0], indent=args.indent))
    else:
        print(json.dumps(parsed, indent=args.indent))


if __name__ == "__main__":
    main()
