"""Repository helpers for quantem-data."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    env = os.environ.get("QUANTEM_DATA_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3]


def datasets_dir() -> Path:
    return _repo_root() / "datasets"


def list_datasets() -> list[str]:
    root = datasets_dir()
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def load_dataset_metadata(dataset_id: str) -> dict[str, Any]:
    path = datasets_dir() / dataset_id / "dataset.json"
    return json.loads(path.read_text())
