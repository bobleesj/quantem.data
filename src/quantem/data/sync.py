"""Helpers for synchronizing dataset metadata from raw files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quantem.data.metadata import parse_velox_emd_metadata


def sync_dataset_metadata(dataset_dir: str | Path) -> dict[str, Any]:
    """Refresh raw metadata sidecars and raw file entries for a dataset folder."""
    dataset_dir = Path(dataset_dir)
    dataset_path = dataset_dir / "dataset.json"

    dataset = json.loads(dataset_path.read_text()) if dataset_path.exists() else {}
    dataset.setdefault("files", [])

    raw_dir = dataset_dir / "raw"
    raw_metadata_dir = dataset_dir / "raw_metadata"
    raw_metadata_dir.mkdir(exist_ok=True)

    parsed_entries = []
    for path in sorted(raw_dir.rglob("*.emd")):
        parsed = parse_velox_emd_metadata(path)
        sidecar = raw_metadata_dir / f"{path.name}.json"
        sidecar.write_text(json.dumps(parsed, indent=2) + "\n")

        rel_path = str(path.relative_to(dataset_dir)).replace("\\", "/")
        parsed_entries.append(
            {
                "path": rel_path,
                "kind": "raw",
                "format": "emd",
                "signal": "haadf",
                "family": "stem",
                "shape": [
                    parsed["scan_size"]["height"],
                    parsed["scan_size"]["width"],
                ],
                "beam_energy_kv": parsed["beam_energy_kv"],
                "convergence_semiangle_mrad": parsed["convergence_semiangle_mrad"],
                "stem_magnification_x": parsed["stem_magnification_x"],
                "full_scan_field_of_view_nm": parsed["full_scan_field_of_view_nm"],
            }
        )

    other_files = [entry for entry in dataset["files"] if entry.get("kind") != "raw"]
    dataset["files"] = parsed_entries + other_files
    dataset["metadata_source"] = "velox_emd"

    dataset_path.write_text(json.dumps(dataset, indent=2) + "\n")
    return dataset
