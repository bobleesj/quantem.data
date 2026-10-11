"""Contribution provenance checks stay in source, separate from runtime metadata."""

import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER = runpy.run_path(str(ROOT / "scripts/check_dataset_metadata.py"))


def _contribution(folder):
    """A documented, fictional acquisition used only by these tests."""
    meta = json.loads((ROOT / "scripts/templates/dataset-meta.json").read_text())
    meta.update(
        name="test-gold",
        source="fictional test acquisition",
        sample="gold",
        date="2026-10-11",
        permission={
            "received": True,
            "from": "Test owner",
            "collectors": ["Test collector"],
            "date": "2026-10-11",
        },
    )
    path = folder / "meta.json"
    path.write_text(json.dumps(meta))
    return path, meta


def test_documented_contribution_passes_without_rewriting_metadata(tmp_path, capsys):
    path, _ = _contribution(tmp_path)
    original = path.read_bytes()
    assert CHECKER["check_meta"](tmp_path) == []
    assert CHECKER["main"]([str(tmp_path)]) == 0
    assert "ok" in capsys.readouterr().out
    assert path.read_bytes() == original


def test_contributor_gets_permission_and_axis_errors_together(tmp_path, capsys):
    path, meta = _contribution(tmp_path)
    meta["permission"].update(received=False, collectors=[False], date="unknown")
    meta["shape"] = [512, True, 48, 48]
    meta["sampling"] = [0.5, 0.5]
    meta["units"] = ["A", "A"]
    path.write_text(json.dumps(meta))
    original = path.read_bytes()
    assert CHECKER["main"]([str(tmp_path)]) == 1
    report = capsys.readouterr().out
    for message in (
        "permission.received must be true",
        "permission.collectors",
        "permission.date must be YYYY-MM-DD",
        "shape must be a list of ints",
        "sampling length must match shape length",
        "units length must match shape length",
    ):
        assert message in report
    assert path.read_bytes() == original


def test_legacy_calibration_is_read_without_contribution_schema(tmp_path):
    from quantem.data import huggingface

    sidecar = tmp_path / "meta.json"
    sidecar.write_text('{"sampling": [0.018, 0.018], "units": ["nm", "nm"]}')
    assert huggingface.parse_sidecar(sidecar) == {
        "sampling": [0.018, 0.018],
        "units": ["nm", "nm"],
    }
    assert "missing permission" in CHECKER["check_meta"](tmp_path)
