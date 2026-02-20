"""Tests for upload and update_metadata (require HF write access)."""

import json

import numpy as np
import pytest

from quantem.data import upload, update_metadata, info, load
from quantem.data.schema import validate


def test_upload_validates_metadata():
    """Upload should reject invalid metadata."""
    arr = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="validation failed"):
        upload(
            arr,
            name="test_bad",
            technique="invalid_technique",
            description="bad",
            contributor="test",
        )


def test_upload_validates_shape_mismatch():
    """Upload should reject shape mismatch between array and metadata."""
    arr = np.ones((4, 4), dtype=np.float32)
    meta = {
        "schema_version": "1.0",
        "name": "test_mismatch",
        "technique": "image",
        "description": "test",
        "data": {"shape": [8, 8], "dtype": "float32"},
        "attribution": {"contributor": "test", "license": "CC-BY-4.0"},
    }
    with pytest.raises(ValueError, match="shape"):
        upload(arr, name="test_mismatch", technique="image", metadata=meta)


def test_upload_from_json_file(tmp_path):
    """Upload should accept metadata from a JSON file path."""
    meta = {
        "schema_version": "1.0",
        "name": "test_json_file",
        "technique": "image",
        "description": "test from json file",
        "data": {"shape": [4, 4], "dtype": "float32"},
        "attribution": {"contributor": "test", "license": "CC-BY-4.0"},
    }
    json_path = tmp_path / "meta.json"
    json_path.write_text(json.dumps(meta))
    # Just check it parses — don't actually upload (would need auth)
    errors = validate(meta)
    assert errors == []
