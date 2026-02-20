import json

import numpy as np
import pytest

from quantem.data import available, info, load, list_files


def test_available_returns_list():
    result = available()
    assert isinstance(result, list)
    assert len(result) > 0


def test_available_sorted():
    result = available()
    assert result == sorted(result)


def test_available_contains_korean_sample():
    result = available()
    assert "korean_sample_c1" in result


def test_info_returns_dict():
    meta = info("korean_sample_c1")
    assert isinstance(meta, dict)
    assert meta["name"] == "korean_sample_c1"
    assert meta["technique"] == "image"


def test_info_has_required_fields():
    meta = info("korean_sample_c1")
    assert "schema_version" in meta
    assert "description" in meta
    assert "data" in meta
    assert "shape" in meta["data"]


def test_info_unknown_raises():
    with pytest.raises(KeyError, match="nonexistent"):
        info("nonexistent")


def test_load_returns_array():
    data = load("korean_sample_c1")
    assert isinstance(data, np.ndarray)
    assert data.shape == (256, 256)
    assert data.dtype == np.float32


def test_load_with_metadata():
    data, meta = load("korean_sample_c1", metadata=True)
    assert isinstance(data, np.ndarray)
    assert isinstance(meta, dict)
    assert data.shape == (256, 256)
    assert meta["technique"] == "image"


def test_load_unknown_raises():
    with pytest.raises(KeyError, match="nonexistent"):
        load("nonexistent")


def test_list_files_returns_list():
    result = list_files()
    assert isinstance(result, list)
    assert len(result) > 0


def test_list_files_has_npy_and_json():
    result = list_files()
    paths = [f["path"] for f in result]
    assert any(p.endswith(".npy") for p in paths)
    assert any(p.endswith(".json") for p in paths)


def test_list_files_type_field():
    result = list_files()
    for f in result:
        assert f["type"] in ("data", "metadata")
        if f["path"].endswith(".json"):
            assert f["type"] == "metadata"
        elif f["path"].endswith(".npy"):
            assert f["type"] == "data"


def test_list_files_filter_by_technique():
    result = list_files("image")
    assert len(result) > 0
    for f in result:
        assert f["path"].startswith("image/")


def test_list_files_filter_empty():
    result = list_files("nonexistent_technique")
    assert result == []


def test_list_files_sorted():
    result = list_files()
    paths = [f["path"] for f in result]
    assert paths == sorted(paths)
