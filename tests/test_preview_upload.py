"""Tests for preview_upload validation."""

import numpy as np

from quantem.data import preview_upload


def test_preview_valid(capsys):
    arr = np.ones((64, 64), dtype=np.float32)
    errors = preview_upload(
        arr,
        name="gold_nanoparticle",
        technique="hrtem",
        description="Gold nanoparticle HRTEM",
        contributor="Test User",
    )
    captured = capsys.readouterr()
    assert errors == []
    assert "Ready to upload" in captured.out
    assert "gold_nanoparticle" in captured.out
    assert "hrtem" in captured.out


def test_preview_bad_name_uppercase(capsys):
    arr = np.ones((4, 4), dtype=np.float32)
    errors = preview_upload(arr, name="BadName", technique="image", contributor="x")
    assert any("lowercase" in e for e in errors)


def test_preview_bad_name_hyphen(capsys):
    arr = np.ones((4, 4), dtype=np.float32)
    errors = preview_upload(arr, name="bad-name", technique="image", contributor="x")
    assert any("underscore" in e.lower() or "hyphen" in e.lower() for e in errors)


def test_preview_bad_name_year(capsys):
    arr = np.ones((4, 4), dtype=np.float32)
    errors = preview_upload(arr, name="sample_2024", technique="image", contributor="x")
    assert any("year" in e.lower() for e in errors)


def test_preview_bad_name_resolution(capsys):
    arr = np.ones((4, 4), dtype=np.float32)
    errors = preview_upload(arr, name="sample_256x256", technique="image", contributor="x")
    assert any("resolution" in e.lower() for e in errors)


def test_preview_invalid_technique(capsys):
    arr = np.ones((4, 4), dtype=np.float32)
    errors = preview_upload(arr, name="test_sample", technique="fake", contributor="x")
    assert any("technique" in e.lower() for e in errors)


def test_preview_shape_mismatch(capsys):
    arr = np.ones((4, 4), dtype=np.float32)
    meta = {
        "schema_version": "1.0",
        "name": "test_sample",
        "technique": "image",
        "description": "test",
        "data": {"shape": [8, 8], "dtype": "float32"},
        "attribution": {"contributor": "x", "license": "CC-BY-4.0"},
    }
    errors = preview_upload(arr, name="test_sample", technique="image", metadata=meta)
    assert any("shape" in e.lower() for e in errors)


def test_preview_existing_name(capsys):
    arr = np.ones((4, 4), dtype=np.float32)
    errors = preview_upload(
        arr, name="korean_sample_c1", technique="image", contributor="x"
    )
    assert any("already exists" in e for e in errors)


def test_preview_prints_summary(capsys):
    arr = np.ones((32, 32), dtype=np.float32)
    preview_upload(
        arr,
        name="test_preview",
        technique="hrtem",
        description="Test description",
        contributor="Jane",
    )
    captured = capsys.readouterr()
    assert "Upload preview" in captured.out
    assert "test_preview" in captured.out
    assert "[32, 32]" in captured.out
    assert "float32" in captured.out
    assert "Jane" in captured.out
