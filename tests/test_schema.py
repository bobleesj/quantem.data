import pytest

from quantem.data.schema import validate, make_template, VALID_TECHNIQUES


def _valid_meta():
    return {
        "schema_version": "1.0",
        "name": "silicon_110",
        "technique": "hrtem",
        "description": "Silicon [110] HRTEM image",
        "data": {"shape": [512, 512], "dtype": "float32"},
        "attribution": {"contributor": "Test", "license": "CC-BY-4.0"},
    }


def test_validate_valid():
    assert validate(_valid_meta()) == []


def test_validate_missing_name():
    meta = _valid_meta()
    del meta["name"]
    errors = validate(meta)
    assert any("name" in e for e in errors)


def test_validate_missing_technique():
    meta = _valid_meta()
    del meta["technique"]
    errors = validate(meta)
    assert any("technique" in e for e in errors)


def test_validate_invalid_technique():
    meta = _valid_meta()
    meta["technique"] = "xray"
    errors = validate(meta)
    assert any("technique" in e.lower() or "xray" in e for e in errors)


def test_validate_missing_data():
    meta = _valid_meta()
    del meta["data"]
    errors = validate(meta)
    assert any("data" in e for e in errors)


def test_validate_missing_data_shape():
    meta = _valid_meta()
    del meta["data"]["shape"]
    errors = validate(meta)
    assert any("shape" in e for e in errors)


def test_validate_missing_attribution():
    meta = _valid_meta()
    del meta["attribution"]
    errors = validate(meta)
    assert any("attribution" in e for e in errors)


def test_validate_missing_contributor():
    meta = _valid_meta()
    del meta["attribution"]["contributor"]
    errors = validate(meta)
    assert any("contributor" in e for e in errors)


def test_validate_missing_license():
    meta = _valid_meta()
    del meta["attribution"]["license"]
    errors = validate(meta)
    assert any("license" in e for e in errors)


def test_make_template():
    t = make_template(
        name="gold_nanoparticle",
        technique="tomo",
        shape=(64, 256, 256),
        description="Au NP tilt series",
        contributor="Jane",
    )
    assert t["name"] == "gold_nanoparticle"
    assert t["technique"] == "tomo"
    assert t["data"]["shape"] == [64, 256, 256]
    assert t["attribution"]["contributor"] == "Jane"
    assert validate(t) == []


def test_all_techniques_valid():
    for tech in VALID_TECHNIQUES:
        meta = _valid_meta()
        meta["technique"] = tech
        assert validate(meta) == [], f"Technique {tech!r} should be valid"
