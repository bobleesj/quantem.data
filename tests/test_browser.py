"""Tests for DataBrowser widget."""

import json

from quantem.data import DataBrowser


def test_browser_creates():
    browser = DataBrowser()
    assert isinstance(browser, DataBrowser)


def test_browser_catalog_populated():
    browser = DataBrowser()
    catalog = json.loads(browser.catalog_json)
    assert isinstance(catalog, list)
    assert len(catalog) > 0


def test_browser_catalog_has_korean_sample():
    browser = DataBrowser()
    catalog = json.loads(browser.catalog_json)
    names = [d["name"] for d in catalog]
    assert "korean_sample_c1" in names


def test_browser_catalog_entries_have_fields():
    browser = DataBrowser()
    catalog = json.loads(browser.catalog_json)
    for entry in catalog:
        assert "name" in entry
        assert "technique" in entry
        assert "shape" in entry
        assert "dtype" in entry
        assert "description" in entry
        assert "size_mb" in entry


def test_browser_technique_filter():
    browser = DataBrowser(technique="image")
    assert browser.technique_filter == "image"


def test_browser_data_none_before_load():
    browser = DataBrowser()
    assert browser.data is None
    assert browser.metadata is None


def test_browser_load_dataset():
    browser = DataBrowser()
    browser.selected_name = "korean_sample_c1"
    browser._load_dataset("korean_sample_c1")
    assert browser.data is not None
    assert browser.data.shape == (256, 256)
    assert browser.loaded_name == "korean_sample_c1"
    assert browser.metadata is not None
    assert browser.metadata["technique"] == "image"


def test_browser_selected_info_populated():
    browser = DataBrowser()
    browser.selected_name = "korean_sample_c1"
    assert browser.selected_info_json != ""
    info = json.loads(browser.selected_info_json)
    assert info["name"] == "korean_sample_c1"
    assert "_npy_path" not in info


def test_browser_techniques_property():
    browser = DataBrowser()
    techniques = browser.techniques
    assert isinstance(techniques, list)
    assert "4dstem" in techniques
    assert "image" in techniques


def test_browser_summary(capsys):
    browser = DataBrowser()
    browser.summary()
    captured = capsys.readouterr()
    assert "DataBrowser" in captured.out
    assert "Datasets:" in captured.out


def test_browser_repr():
    browser = DataBrowser()
    r = repr(browser)
    assert "DataBrowser(" in r
    assert "datasets" in r


def test_browser_repr_with_filter():
    browser = DataBrowser(technique="image")
    r = repr(browser)
    assert "filter='image'" in r


def test_browser_repr_with_loaded():
    browser = DataBrowser()
    browser._load_dataset("korean_sample_c1")
    r = repr(browser)
    assert "loaded='korean_sample_c1'" in r
