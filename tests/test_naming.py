"""find_dataset: resolve a flat dataset name to its repo path + kind, with no network.

These rules decide where download/delete act, so they are the highest-value thing to pin:
a wrong resolution means pulling or deleting the wrong dataset. Pure function, plain lists."""
from pathlib import PureWindowsPath

import pytest

from quantem.data import huggingface
from quantem.data.huggingface import _find_sidecar, find_dataset, parse_sidecar


def test_folder_dataset_resolves_to_dir():
    files = ["4dstem/gold_512/gold_master.h5", "4dstem/gold_512/gold_data_000001.h5"]
    assert find_dataset(files, "gold_512") == ("4dstem/gold_512", "dir")


def test_single_file_dataset_resolves_to_file():
    files = ["haadf/gold.tif"]
    assert find_dataset(files, "gold") == ("haadf/gold.tif", "file")


def test_json_sidecar_is_not_a_rival_dataset():
    files = ["haadf/gold.tif", "haadf/gold.json"]  # sidecar must be ignored
    assert find_dataset(files, "gold") == ("haadf/gold.tif", "file")


def test_missing_name_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        find_dataset(["haadf/gold.tif"], "silver")


def test_same_name_in_two_buckets_is_ambiguous():
    files = ["4dstem/gold/gold_master.h5", "haadf/gold.tif"]
    with pytest.raises(ValueError):
        find_dataset(files, "gold")


def test_folder_and_file_with_one_name_are_both_reachable():
    files = ["haadf/gold/data.npy", "haadf/gold.tif"]
    with pytest.raises(ValueError):
        find_dataset(files, "gold")  # the flat name cannot choose between them
    assert find_dataset(files, "haadf/gold") == ("haadf/gold", "dir")  # an exact repo path wins
    assert find_dataset(files, "haadf/gold.tif") == ("haadf/gold.tif", "file")


def test_non_data_bucket_is_not_a_dataset():
    # `notebooks/` holds Colab demos, not datasets - it must never resolve as one
    with pytest.raises(FileNotFoundError):
        find_dataset(["notebooks/demo/show.ipynb"], "demo")


def test_placeholder_is_skipped():
    with pytest.raises(FileNotFoundError):
        find_dataset(["4dstem/placeholder_gold/keep.txt"], "placeholder_gold")


def test_yaml_sidecar_is_not_a_rival_dataset():
    files = ["haadf/gold.tif", "haadf/gold.yaml"]  # the YAML sidecar must be ignored
    assert find_dataset(files, "gold") == ("haadf/gold.tif", "file")


def test_find_sidecar_prefers_yaml_over_legacy_json():
    files = ["haadf/gold.tif", "haadf/gold.json", "haadf/gold.yaml"]
    assert _find_sidecar(files, "gold") == "haadf/gold.yaml"


def test_find_sidecar_still_reads_legacy_json():
    files = ["4dstem/gold/master.h5", "4dstem/gold/meta.json"]  # old dataset, JSON only
    assert _find_sidecar(files, "gold") == "4dstem/gold/meta.json"


def test_find_sidecar_keeps_forward_slashes_on_windows(monkeypatch):
    # repo paths always use "/"; a Windows path object would answer haadf\gold.yaml and match nothing
    monkeypatch.setattr(huggingface, "Path", PureWindowsPath)
    files = ["haadf/gold.tif", "haadf/gold.yaml"]
    assert _find_sidecar(files, "haadf/gold") == "haadf/gold.yaml"


def test_parse_sidecar_reads_yaml_and_json(tmp_path):
    yaml_path = tmp_path / "a.yaml"
    yaml_path.write_text("voltage_kV: 300\nscan_shape: [512, 512]\n")
    json_path = tmp_path / "b.json"
    json_path.write_text('{"voltage_kV": 300, "scan_shape": [512, 512]}')
    expected = {"voltage_kV": 300, "scan_shape": [512, 512]}
    assert parse_sidecar(yaml_path) == expected
    assert parse_sidecar(json_path) == expected
