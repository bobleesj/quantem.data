"""find_dataset: resolve a flat dataset name to its repo path + kind, with no network.

These rules decide where download/delete act, so they are the highest-value thing to pin:
a wrong resolution means pulling or deleting the wrong dataset. Pure function, plain lists."""
import pytest

from quantem.data.huggingface import find_dataset


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
