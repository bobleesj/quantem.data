"""Orchestration of the network verbs (download / upload / delete / status), with a fake
Hugging Face client - no network. We are not testing huggingface_hub; we are testing that
OUR code calls it with the right repo paths: download builds the right allow-pattern, upload
writes the data plus its sidecar, delete removes the data AND its .json, status aggregates
sizes only over real datasets. monkeypatch swaps ``_hf()`` for the fake."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from quantem.data import huggingface as hg


class _Entry:
    """A repo-tree node as status() reads it: a path and a byte size."""
    def __init__(self, path, size):
        self.path = path
        self.size = size


class _FakeApi:
    def __init__(self, tree):
        self._tree = tree
    def whoami(self, token=None):
        return {"name": "tester"}
    def list_repo_tree(self, repo_id, repo_type="dataset", recursive=True):
        return self._tree


class _FakeHub:
    """Records every call OUR code makes, returns canned values, touches no network."""
    def __init__(self, files=(), tree=(), cache="/no/such/cache"):
        self._files = list(files)
        self._tree = list(tree)
        self.calls = []
        self.constants = SimpleNamespace(HF_HUB_CACHE=cache)
        self.errors = SimpleNamespace(HfHubHTTPError=Exception, EntryNotFoundError=Exception)

    def list_repo_files(self, repo_id, repo_type="dataset"):
        return self._files
    def snapshot_download(self, repo_id, repo_type="dataset", allow_patterns=None, local_dir=None):
        self.calls.append(("snapshot_download", allow_patterns, local_dir))
        return "/fake/root"
    def upload_file(self, path_or_fileobj=None, path_in_repo=None, repo_id=None, repo_type=None):
        self.calls.append(("upload_file", path_in_repo))
        return "https://hf/commit"
    def upload_folder(self, folder_path=None, path_in_repo=None, repo_id=None, repo_type=None):
        self.calls.append(("upload_folder", path_in_repo))
        return "https://hf/commit"
    def delete_file(self, path_in_repo=None, repo_id=None, repo_type=None):
        self.calls.append(("delete_file", path_in_repo))
    def delete_folder(self, path_in_repo=None, repo_id=None, repo_type=None):
        self.calls.append(("delete_folder", path_in_repo))
    def HfApi(self):
        return _FakeApi(self._tree)
    def get_token(self):
        return "tok"


def _install(monkeypatch, hub):
    monkeypatch.setattr(hg, "_hf", lambda: hub)
    return hub


def _uploaded(hub):
    return [path for verb, path in ((c[0], c[1]) for c in hub.calls) if verb == "upload_file"]


# --- download -------------------------------------------------------------------

def test_download_folder_dataset_globs_the_directory(monkeypatch):
    hub = _install(monkeypatch, _FakeHub(files=["4dstem/gold/master.h5", "4dstem/gold/data.h5"]))
    path = hg.download("gold", verbose=False)
    assert ("snapshot_download", "4dstem/gold/*", None) in hub.calls
    assert path == Path("/fake/root/4dstem/gold")


def test_download_single_file_fetches_just_that_file(monkeypatch):
    hub = _install(monkeypatch, _FakeHub(files=["haadf/gold.tif"]))
    path = hg.download("gold", verbose=False)
    assert ("snapshot_download", "haadf/gold.tif", None) in hub.calls
    assert path == Path("/fake/root/haadf/gold.tif")


# --- upload ---------------------------------------------------------------------

def test_upload_file_with_meta_writes_data_and_sidecar(monkeypatch, tmp_path):
    hub = _install(monkeypatch, _FakeHub())
    src = tmp_path / "gold.tif"
    src.write_bytes(b"x")
    url = hg.upload(str(src), name="gold", folder="haadf", meta={"voltage_kV": 300})
    assert _uploaded(hub) == ["haadf/gold.tif", "haadf/gold.json"]  # data, then its sidecar
    assert url == "https://hf/commit"


def test_upload_file_without_meta_writes_only_data(monkeypatch, tmp_path):
    hub = _install(monkeypatch, _FakeHub())
    src = tmp_path / "gold.tif"
    src.write_bytes(b"x")
    hg.upload(str(src), name="gold", folder="haadf")
    assert _uploaded(hub) == ["haadf/gold.tif"]  # no sidecar when there is no metadata


# --- delete ---------------------------------------------------------------------

def test_delete_folder_dataset_removes_the_folder(monkeypatch):
    hub = _install(monkeypatch, _FakeHub(files=["4dstem/gold/master.h5"]))
    removed = hg.delete("gold")
    assert ("delete_folder", "4dstem/gold") in hub.calls
    assert removed == ["4dstem/gold/"]


def test_delete_file_dataset_removes_data_and_sidecar(monkeypatch):
    hub = _install(monkeypatch, _FakeHub(files=["haadf/gold.tif", "haadf/gold.json"]))
    removed = hg.delete("gold")
    assert ("delete_file", "haadf/gold.tif") in hub.calls
    assert ("delete_file", "haadf/gold.json") in hub.calls
    assert set(removed) == {"haadf/gold.tif", "haadf/gold.json"}


# --- load (download + dispatch to the widget loader) ----------------------------

def _fake_widget(monkeypatch):
    """Stand in for quantem.widget.io so load()'s dispatch is testable without the renderer
    installed: read_image / load / discover_masters just echo their inputs so we can see
    which path load() took."""
    fake_io = SimpleNamespace(
        read_image=lambda path: ("image", path),
        load=lambda masters, **kw: ("4d", masters),
        discover_masters=lambda folder: ["m1", "m2"],
    )
    monkeypatch.setitem(sys.modules, "quantem.widget", SimpleNamespace(io=fake_io))
    monkeypatch.setitem(sys.modules, "quantem.widget.io", fake_io)


def test_load_single_image_goes_through_read_image(monkeypatch, tmp_path):
    img = tmp_path / "gold.tif"
    img.write_bytes(b"x")
    monkeypatch.setattr(hg, "download", lambda name, repo=None, out=None: img)
    _fake_widget(monkeypatch)
    assert hg.load("gold") == ("image", img)


def test_load_acquisition_folder_goes_through_4d_loader(monkeypatch, tmp_path):
    folder = tmp_path / "gold_512"
    folder.mkdir()
    monkeypatch.setattr(hg, "download", lambda name, repo=None, out=None: folder)
    _fake_widget(monkeypatch)
    assert hg.load("gold_512") == ("4d", ["m1", "m2"])  # dir -> load(discover_masters(...))


# --- status ---------------------------------------------------------------------

def test_status_aggregates_only_real_datasets(monkeypatch):
    tree = [
        _Entry("4dstem/gold/master.h5", 1_000_000),
        _Entry("4dstem/gold/data.h5", 3_000_000),
        _Entry("haadf/silver.tif", 2_000_000),
        _Entry("notebooks/demo.ipynb", 9_000_000),     # not a dataset bucket -> ignored
        _Entry("4dstem/placeholder_x/keep.txt", 5),    # placeholder -> ignored
    ]
    hub = _install(monkeypatch, _FakeHub(tree=tree))
    snap = hg.status()
    by_name = {d["name"]: d for d in snap["datasets"]}
    assert set(by_name) == {"4dstem/gold", "haadf/silver"}
    assert by_name["4dstem/gold"]["files"] == 2
    assert by_name["4dstem/gold"]["size_mb"] == pytest.approx(4.0)
    assert snap["logged_in_as"] == "tester"
