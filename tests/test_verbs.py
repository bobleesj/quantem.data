"""Orchestration of the network verbs (download / upload / delete / status), with a fake
Hugging Face client - no network. We are not testing huggingface_hub; we are testing that
OUR code calls it with the right repo paths: download builds the right allow-pattern, upload
writes the data plus its sidecar, delete removes the data AND its .json, status aggregates
sizes only over real datasets. monkeypatch swaps ``_hf()`` for the fake."""
import sys
from fnmatch import fnmatch
from pathlib import Path
from types import SimpleNamespace

import pytest

from quantem.data import cli
from quantem.data import huggingface as hg


class _Entry:
    """A repo-tree node as status() reads it: a path and a byte size."""
    def __init__(self, path, size):
        self.path = path
        self.size = size


class _FakeApi:
    def __init__(self, hub):
        self._hub = hub
    def whoami(self, token=None):
        return {"name": "tester"}
    def list_repo_tree(self, repo_id, repo_type="dataset", recursive=True, revision=None):
        self._hub.calls.append(("list_repo_tree", revision))
        return self._hub._revision_trees.get(revision, self._hub._tree)


class _FakeHub:
    """Records every call OUR code makes, returns canned values, touches no network."""
    def __init__(
        self, files=(), tree=(), cache="/no/such/cache", root=None, sidecar="{}",
        revisions=None, revision_trees=None,
    ):
        self._files = list(files)
        self._revisions = revisions or {}  # revision -> the files that revision holds, when they differ
        self._tree = list(tree)
        self._revision_trees = revision_trees or {}  # the same, for the sized tree status() reads
        self.calls = []
        self.root = root
        self.sidecar = sidecar
        self.constants = SimpleNamespace(HF_HUB_CACHE=cache)
        self.errors = SimpleNamespace(HfHubHTTPError=Exception, EntryNotFoundError=Exception)

    def list_repo_files(self, repo_id, repo_type="dataset", revision=None):
        self.calls.append(("list_repo_files", revision))
        return self._revisions.get(revision, self._files)
    def snapshot_download(
        self, repo_id, repo_type="dataset", allow_patterns=None, local_dir=None,
        revision=None, cache_dir=None, force_download=False,
    ):
        self.calls.append(("snapshot_revision", revision))
        self.calls.append(("snapshot_download", allow_patterns, local_dir))
        self.calls.append(("cache", cache_dir, force_download))
        patterns = [allow_patterns] if isinstance(allow_patterns, str) else allow_patterns
        root = Path(local_dir or cache_dir or self.root)
        for filename in self._files:
            if any(fnmatch(filename, pattern) for pattern in patterns):
                path = root / filename
                if force_download or not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"input")
                    self.calls.append(("transfer", filename))
        return root
    def hf_hub_download(self, repo_id=None, repo_type=None, filename=None, revision=None, cache_dir=None):
        self.calls.append(("fetch", filename, revision, cache_dir))
        path = Path(cache_dir or self.root) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.sidecar)
        return path
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
        return _FakeApi(self)
    def get_token(self):
        return "tok"


def _install(monkeypatch, hub):
    monkeypatch.setattr(hg, "_hf", lambda: hub)
    return hub


def _uploaded(hub):
    return [path for verb, path in ((c[0], c[1]) for c in hub.calls) if verb == "upload_file"]


# --- download -------------------------------------------------------------------

def test_download_folder_dataset_globs_the_directory(monkeypatch, tmp_path):
    hub = _install(monkeypatch, _FakeHub(
        files=["4dstem/gold/master.h5", "4dstem/gold/data.h5"], root=tmp_path,
    ))
    path = hg.download("gold", verbose=False)
    assert ("snapshot_download", "4dstem/gold/*", None) in hub.calls
    assert path == tmp_path / "4dstem/gold"


def test_download_single_file_fetches_just_that_file(monkeypatch, tmp_path):
    hub = _install(monkeypatch, _FakeHub(files=["haadf/gold.tif"], root=tmp_path))
    path = hg.download("gold", verbose=False)
    assert ("snapshot_download", "haadf/gold.tif", None) in hub.calls
    assert path == tmp_path / "haadf/gold.tif"


# --- upload ---------------------------------------------------------------------

def test_upload_file_with_meta_writes_data_and_sidecar(monkeypatch, tmp_path):
    hub = _install(monkeypatch, _FakeHub())
    src = tmp_path / "gold.tif"
    src.write_bytes(b"x")
    url = hg.upload(str(src), name="gold", folder="haadf", meta={"voltage_kV": 300})
    assert _uploaded(hub) == ["haadf/gold.tif", "haadf/gold.yaml"]  # data, then its YAML sidecar
    assert url == "https://hf/commit"


def test_upload_file_without_meta_writes_only_data(monkeypatch, tmp_path):
    hub = _install(monkeypatch, _FakeHub())
    src = tmp_path / "gold.tif"
    src.write_bytes(b"x")
    hg.upload(str(src), name="gold", folder="haadf")
    assert _uploaded(hub) == ["haadf/gold.tif"]  # no sidecar when there is no metadata


@pytest.mark.parametrize("filename, sidecar", [
    # a dot inside the name is not an extension: this one was stored as "...01.7Mx gold overview 01.emd"
    ("0021 - HAADF 3.7Mx gold overview 01.emd", "0021 - HAADF 3.7Mx gold overview 01.yaml"),
    ("gold overview 01.emd", "gold overview 01.yaml"),
    ("gold_overview", "gold_overview.yaml"),  # no dot at all
])
def test_upload_without_a_name_keeps_the_file_name(monkeypatch, tmp_path, filename, sidecar):
    hub = _install(monkeypatch, _FakeHub())
    src = tmp_path / filename
    src.write_bytes(b"x")
    hg.upload(src, meta={"voltage_kV": 300})
    assert _uploaded(hub) == [f"haadf/{filename}", f"haadf/{sidecar}"]


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
    """Match rc41's image-only I/O surface; removed acquisition verbs are absent."""
    fake_io = SimpleNamespace(read_image=lambda path: ("image", path))
    monkeypatch.setitem(sys.modules, "quantem.widget", SimpleNamespace(io=fake_io))
    monkeypatch.setitem(sys.modules, "quantem.widget.io", fake_io)


def test_load_single_image_goes_through_read_image(monkeypatch, tmp_path):
    img = tmp_path / "gold.tif"
    img.write_bytes(b"x")
    monkeypatch.setattr(hg, "download", lambda name, **options: img)
    monkeypatch.setattr(hg, "read_meta", lambda name, **options: None)
    _fake_widget(monkeypatch)
    assert hg.load("gold") == ("image", img)


def test_load_folder_backed_npy_image_goes_through_read_image(monkeypatch, tmp_path):
    folder = tmp_path / "gold_haadf_npy"
    folder.mkdir()
    img = folder / "data.npy"
    img.write_bytes(b"x")
    (folder / "meta.yaml").write_text("pixel_size_nm: 0.018\n")
    monkeypatch.setattr(hg, "download", lambda name, **options: folder)
    monkeypatch.setattr(hg, "read_meta", lambda name, **options: None)
    _fake_widget(monkeypatch)
    assert hg.load("gold_haadf_npy") == ("image", img)


def test_load_image_applies_sidecar_calibration(monkeypatch, tmp_path):
    img = tmp_path / "gold.tif"
    img.write_bytes(b"x")
    dataset = SimpleNamespace(name="raw", sampling=[1, 1], units=["pixels", "pixels"], metadata={})
    fake_io = SimpleNamespace(read_image=lambda path: dataset)
    monkeypatch.setitem(sys.modules, "quantem.widget", SimpleNamespace(io=fake_io))
    monkeypatch.setitem(sys.modules, "quantem.widget.io", fake_io)
    monkeypatch.setattr(hg, "download", lambda name, **options: img)
    monkeypatch.setattr(
        hg,
        "read_meta",
        lambda name, **options: {"name": "Gold", "sampling": [0.018, 0.018], "units": ["nm", "nm"]},
    )
    assert hg.load("gold") is dataset
    assert dataset.name == "Gold"
    assert dataset.sampling == [0.018, 0.018]
    assert dataset.units == ["nm", "nm"]
    assert dataset.metadata["name"] == "Gold"


def test_load_image_raises_when_the_dataset_rejects_a_sidecar_value(monkeypatch, tmp_path):
    class Image:
        """Stands in for the core image dataset, whose sampling setter validates."""
        def __init__(self):
            self.name, self.metadata, self._sampling = "raw", {}, [1.0, 1.0]
        @property
        def sampling(self):
            return self._sampling
        @sampling.setter
        def sampling(self, value):
            if not isinstance(value, list):
                raise TypeError(f"sampling must be a list, got {value!r}")
            self._sampling = value

    _install(monkeypatch, _FakeHub(
        files=["haadf/gold.tif", "haadf/gold.yaml"], root=tmp_path, sidecar="sampling: 0.018 nm\n",
    ))
    fake_io = SimpleNamespace(read_image=lambda path: Image())
    monkeypatch.setitem(sys.modules, "quantem.widget", SimpleNamespace(io=fake_io))
    monkeypatch.setitem(sys.modules, "quantem.widget.io", fake_io)
    with pytest.raises(TypeError, match="sampling must be a list, got '0.018 nm'"):
        hg.load("haadf/gold", verbose=False)  # used to return the image with sampling [1.0, 1.0]


@pytest.mark.parametrize("payload", ["masters", "data.npy", "scan.h5"])
def test_load_native_4dstem_uses_gpu_io_without_widget(monkeypatch, tmp_path, payload):
    folder = tmp_path / "4dstem" / "gold_512"
    folder.mkdir(parents=True)
    path = folder
    if payload == "data.npy":
        (folder / payload).write_bytes(b"npy")
    elif payload == "scan.h5":
        path = folder.parent / payload
        path.write_bytes(b"hdf5")
    monkeypatch.setattr(hg, "download", lambda *a, **kw: path)
    monkeypatch.setitem(sys.modules, "quantem.widget", None)
    calls = []
    fake_io = SimpleNamespace(
        discover=lambda path, **kw: calls.append(("discover", path, kw)) or ["master.h5"],
        load=lambda source, **kw: calls.append(("load", source, kw)) or "resident",
    )
    monkeypatch.setitem(sys.modules, "quantem.gpu", SimpleNamespace(io=fake_io))
    options = {"backend": "cuda", "representation": "encoded", "hot_pixel_correction": "none"}
    assert hg.load("4dstem/gold_512", revision="input-commit", verbose=False, **options) == "resident"
    expected_source = ["master.h5"] if payload == "masters" else folder / payload if payload == "data.npy" else path
    assert calls[-1] == ("load", expected_source, options)
    assert len(calls) == (2 if payload == "masters" else 1)


def test_load_native_missing_gpu_explains_download_alternative(monkeypatch, tmp_path):
    folder = tmp_path / "4dstem" / "gold"
    folder.mkdir(parents=True)
    monkeypatch.setattr(hg, "download", lambda *a, **kw: folder)
    monkeypatch.setitem(sys.modules, "quantem.gpu", None)
    with pytest.raises(ImportError, match="pip install quantem.gpu.*download"):
        hg.load("4dstem/gold")


@pytest.mark.parametrize("name, files, sidecars_read", [
    ("haadf/gold", ["haadf/gold.tif", "haadf/gold.yaml"], ["haadf/gold.yaml"]),
    ("4dstem/gold_512", ["4dstem/gold_512/data.npy", "4dstem/gold_512/meta.yaml"], []),
])
def test_load_keeps_download_options_and_refuses_unknown_keywords(
    monkeypatch, tmp_path, name, files, sidecars_read,
):
    hub = _install(monkeypatch, _FakeHub(files=files, root=tmp_path / "default"))
    _fake_widget(monkeypatch)
    # the real 4D-STEM loader has a closed signature too: no cache_dir, no force_download
    gpu_io = SimpleNamespace(load=lambda source, *, backend="auto": ("resident", source))
    monkeypatch.setitem(sys.modules, "quantem.gpu", SimpleNamespace(io=gpu_io))
    chosen = tmp_path / "chosen"
    _, source = hg.load(
        name, revision="input-commit", cache_dir=chosen, force_download=True, verbose=False,
    )
    assert source.is_relative_to(chosen)
    assert not (tmp_path / "default").exists()
    assert ("cache", chosen, True) in hub.calls
    assert [call[1:] for call in hub.calls if call[0] == "fetch"] == [
        (path, "input-commit", chosen) for path in sidecars_read
    ]
    with pytest.raises(TypeError, match="not_an_argument"):
        hg.load(name, not_an_argument=123, verbose=False)


# --- status ---------------------------------------------------------------------

def test_status_aggregates_only_real_datasets(monkeypatch):
    tree = [
        _Entry("4dstem/gold/master.h5", 1_000_000),
        _Entry("4dstem/gold/data.h5", 3_000_000),
        _Entry("haadf/silver.tif", 2_000_000),
        _Entry("notebooks/demo.ipynb", 9_000_000),     # not a dataset bucket -> ignored
        _Entry("4dstem/placeholder_x/keep.txt", 5),    # placeholder -> ignored
    ]
    _install(monkeypatch, _FakeHub(tree=tree))
    snap = hg.status()
    by_name = {d["name"]: d for d in snap["datasets"]}
    assert set(by_name) == {"4dstem/gold", "haadf/silver"}
    assert by_name["4dstem/gold"]["files"] == 2
    assert by_name["4dstem/gold"]["size_mb"] == pytest.approx(4.0)
    assert snap["logged_in_as"] == "tester"


def test_datasets_returns_human_friendly_load_table(monkeypatch):
    tree = [
        _Entry("4dstem/gold/master.h5", 1_000_000),
        _Entry("haadf/silver.tif", 2_000_000),
    ]
    _install(monkeypatch, _FakeHub(tree=tree))
    catalog = hg.datasets(modality="haadf")

    assert list(catalog) == [{"name": "silver", "modality": "haadf", "size_mb": 2.0}]
    text = repr(catalog)
    assert "qdata.load('silver')" in text
    assert "haadf/silver" not in text


def test_every_listing_reads_the_revision_it_is_given(monkeypatch, tmp_path, capsys):
    current = [_Entry("haadf/gold.tif", 3_000_000), _Entry("widget/ducky/show1d_monitor.jsonl", 200)]
    older = [_Entry("haadf/silver.tif", 2_000_000), _Entry("4dstem/gold/master.h5", 1_000_000)]
    hub = _install(monkeypatch, _FakeHub(
        root=tmp_path, tree=current, revision_trees={"older-layout": older},
    ))
    snap = hg.status(revision="older-layout")
    assert [item["name"] for item in snap["datasets"]] == ["4dstem/gold", "haadf/silver"]
    catalog = hg.datasets(revision="older-layout")
    assert [row["name"] for row in catalog] == ["gold", "silver"]
    assert "qdata.load('silver', revision='older-layout')" in repr(catalog)
    assert "qdata.load(&quot;silver&quot;, revision=&quot;older-layout&quot;)" in catalog._repr_html_()
    gallery = hg.browse(revision="older-layout")
    cards = gallery._repr_html_()
    assert 'download("haadf/silver", revision="older-layout")' in cards
    assert ("fetch", ".thumbnails/haadf/silver.png", "older-layout", None) in hub.calls
    hg.tree(revision="older-layout")
    listings = [capsys.readouterr().out, repr(gallery), cards]
    for command in ("tree", "status"):
        assert cli.main([command, "--revision", "older-layout"]) == 0
        listings.append(capsys.readouterr().out)
    for listing in listings:
        assert "silver" in listing and "ducky" not in listing
    assert {call[1] for call in hub.calls if call[0] == "list_repo_tree"} == {"older-layout"}
    assert "qdata.load('gold')" in repr(hg.datasets())  # no revision: default branch, bare call
    assert hub.calls[-1] == ("list_repo_tree", None)


def test_tutorial_download_and_metadata_use_the_same_revision(monkeypatch, tmp_path):
    prefix = "denoise/simulated_au_nanobeam_pg_dose1"
    hub = _install(monkeypatch, _FakeHub(root=tmp_path / "cache", files=[
        f"{prefix}/truth_master.h5", f"{prefix}/meta.json",
    ]))
    metadata = tmp_path / "meta.json"
    metadata.write_text('{"signal_units": "events"}')

    def fetch_metadata(**kwargs):
        hub.calls.append(("metadata", kwargs["filename"], kwargs["revision"]))
        return metadata

    hub.hf_hub_download = fetch_metadata
    revision = "5925e4b49b2ca28ff82d6d5a6b9a8bfea9edfb20"
    path = hg.download(prefix, revision=revision, verbose=False)
    assert path == tmp_path / "cache" / prefix
    assert hg.read_meta(prefix, revision=revision) == {"signal_units": "events"}
    assert ("list_repo_files", revision) in hub.calls
    assert ("snapshot_revision", revision) in hub.calls
    assert ("metadata", f"{prefix}/meta.json", revision) in hub.calls


def test_metadata_of_an_unlisted_dataset_raises_instead_of_reporting_none(monkeypatch, tmp_path, capsys):
    _install(monkeypatch, _FakeHub(
        root=tmp_path, files=["haadf/gold.tif", "widget/ducky/show1d_monitor.jsonl"],
        revisions={"older-layout": ["haadf/gold.tif"]},
    ))
    assert hg.read_meta("haadf/gold") is None  # listed, no sidecar: the only case that answers None
    with pytest.raises(FileNotFoundError, match="'no_such_dataset' not found"):
        hg.read_meta("no_such_dataset")
    assert hg.read_meta("widget/ducky") is None
    with pytest.raises(FileNotFoundError, match="'widget/ducky' not found"):
        hg.read_meta("widget/ducky", revision="older-layout")
    assert cli.main(["meta", "widget/ducky", "--revision", "older-layout"]) == 1
    printed = capsys.readouterr()
    assert "'widget/ducky' not found" in printed.err
    assert "no metadata" not in printed.out


def test_discovery_includes_workflows_without_listing_code(monkeypatch):
    tree = [
        _Entry("denoise/gold/truth_master.h5", 1000),
        _Entry("denoise/gold/meta.json", 100),
        _Entry("widget/session/frame.emd", 2000),
        _Entry("widget/ducky/show1d_monitor.jsonl", 200),
        _Entry("widget/ducky/metadata.json", 100),
        _Entry("widget/notes/metadata.json", 100),
        _Entry("survey/session/frame.emd", 2000),  # retired catalog bucket
        _Entry("denoise/README.md", 500),
        _Entry("denoise/tutorial/run.ipynb", 500),
        _Entry("denoise/notes/meta.json", 500),
        _Entry("haadf/gold.tif", 3000),
    ]
    _install(monkeypatch, _FakeHub(tree=tree, files=[entry.path for entry in tree]))
    assert hg.list_datasets() == ["denoise/gold", "haadf/gold", "widget/ducky", "widget/session"]
    catalog = hg.datasets()
    rows = {row["name"]: row for row in catalog}
    assert set(rows) == {"denoise/gold", "haadf/gold", "session", "ducky"}
    assert rows["denoise/gold"]["size_mb"] == pytest.approx(0.0011)
    assert "qdata.download('session')" in repr(catalog)
    assert "qdata.load('session')" not in repr(catalog)
    assert hg.find_dataset([entry.path for entry in tree], "widget/session") == (
        "widget/session", "dir",
    )
    assert hg._find_sidecar([entry.path for entry in tree], "ducky") == "widget/ducky/metadata.json"
    assert "qdata.download('ducky')" in repr(catalog)


def test_qualified_selection_keeps_the_right_metadata_and_delete_target(monkeypatch):
    files = [
        "4dstem/gold/master.h5", "4dstem/gold/meta.json",
        "haadf/gold.tif", "haadf/gold.yaml",
    ]
    hub = _install(monkeypatch, _FakeHub(files=files))
    assert hg.find_dataset(files, "4dstem/gold") == ("4dstem/gold", "dir")
    assert hg.find_dataset(files, "haadf/gold.tif") == ("haadf/gold.tif", "file")
    assert hg._find_sidecar(files, "haadf/gold") == "haadf/gold.yaml"
    with pytest.raises(ValueError, match="qualified dataset"):
        hg._find_sidecar(files, "gold")
    assert hg.delete("haadf/gold") == ["haadf/gold.tif", "haadf/gold.yaml"]
    assert not any("4dstem" in call[1] for call in hub.calls if call[0].startswith("delete"))


def test_pinned_widget_collection_reuses_cache_and_keeps_all_companions(monkeypatch, tmp_path):
    files = [
        "widget/ducky/show1d_monitor.jsonl", "widget/ducky/metadata.json",
        "widget/ducky/snapshots/frame.npy", "widget/ducky/snapshots/lambda_0.npy",
        "widget/other/array.npy", "haadf/ducky.npy",
    ]
    hub = _install(monkeypatch, _FakeHub(files=files, root=tmp_path / "cache"))
    options = dict(revision="pinned-commit", cache_dir=tmp_path / "cache", verbose=False)
    folder = hg.download("widget/ducky", **options)
    assert sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()) == [
        "metadata.json", "show1d_monitor.jsonl", "snapshots/frame.npy", "snapshots/lambda_0.npy",
    ]
    first_transfers = [call for call in hub.calls if call[0] == "transfer"]
    assert len(first_transfers) == 4
    assert hg.download("widget/ducky", **options) == folder
    assert [call for call in hub.calls if call[0] == "transfer"] == first_transfers
    hg.download("widget/ducky", force_download=True, **options)
    assert len([call for call in hub.calls if call[0] == "transfer"]) == 8
    assert all(call[1] == "pinned-commit" for call in hub.calls if call[0] in (
        "list_repo_files", "snapshot_revision",
    ))
    assert ("cache", tmp_path / "cache", True) in hub.calls
    assert not (tmp_path / "cache" / "widget/other").exists()
    with pytest.raises(ValueError, match="widget input collection"):
        hg.load("widget/ducky", revision="pinned-commit", verbose=False)


def test_single_image_export_includes_calibration_and_missing_shard_fails(monkeypatch, tmp_path):
    hub = _install(monkeypatch, _FakeHub(files=[
        "haadf/gold.tif", "haadf/gold.yaml", "haadf/silver.tif",
    ]))
    image = hg.download("haadf/gold", out=tmp_path, revision="old-input", verbose=False)
    assert image == tmp_path / "haadf/gold.tif"
    assert image.with_suffix(".yaml").is_file()
    assert not (tmp_path / "haadf/silver.tif").exists()
    hub._files = ["4dstem/gold/master.h5", "4dstem/gold/chunk.h5"]
    monkeypatch.setattr(hub, "snapshot_download", lambda **kwargs: tmp_path)
    with pytest.raises(FileNotFoundError, match="chunk.h5.*force_download=True"):
        hg.download("4dstem/gold", revision="old-input", verbose=False)
