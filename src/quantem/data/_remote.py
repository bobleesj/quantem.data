"""Share raw 4D-STEM / HAADF datasets through one Hugging Face dataset repo.

Collaborators with no instrument data need a one-line way to pull a reference
acquisition; the data owner needs a one-line way to publish one. Both sides
install ``quantem.data`` once and never touch ``huggingface_hub`` directly.

A single dataset repo (default ``bobleesj/quantem-data``, override with the
``QUANTEM_DATA_REPO`` env var or a ``repo=`` argument) is the storage backend.
Keep it simple: two top-level buckets, ``4dstem/`` for acquisitions and
``haadf/`` for images, each holding one folder/file per dataset (``4dstem/gold_512/``,
``haadf/gold_haadf.tif``). An Arina acquisition keeps its master + ``_data_*.h5``
chunk siblings together inside its folder, so ``download`` returns a directory
``discover_masters`` can read. ``download`` takes a flat name and finds its bucket.

The verbs are plain English for a microscopist audience: ``upload``, ``download``,
``list_datasets``, ``delete``, ``status``. ``huggingface_hub`` is imported lazily
so importing ``quantem.data`` on a CUDA-less laptop stays cheap; it is a regular
install dependency so a single ``pip install`` gives both the loader and this path.
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

DEFAULT_REPO = "bobleesj/quantem-data"

# The only buckets that hold shareable DATASETS. Other top-level folders in the repo
# (e.g. `notebooks/` for Colab demos) are not data and must not show up in list / status
# / tree / browse as if they were downloadable acquisitions.
DATA_BUCKETS = ("4dstem", "haadf")


def _resolve_repo(repo: str | None) -> str:
    """Pick the dataset repo: explicit arg, else env, else the project default."""
    return repo or os.environ.get("QUANTEM_DATA_REPO") or DEFAULT_REPO


def _hf():
    """Import huggingface_hub lazily with a clear install hint when missing."""
    # Our datasets are PUBLIC - no token needed. Silence huggingface_hub's
    # "HF_TOKEN secret does not exist" nudge (it fires on every download in Colab
    # and confuses users into thinking auth is required).
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    warnings.filterwarnings("ignore", message=r"(?s).*HF_TOKEN.*")
    # The "set a HF_TOKEN for higher rate limits" nudge on public downloads comes through
    # the logging system, not warnings.warn, so the filter above misses it. Mute the huggingface_hub
    # logger to ERROR so a normal public download is quiet (real errors still surface).
    import logging  # noqa: PLC0415
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    try:
        import huggingface_hub  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to share datasets. "
            "Install it with `pip install huggingface_hub`."
        ) from exc
    return huggingface_hub


def upload(path: str | Path, name: str | None = None, *,
           folder: str | None = None, repo: str | None = None,
           meta: dict | None = None) -> str:
    """Upload a file or folder to the shared repo under ``<folder>/<name>``.

    ``folder`` is the top-level bucket; it defaults to ``haadf`` for a single
    file and ``4dstem`` for an acquisition folder, so HAADF images and 4D-STEM
    land in the right place with no thought. A folder uploads its whole contents
    (Arina master + chunk siblings stay together) so ``download`` returns a loadable
    dir. Returns the commit URL. Needs a write token on this machine
    (``hf auth login`` or ``HF_TOKEN``); publishing is an explicit owner action.

    ``meta`` carries the calibration the raw Arina master can NOT store itself
    (scan sampling, FOV, voltage, semiangle): the detector h5 only knows detector
    pixels, so a collaborator who downloads the data would otherwise have no FOV.
    When given, it is merged with auto-derived ``det_shape``/``scan_shape`` (read
    from the master) and written as a ``meta.json`` sidecar travelling with the
    dataset; ``read_meta`` returns it on the other side (it also reads the older
    ``quantem_meta.json`` name so existing datasets keep working).
    """
    hf = _hf()
    src = Path(path)
    repo_id = _resolve_repo(repo)
    if name is None:
        name = src.stem if src.is_file() else src.name
    if folder is None:
        folder = "haadf" if src.is_file() else "4dstem"
    if src.is_dir():
        info = hf.upload_folder(
            folder_path=str(src), path_in_repo=f"{folder}/{name}",
            repo_id=repo_id, repo_type="dataset",
        )
    else:
        suffix = "".join(src.suffixes)  # keep multi-part extensions like .nii.gz
        info = hf.upload_file(
            path_or_fileobj=str(src), path_in_repo=f"{folder}/{name}{suffix}",
            repo_id=repo_id, repo_type="dataset",
        )
    sidecar = _build_meta(src, meta)
    if sidecar:
        _upload_meta(hf, repo_id, folder, name, sidecar, is_dir=src.is_dir())
    return getattr(info, "commit_url", info)  # CommitInfo in modern huggingface_hub, str in old


def _derive_4dstem_shapes(folder: Path) -> dict:
    """Read det_shape (+ square scan_shape) from an Arina master, best-effort.

    The detector h5 stores its own pixel count even though it knows no scan FOV;
    surfacing det_shape/scan_shape in the sidecar saves the collaborator from
    re-deriving them. Returns ``{}`` if no master or the read fails - never blocks
    the upload over a metadata convenience.
    """
    try:
        import h5py  # noqa: PLC0415
        import math
        masters = sorted(folder.glob("*_master.h5"))
        if not masters:
            return {}
        with h5py.File(masters[0], "r") as f:
            spec = f["entry/instrument/detector/detectorSpecific"]
            out: dict = {"det_shape": [int(spec["y_pixels_in_detector"][()]),
                                       int(spec["x_pixels_in_detector"][()])]}
            data = f.get("entry/data/data")
            if data is not None and data.ndim >= 1:
                n = int(data.shape[0])
                side = math.isqrt(n)
                if side * side == n:  # square scan - the common case
                    out["scan_shape"] = [side, side]
            return out
    except (OSError, KeyError, ValueError, ImportError):
        return {}


def _build_meta(src: Path, meta: dict | None) -> dict:
    """Merge auto-derived shapes (4D-STEM folder) under explicit operator meta."""
    out: dict = {}
    if src.is_dir():
        out.update(_derive_4dstem_shapes(src))
    if meta:
        out.update({k: v for k, v in meta.items() if v is not None})
    return out


def _upload_meta(hf, repo_id: str, folder: str, name: str,
                 sidecar: dict, *, is_dir: bool) -> None:
    """Write the calibration sidecar next to the dataset.

    Folder dataset -> ``<bucket>/<name>/meta.json`` inside the folder (download
    returns the dir, so it rides along). File dataset -> a sibling
    ``<bucket>/<name>.json`` (the same stem ``delete`` already removes).
    """
    import json  # noqa: PLC0415
    import tempfile
    path_in_repo = (f"{folder}/{name}/meta.json" if is_dir
                    else f"{folder}/{name}.json")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(sidecar, fh, indent=2)
        tmp = fh.name
    try:
        hf.upload_file(path_or_fileobj=tmp, path_in_repo=path_in_repo,
                        repo_id=repo_id, repo_type="dataset")
    finally:
        os.unlink(tmp)


def read_meta(name: str, *, repo: str | None = None) -> dict | None:
    """Return a dataset's calibration sidecar, or ``None`` if it has none.

    The counterpart to ``upload(..., meta=...)``: a collaborator who downloads a
    4D-STEM acquisition gets back the scan sampling / FOV / voltage / semiangle the
    raw detector master cannot carry. Public repos need no token.
    """
    import json  # noqa: PLC0415
    hf = _hf()
    repo_id = _resolve_repo(repo)
    target = None
    for f in hf.list_repo_files(repo_id=repo_id, repo_type="dataset"):
        parts = f.split("/")
        # `quantem_meta.json` is the current sidecar name; `meta.json` is a legacy
        # name some early uploads used - accept both so older datasets stay readable.
        if len(parts) == 3 and parts[1] == name and parts[2] in ("quantem_meta.json", "meta.json"):
            target = f
            break
        if len(parts) == 2 and f.endswith(".json") and Path(parts[1]).stem == name:
            target = f
            break
    if target is None:
        return None
    local = hf.hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=target)
    return json.loads(Path(local).read_text())


def download(name: str, *, repo: str | None = None, out: str | Path | None = None,
             verbose: bool = True) -> Path:
    """Download one shared dataset by flat name and return its local path.

    The collaborator names just the dataset (``"gold_512"``); this searches
    every bucket to find where it lives, so they never need to know it is under
    ``4dstem/`` or ``haadf/``. Returns a directory for a multi-file acquisition
    (ready for ``discover_masters`` / ``load``) or the file path for a single-file
    dataset. Public repos need no token.

    ``verbose`` (default) frames Hugging Face's own per-file byte progress bars
    with a clear "downloading from the internet" header and a size/speed summary,
    so the user can tell the wait is the network, not our code. A second call on
    the same dataset is a local cache hit and prints "(already downloaded)".
    """
    hf = _hf()
    repo_id = _resolve_repo(repo)
    files = hf.list_repo_files(repo_id=repo_id, repo_type="dataset")
    candidates: dict[str, str] = {}  # target_rel -> "dir" | "file"
    for f in files:
        parts = f.split("/")
        if len(parts) >= 3 and parts[1] == name:
            candidates[f"{parts[0]}/{name}"] = "dir"
        elif len(parts) == 2 and Path(parts[1]).stem == name and not f.endswith(".json"):
            candidates[f] = "file"  # .json is a sidecar of the data file, not a rival dataset
    if not candidates:
        raise FileNotFoundError(
            f"{name!r} not found in {repo_id}. Call quantem.data.list_datasets() "
            "(or `quantem-data list`) to see available names.")
    if len(candidates) > 1:
        raise ValueError(
            f"{name!r} is ambiguous in {repo_id}: {sorted(candidates)}. "
            "Rename one, or set --repo to a repo where it is unique."
        )
    target_rel, kind = next(iter(candidates.items()))
    pattern = f"{target_rel}/*" if kind == "dir" else target_rel
    if verbose:
        print(f"Downloading '{name}' from Hugging Face ({repo_id}) over the internet - "
              "speed depends on your connection, not your computer ...", flush=True)
    t0 = time.perf_counter()
    root = hf.snapshot_download(
        repo_id=repo_id, repo_type="dataset",
        allow_patterns=pattern,
        local_dir=str(out) if out is not None else None,
    )
    result = Path(root) / target_rel
    if verbose:
        dt = time.perf_counter() - t0
        gb = (sum(f.stat().st_size for f in result.rglob("*") if f.is_file())
              if result.is_dir() else result.stat().st_size) / 1e9
        if dt < 1.0:
            print(f"'{name}' ({gb:.2f} GB) is already cached on disk - no re-download.\n"
                  f"  cached at: {result}", flush=True)
        else:
            print(f"Downloaded '{name}' ({gb:.2f} GB) in {dt:.0f}s "
                  f"({gb * 1000 / dt:.0f} MB/s from Hugging Face).\n"
                  f"  cached on disk - future loads are instant, no re-download.\n"
                  f"  cached at: {result}", flush=True)
    return result


def list_datasets(*, repo: str | None = None) -> list[str]:
    """List shared datasets as ``<bucket>/<name>`` (skips placeholders/docs)."""
    hf = _hf()
    repo_id = _resolve_repo(repo)
    names = set()
    for f in hf.list_repo_files(repo_id=repo_id, repo_type="dataset"):
        parts = f.split("/")
        if len(parts) < 2 or parts[0] not in DATA_BUCKETS or parts[1].startswith("placeholder_"):
            continue
        if len(parts) >= 3:
            names.add(f"{parts[0]}/{parts[1]}")
        elif not parts[1].endswith(".json"):
            names.add(f"{parts[0]}/{Path(parts[1]).stem}")
    return sorted(names)


def delete(name: str, *, repo: str | None = None) -> list[str]:
    """Delete a shared dataset by flat name; returns the repo paths removed.

    A folder dataset removes the whole folder; a file dataset removes the data
    file and its ``.json`` sidecar (same stem). Refuses to act when the name
    matches more than one dataset so a delete never nukes the wrong bucket. The
    CLI adds a re-type-to-confirm prompt; this function deletes immediately.
    """
    hf = _hf()
    repo_id = _resolve_repo(repo)
    dir_locs: set[str] = set()
    file_groups: dict[str, list[str]] = {}
    for f in hf.list_repo_files(repo_id=repo_id, repo_type="dataset"):
        parts = f.split("/")
        if len(parts) >= 3 and parts[1] == name:
            dir_locs.add(f"{parts[0]}/{name}")
        elif len(parts) == 2 and Path(parts[1]).stem == name:
            file_groups.setdefault(parts[0], []).append(f)
    locations = list(dir_locs) + [f"{b}/{name}" for b in file_groups]
    if not locations:
        raise FileNotFoundError(
            f"{name!r} not found in {repo_id}. Call quantem.data.list_datasets() "
            "(or `quantem-data list`) to see available names.")
    if len(locations) > 1:
        raise ValueError(f"{name!r} is ambiguous in {repo_id}: {sorted(locations)}. Delete one explicitly.")
    deleted = []
    if dir_locs:
        loc = next(iter(dir_locs))
        hf.delete_folder(path_in_repo=loc, repo_id=repo_id, repo_type="dataset")
        deleted.append(f"{loc}/")
    else:
        for f in next(iter(file_groups.values())):  # data file + its .json sidecar
            hf.delete_file(path_in_repo=f, repo_id=repo_id, repo_type="dataset")
            deleted.append(f)
    return deleted


def status(*, repo: str | None = None) -> dict:
    """Snapshot of the shared repo: auth, datasets + sizes, and local cache size.

    Answers the operator's "where does my data live, can I upload, what is shared,
    what do I already have locally" in one call. No token needed for the dataset
    listing; auth is reported as whoever is logged in (or ``None`` = download-only).
    """
    hf = _hf()
    repo_id = _resolve_repo(repo)
    api = hf.HfApi()
    token = hf.get_token()
    user = None
    if token:
        try:
            user = api.whoami(token=token).get("name")
        except hf.errors.HfHubHTTPError:
            user = None  # stale/invalid token -> treat as download-only
    sizes: dict[str, int] = {}
    counts: dict[str, int] = {}
    for entry in api.list_repo_tree(repo_id, repo_type="dataset", recursive=True):
        size = getattr(entry, "size", None)
        if size is None:
            continue  # folder entry, not a file
        parts = entry.path.split("/")
        if len(parts) < 2 or parts[0] not in DATA_BUCKETS or parts[1].startswith("placeholder_"):
            continue
        if len(parts) >= 3:
            key = f"{parts[0]}/{parts[1]}"
        elif entry.path.endswith(".json"):
            continue  # top-level sidecar, folded into its data file's dataset
        else:
            key = f"{parts[0]}/{Path(parts[1]).stem}"
        sizes[key] = sizes.get(key, 0) + size
        counts[key] = counts.get(key, 0) + 1
    datasets = [{"name": k, "files": counts[k], "size_mb": sizes[k] / 1e6} for k in sorted(sizes)]
    cache_dir = Path(hf.constants.HF_HUB_CACHE) / f"datasets--{repo_id.replace('/', '--')}"
    cached_mb = 0.0
    if cache_dir.exists():
        cached_mb = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()) / 1e6
    return {
        "repo": repo_id,
        "logged_in_as": user,
        "datasets": datasets,
        "total_mb": sum(sizes.values()) / 1e6,
        "cache_dir": str(cache_dir),
        "cached_mb": cached_mb,
    }


# --- discovery: structure + visual browse (look before a multi-GB download) ---

def _by_bucket(repo: str | None = None) -> tuple[dict, dict]:
    """``status()`` datasets grouped by bucket (top-level data type). Returns (groups, snapshot)."""
    snap = status(repo=repo)
    groups: dict[str, list[dict]] = {}
    for d in snap["datasets"]:
        groups.setdefault(d["name"].split("/", 1)[0], []).append(d)
    return groups, snap


def tree(*, repo: str | None = None) -> None:
    """Print the shared repo grouped by bucket (data type) with sizes: the folder structure
    at a glance, so a user can pick a dataset without opening Hugging Face in a browser."""
    groups, snap = _by_bucket(repo=repo)
    print(f"{snap['repo']}  ({snap['total_mb'] / 1000:.1f} GB total, {len(snap['datasets'])} datasets)")
    for bucket in sorted(groups):
        items = sorted(groups[bucket], key=lambda d: d["name"])
        gb = sum(d["size_mb"] for d in items) / 1000
        print(f"\n{bucket}/   ({len(items)} datasets, {gb:.2f} GB)")
        for d in items:
            print(f"  {d['size_mb']:>9.1f} MB  {d['name'].split('/', 1)[1]}")


THUMB_DIR = ".thumbnails"  # top-level tree, OUTSIDE the data buckets, so a thumbnail
# never looks like a dataset folder (a `<bucket>/<name>/thumbnail.png` made single-file
# datasets ambiguous with their data file in `download`).


def _thumb_data_uri(full_name: str, repo_id: str) -> str | None:
    """Fetch only ``.thumbnails/<bucket>/<name>.png`` (KB, not the data) and return it as a
    base64 data URI for inline display, or None if the dataset has no thumbnail yet."""
    import base64  # noqa: PLC0415
    hf = _hf()
    try:
        path = hf.hf_hub_download(repo_id=repo_id, repo_type="dataset",
                                   filename=f"{THUMB_DIR}/{full_name}.png")
    except Exception:
        return None
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return f"data:image/png;base64,{b64}"


def _card_html(d: dict, repo_id: str) -> str:
    """One dataset card: thumbnail (or placeholder) + flat name + bucket + size."""
    full = d["name"]
    bucket, short = full.split("/", 1)
    uri = _thumb_data_uri(full, repo_id)
    img = (f'<img src="{uri}" style="width:160px;height:160px;object-fit:cover;border-radius:6px">'
           if uri else
           '<div style="width:160px;height:160px;border-radius:6px;background:#eee;display:flex;'
           'align-items:center;justify-content:center;color:#999;font-size:11px">no preview yet</div>')
    return (f'<div style="display:inline-block;margin:6px;text-align:center;font-family:sans-serif">'
            f'{img}<div style="font-weight:600;font-size:13px;margin-top:4px">{short}</div>'
            f'<div style="font-size:11px;color:#666">{bucket} · {d["size_mb"]/1000:.2f} GB</div>'
            f'<div style="font-size:10px;color:#999">download("{short}")</div></div>')


class _Gallery:
    """Renders a thumbnail grid inline in Jupyter (``_repr_html_``); prints the tree in a
    terminal (``__repr__``). Returned by ``browse()`` so the notebook displays it directly."""

    def __init__(self, repo: str | None):
        self._groups, self._snap = _by_bucket(repo=repo)
        self._repo_id = _resolve_repo(repo)

    def _repr_html_(self) -> str:
        parts = [f'<div style="font-family:sans-serif"><b>{self._snap["repo"]}</b> '
                 f'— {self._snap["total_mb"]/1000:.1f} GB, {len(self._snap["datasets"])} datasets']
        for bucket in sorted(self._groups):
            items = sorted(self._groups[bucket], key=lambda d: d["name"])
            parts.append(f'<h4 style="margin:12px 0 0">{bucket}/ '
                         f'<span style="font-weight:400;color:#888">({len(items)})</span></h4>')
            parts.append("".join(_card_html(d, self._repo_id) for d in items))
        parts.append("</div>")
        return "".join(parts)

    def __repr__(self) -> str:
        tree(repo=self._repo_id if self._repo_id != DEFAULT_REPO else None)
        return ""


def browse(*, repo: str | None = None) -> "_Gallery":
    """Visual dataset picker: a grid of thumbnails + names + sizes you can scan before
    downloading anything. In Jupyter it renders inline; in a terminal it prints the tree.
    Thumbnails are tiny (KB) and fetched on the fly - the multi-GB data is never touched.
    Pick a name and pass it to ``download(...)``."""
    return _Gallery(repo)


def load(name: str, *, repo: str | None = None, out: str | Path | None = None, **kwargs):
    """Download a dataset AND return it ready to use, in one call.

    The intuitive primitive: ``ds = load("gold_haadf")`` instead of download-a-path then
    hand it to a separate loader. Dispatches by shape - a 4D-STEM acquisition (a folder of
    Arina masters) returns the loaded 4D data; a single image returns a ``Dataset2d``.
    Loading lives in ``quantem.widget`` (the renderer), imported lazily so ``quantem.data``
    stays a standalone, widget-free transfer layer; extra kwargs pass through to the 4D
    loader (e.g. ``det_bin=``)."""
    path = download(name, repo=repo, out=out)
    try:
        from quantem.widget import io as _io  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "load() needs the quantem.widget loader: pip install quantem.widget "
            "(download() alone returns the file path without it)."
        ) from exc
    if Path(path).is_dir():
        return _io.load(_io.discover_masters(path), **kwargs)  # 4D-STEM acquisition
    return _io.read_image(path)  # single image -> Dataset2d
