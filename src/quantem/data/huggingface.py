"""Share raw 4D-STEM / HAADF datasets through one Hugging Face dataset repo.

Collaborators with no instrument data need a one-line way to pull a reference
acquisition; the data owner needs a one-line way to publish one. Both sides
install ``quantem.data`` once and never touch ``huggingface_hub`` directly.

A single dataset repo (default ``bobleesj/quantem-data``, override with the
``QUANTEM_DATA_REPO`` env var or a ``repo=`` argument) is the storage backend.
Shared acquisitions live under ``4dstem/`` and ``haadf/``; ``denoise/`` holds
synthetic inputs and ``widget/`` holds viewer input collections. Each contains one folder
or file per dataset. An Arina acquisition keeps its master + ``_data_*.h5``
chunk siblings together inside its folder, so ``download`` returns a directory
whose master ``quantem.gpu.io.load`` can read. ``download`` accepts a qualified path or a unique
flat name. Tutorials reuse these inputs instead of copying them per viewer.

The verbs are plain English for a microscopist audience: ``upload``, ``download``,
``list_datasets``, ``delete``, ``status``. ``huggingface_hub`` is imported lazily
so importing ``quantem.data`` on a CUDA-less laptop stays cheap; it is a regular
install dependency so a single ``pip install`` gives both the loader and this path.
"""
from __future__ import annotations

import os
import time
import warnings
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Literal

DEFAULT_REPO = "bobleesj/quantem-data"

# The only buckets that hold shareable DATASETS. Other top-level folders in the repo
# (e.g. `notebooks/` for Colab demos) are not data and must not show up in list / status
# / tree / browse as if they were downloadable acquisitions.
DATA_BUCKETS = ("4dstem", "haadf", "denoise", "widget")
DATA_SUFFIXES = {
    ".h5", ".hdf5", ".qem", ".emd", ".npy", ".npz", ".tif", ".tiff",
    ".mrc", ".mrcs", ".dm3", ".dm4", ".raw", ".bin", ".png", ".jpg",
    ".jpeg",
}

# A dataset's calibration travels as a YAML sidecar (humans browsing the repo read clean YAML,
# not JSON braces). Older datasets used JSON, so read_meta still accepts it; YAML is preferred
# when both exist. These suffixes/names are NOT datasets themselves - they ride along a dataset.
SIDECAR_SUFFIXES = (".yaml", ".yml", ".json")
SIDECAR_BASENAMES = (
    "meta.yaml", "meta.yml", "quantem_meta.json", "meta.json", "metadata.json",
)


def _dump_sidecar(sidecar: dict) -> str:
    """Serialize a calibration dict to YAML text, keys in insertion order (modality first)."""
    import yaml  # noqa: PLC0415
    return yaml.safe_dump(sidecar, sort_keys=False)


def parse_sidecar(path: str | Path) -> dict:
    """Read a calibration sidecar by extension: YAML for ``.yaml``/``.yml``, else JSON. Used for
    both downloaded sidecars and a local ``--meta`` file, so the two never parse differently."""
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml  # noqa: PLC0415
        return yaml.safe_load(text)
    import json  # noqa: PLC0415
    return json.loads(text)


def _resolve_repo(repo: str | None) -> str:
    """Pick the dataset repo: explicit arg, else env, else the project default."""
    return repo or os.environ.get("QUANTEM_DATA_REPO") or DEFAULT_REPO


def _hf():
    """Import huggingface_hub lazily, with a clear install hint when it is missing.

    Our datasets are PUBLIC, so no token is ever needed. huggingface_hub still nudges
    anonymous users ("set a HF_TOKEN for higher rate limits", "HF_TOKEN secret does not
    exist") on every request - noise that wrongly implies auth is required. We mute it: the
    env var stops the implicit-token lookup, and the library's own logging is set to
    errors-only AFTER import (doing it before does nothing, the import reconfigures the
    logger). Real errors still surface."""
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    warnings.filterwarnings("ignore", message=r"(?s).*HF_TOKEN.*")
    try:
        import huggingface_hub  # noqa: PLC0415
        from huggingface_hub.utils import logging as hf_logging  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to share datasets. "
            "Install it with `pip install huggingface_hub`."
        ) from exc
    hf_logging.set_verbosity_error()
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
    from the master) and written as a ``meta.yaml`` sidecar travelling with the
    dataset; ``read_meta`` returns it on the other side (it also reads the older
    ``meta.json``/``quantem_meta.json`` names so existing datasets keep working).

    Examples
    --------
    >>> upload("scan.emd", name="gold_run", folder="haadf", meta={"voltage_kV": 300})
    'https://huggingface.co/datasets/bobleesj/quantem-data/commit/...'
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


def _derive_velox_meta(path: Path) -> dict:
    """Read calibration from a Velox ``.emd`` so an uploaded HAADF carries its own voltage,
    semiangle, magnification, FOV and scan shape - a collaborator who downloads it would
    otherwise have none. Best-effort: returns ``{}`` if the parse fails, since a metadata
    convenience must never block the upload itself.
    """
    try:
        from quantem.data.metadata import parse_velox_emd_metadata  # noqa: PLC0415
        parsed = parse_velox_emd_metadata(path)
    except (OSError, KeyError, ValueError, ImportError):
        return {}
    scan = parsed.get("scan_size") or {}
    mag = parsed.get("stem_magnification_x")
    out = {
        "voltage_kV": parsed.get("beam_energy_kv"),
        "semiangle_mrad": parsed.get("convergence_semiangle_mrad"),
        "magnification_MX": mag / 1e6 if mag else None,  # Velox stores plain times -> Megatimes
        "scan_fov_nm": parsed.get("full_scan_field_of_view_nm"),
        "scan_shape": [scan["height"], scan["width"]] if scan.get("height") and scan.get("width") else None,
        "date": parsed.get("acquisition_date"),
    }
    return {k: v for k, v in out.items() if v is not None}


def auto_meta(path: str | Path) -> dict:
    """The calibration ``upload`` reads straight from the file, so a user is never asked for it.

    A Velox ``.emd`` yields voltage / semiangle / magnification / FOV / scan shape; an Arina
    folder yields detector and scan shape. Empty if the file carries nothing extractable. The
    interactive upload calls this first and only prompts for the fields it does NOT return.

    Examples
    --------
    >>> auto_meta("scan.emd")
    {'voltage_kV': 300.0, 'semiangle_mrad': 25.0, 'magnification_MX': 1.3, ...}
    """
    return _build_meta(Path(path), None)


def _build_meta(src: Path, meta: dict | None) -> dict:
    """Auto-derive the calibration a raw file cannot carry, then layer operator ``meta`` on top.

    A 4D-STEM folder yields det_shape/scan_shape from its Arina master; a Velox ``.emd`` yields
    voltage/semiangle/magnification/FOV/scan_shape from its embedded metadata. Explicit operator
    ``meta`` is applied last so it always wins over anything auto-derived."""
    out: dict = {}
    if src.is_dir():
        out.update(_derive_4dstem_shapes(src))
    elif src.suffix.lower() == ".emd":
        out.update(_derive_velox_meta(src))
    if meta:
        out.update({k: v for k, v in meta.items() if v is not None})
    return out


def _upload_meta(hf, repo_id: str, folder: str, name: str,
                 sidecar: dict, *, is_dir: bool) -> None:
    """Write the calibration sidecar (YAML) next to the dataset.

    Folder dataset -> ``<bucket>/<name>/meta.yaml`` inside the folder (download
    returns the dir, so it rides along). File dataset -> a sibling
    ``<bucket>/<name>.yaml`` (the same stem ``delete`` already removes).
    """
    import tempfile  # noqa: PLC0415
    path_in_repo = (f"{folder}/{name}/meta.yaml" if is_dir
                    else f"{folder}/{name}.yaml")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(_dump_sidecar(sidecar))
        tmp = fh.name
    try:
        hf.upload_file(path_or_fileobj=tmp, path_in_repo=path_in_repo,
                        repo_id=repo_id, repo_type="dataset")
    finally:
        os.unlink(tmp)


def read_meta(
    name: str, *, repo: str | None = None, revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> dict | None:
    """Return a dataset's calibration sidecar, or ``None`` if it has none.

    The counterpart to ``upload(..., meta=...)``: a collaborator who downloads a
    4D-STEM acquisition gets back the scan sampling / FOV / voltage / semiangle the
    raw detector master cannot carry. Public repos need no token. A qualified
    name selects the bucket explicitly; ``revision`` selects the same commit
    used when downloading the dataset.

    Parameters
    ----------
    name : str
        Qualified dataset ID or unique short name.
    repo : str, optional
        Dataset repository; defaults to ``bobleesj/quantem-data``.
    revision : str, optional
        Commit or tag to read. Use the download's immutable commit for matching
        calibration. Omit to read the default branch.
    cache_dir : str or pathlib.Path, optional
        Hugging Face cache root, matching ``download(cache_dir=...)``.

    Returns
    -------
    dict or None
        Parsed metadata, or ``None`` if no matching dataset sidecar exists.

    Raises
    ------
    ValueError
        If a short name matches multiple datasets; use a qualified ID.

    Examples
    --------
    >>> read_meta("gold_512")
    {'voltage_kV': 300, 'semiangle_mrad': 25.0, 'scan_shape': [512, 512]}
    """
    hf = _hf()
    repo_id = _resolve_repo(repo)
    files = hf.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    target = _find_sidecar(files, name)
    if target is None:
        return None
    local = hf.hf_hub_download(
        repo_id=repo_id, repo_type="dataset", filename=target, revision=revision,
        cache_dir=cache_dir,
    )
    return parse_sidecar(local)


def _find_sidecar(files: list[str], name: str) -> str | None:
    """The repo path of a dataset's calibration sidecar, preferring YAML over legacy JSON.

    A folder dataset keeps it inside (``<name>/meta.yaml``); a single file keeps it as a sibling
    (``<name>.yaml``). Both YAML and the older JSON names are accepted so existing datasets stay
    readable; if a dataset somehow has both, YAML wins."""
    try:
        target, kind = find_dataset(files, name)
    except FileNotFoundError:
        return None
    if kind == "dir":
        candidates = [f"{target}/{basename}" for basename in SIDECAR_BASENAMES]
    else:
        candidates = _sibling_sidecars(target)
    return next((candidate for candidate in candidates if candidate in files), None)


def _sibling_sidecars(repo_path: str) -> list[str]:
    """Sidecar paths a single-file dataset may have, preferred first.

    ``haadf/gold.tif`` gives ``haadf/gold.yaml``, ``haadf/gold.yml``, ``haadf/gold.json``.
    Repo paths always use ``/``. ``PurePosixPath`` keeps that on Windows, where ``Path``
    answers ``haadf\\gold.yaml``; that matches nothing in a repo listing, so the
    calibration would be neither found nor downloaded."""
    return [str(PurePosixPath(repo_path).with_suffix(suffix)) for suffix in SIDECAR_SUFFIXES]


def dataset_entry(parts: list[str]) -> tuple[str, str, Literal["dir", "file"]] | None:
    """``(display_key, repo_path, kind)`` for a repo path that is a dataset, else ``None``.

    The single source of truth for "what counts as a shared dataset" - ``list_datasets``,
    ``status`` and ``find_dataset`` all defer to it, so they can never disagree about which
    paths are datasets or what their names are. A path qualifies only if it sits in a data
    bucket, is not a placeholder, and contains a recognized data file.
    Documentation, source code and metadata alone do not declare a dataset.

    ``display_key`` is ``<bucket>/<flat-name>`` (what users see; a single file's extension is
    stripped). ``repo_path`` is the actual thing download/delete act on - the folder, or the
    data file WITH its extension. ``kind`` is ``"dir"`` or ``"file"``."""
    if (len(parts) < 2 or parts[0] not in DATA_BUCKETS
            or parts[1].startswith("placeholder_")):
        return None
    suffix = Path(parts[-1]).suffix.lower()
    # A widget monitor is numerical input, even without a detector acquisition.
    if suffix not in DATA_SUFFIXES and not (parts[0] == "widget" and suffix == ".jsonl"):
        return None
    if len(parts) >= 3:  # a payload under <bucket>/<name>/ declares the dataset
        key = f"{parts[0]}/{parts[1]}"
        return key, key, "dir"
    return f"{parts[0]}/{Path(parts[1]).stem}", f"{parts[0]}/{parts[1]}", "file"


def find_dataset(files: list[str], name: str) -> tuple[str, Literal["dir", "file"]]:
    """Resolve a qualified or unique flat name to its ``(repo_path, kind)``.

    Resolving by flat name is what lets a user say ``"gold_512"`` without knowing its bucket.
    Raises ``FileNotFoundError`` if nothing matches, ``ValueError`` if the name is ambiguous
    across buckets - so download and delete can never silently hit the wrong one. Pure (no
    network): the resolution rules are unit-tested against a plain list of paths.
    Use ``haadf/gold`` (or ``haadf/gold.tif``) when a flat name is ambiguous.
    """
    found: dict[str, Literal["dir", "file"]] = {}  # repo_path -> kind
    for f in files:
        entry = dataset_entry(f.split("/"))
        if entry and name in (entry[0], entry[1], entry[0].split("/", 1)[1]):
            found[entry[1]] = entry[2]
    if not found:
        raise FileNotFoundError(
            f"{name!r} not found. Call quantem.data.list_datasets() "
            "(or `quantem-data list`) to see available names.")
    if name in found:  # a folder and a file can share a display key; the exact repo path still names one
        return name, found[name]
    if len(found) > 1:
        raise ValueError(
            f"{name!r} is ambiguous: {sorted(found)}. Pass a qualified dataset "
            "path from this list.")
    return next(iter(found.items()))


def download(
    name: str, *, repo: str | None = None, out: str | Path | None = None,
    verbose: bool = True, revision: str | None = None,
    cache_dir: str | Path | None = None, force_download: bool = False,
) -> Path:
    """Download one shared dataset by name and return its local path.

    The collaborator names just the dataset (``"gold_512"``); ``find_dataset`` searches
    every bucket to find where it lives, so they never need to know it is under
    ``4dstem/`` or ``haadf/``. Returns a directory for a multi-file acquisition
    (pass its master file to ``quantem.gpu.io.load``) or the file path for a single-file
    dataset. Public repos need no token. A qualified name, such as
    ``denoise/simulated_au_nanobeam_pg_dose1``, selects the bucket explicitly.
    ``revision`` accepts a commit or tag and applies to both the listing and
    download. Use an immutable commit for reproducible tutorials.

    Only this dataset and its companions are fetched. Hugging Face reuses cached
    files for the selected revision; a repeat call may still check the network.
    Its normal transfer checks apply. This function checks that all selected
    files exist, but does not rehash an existing cache or validate scientific
    calibration. Use a new cache or ``force_download=True`` to replace damaged
    cached files. ``verbose`` frames the Hub's per-file progress with a summary.

    Parameters
    ----------
    name : str
        Qualified dataset ID or unique short name.
    repo : str, optional
        Dataset repository; defaults to ``bobleesj/quantem-data``.
    out : str or pathlib.Path, optional
        Optional export directory. It preserves repository-relative paths but
        does not isolate revisions; use a separate directory for each revision.
    verbose : bool, optional
        Whether to print progress and a local-path summary.
    revision : str, optional
        Commit or tag for both listing and transfer. An immutable commit pins
        the input; ``None`` uses the repository's default branch.
    cache_dir : str or pathlib.Path, optional
        Hugging Face cache root. Cached snapshots isolate repository revisions.
        The Hub uses ``out`` instead when an export directory is supplied.
    force_download : bool, optional
        Fetch selected files again, even when present in the cache.

    Returns
    -------
    pathlib.Path
        Dataset directory or single payload file. Metadata and companions are
        retained next to their payloads.

    Raises
    ------
    FileNotFoundError
        If the dataset is absent or the completed snapshot is missing a file.
    ValueError
        If a short name is ambiguous. Pass one of the listed qualified IDs.

    Examples
    --------
    >>> path = download("gold_drift_0deg")
    >>> path.is_file()
    True
    """
    hf = _hf()
    repo_id = _resolve_repo(repo)
    files = hf.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    target_rel, kind = find_dataset(files, name)
    if kind == "dir":
        selected = [path for path in files if path.startswith(f"{target_rel}/")]
        patterns = f"{target_rel}/*"
    else:
        sidecars = _sibling_sidecars(target_rel)
        selected = [path for path in files if path == target_rel or path in sidecars]
        patterns = selected if len(selected) > 1 else target_rel
    if verbose:
        print(f"Fetching '{name}' from Hugging Face ({repo_id}, "
              f"revision={revision or 'default branch'}); reusing cached files "
              "when available ...", flush=True)
    t0 = time.perf_counter()
    was_disabled = False
    if not verbose:
        try:
            from huggingface_hub.utils import (  # noqa: PLC0415
                are_progress_bars_disabled,
                disable_progress_bars,
                enable_progress_bars,
            )
            was_disabled = are_progress_bars_disabled()
            disable_progress_bars()
        except ImportError:
            enable_progress_bars = None
    else:
        enable_progress_bars = None
    try:
        root = hf.snapshot_download(
            repo_id=repo_id, repo_type="dataset",
            allow_patterns=patterns,
            local_dir=str(out) if out is not None else None,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
        )
    finally:
        if enable_progress_bars is not None and not was_disabled:
            enable_progress_bars()
    result = Path(root) / target_rel
    missing = [path for path in selected if not (Path(root) / path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete download of {name!r} at revision {revision!r}: "
            f"missing {missing}. Retry with force_download=True or a new cache_dir."
        )
    if verbose:
        dt = time.perf_counter() - t0
        gb = sum((Path(root) / path).stat().st_size for path in selected) / 1e9
        print(f"Ready: '{name}' ({gb:.2f} GB, {dt:.1f}s).\n"
              f"  local path: {result}", flush=True)
    return result


def list_datasets(
    *, repo: str | None = None, revision: str | None = None,
) -> list[str]:
    """List shared datasets as ``<bucket>/<name>`` (skips placeholders/docs).

    Parameters
    ----------
    repo : str, optional
        Dataset repository; defaults to ``bobleesj/quantem-data``.
    revision : str, optional
        Commit or tag to list. Omit for the default branch.

    Returns
    -------
    list[str]
        Sorted qualified IDs with a recognized payload in the selected revision.

    Examples
    --------
    >>> list_datasets()
    ['4dstem/gold_512', 'haadf/gold_drift_0deg']
    """
    hf = _hf()
    repo_id = _resolve_repo(repo)
    files = hf.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    return sorted({e[0] for f in files if (e := dataset_entry(f.split("/")))})


class _DatasetCatalog:
    """Notebook-friendly dataset summary returned by ``datasets()``.

    This is deliberately a lightweight view over ``status()``: no data files are
    downloaded, and each row shows how to load one acquisition or download a
    widget collection for a viewer.
    """

    def __init__(self, rows: list[dict], repo_id: str):
        self.rows = rows
        self.repo_id = repo_id

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def _repr_html_(self) -> str:
        def esc(value) -> str:
            import html  # noqa: PLC0415
            return html.escape(str(value))

        body = "".join(
            "<tr>"
            f"<td><code>{esc(row['name'])}</code></td>"
            f"<td>{esc(row['modality'])}</td>"
            f"<td style=\"text-align:right\">{row['size_mb']:.1f} MB</td>"
            f"<td><code>qdata.{'download' if row['modality'] in ('widget', 'denoise') else 'load'}"
            f"(&quot;{esc(row['name'])}&quot;)</code></td>"
            "</tr>"
            for row in self.rows
        )
        return (
            '<table style="border-collapse:collapse">'
            "<thead><tr><th>Name</th><th>Data</th><th>Size</th><th>Use</th></tr></thead>"
            f"<tbody>{body}</tbody></table>"
        )

    def __repr__(self) -> str:
        if not self.rows:
            return "No datasets found."
        widths = {
            "name": max(len("name"), *(len(row["name"]) for row in self.rows)),
            "modality": max(len("data"), *(len(row["modality"]) for row in self.rows)),
        }
        lines = [
            f"{'name':<{widths['name']}}  {'data':<{widths['modality']}}  size      use",
            f"{'-' * widths['name']}  {'-' * widths['modality']}  --------  ------------------------",
        ]
        for row in self.rows:
            verb = "download" if row["modality"] in ("widget", "denoise") else "load"
            lines.append(
                f"{row['name']:<{widths['name']}}  "
                f"{row['modality']:<{widths['modality']}}  "
                f"{row['size_mb']:>7.1f} MB  qdata.{verb}({row['name']!r})"
            )
        return "\n".join(lines)


def datasets(*, repo: str | None = None, modality: str | None = None) -> _DatasetCatalog:
    """Return a notebook-friendly catalog of available public datasets.

    Unlike ``list_datasets()``, this is meant for humans reading a notebook: it
    shows the name, data type, size, and call (download for workflow collections).
    It never downloads the actual data.

    Parameters
    ----------
    repo
        Hugging Face dataset repo. Defaults to ``bobleesj/quantem-data``.
    modality
        Optional bucket filter such as ``"haadf"`` or ``"4dstem"``.

    Examples
    --------
    >>> datasets(modality="haadf")
    name             data   size      use
    ...              ...    ...       qdata.load('...')
    """
    snap = status(repo=repo)
    names = Counter(item["name"].split("/", 1)[1] for item in snap["datasets"])
    rows = []
    for item in snap["datasets"]:
        bucket, short = item["name"].split("/", 1)
        if modality is not None and bucket != modality:
            continue
        rows.append({
            "name": short if names[short] == 1 else item["name"],
            "modality": bucket,
            "size_mb": item["size_mb"],
        })
    return _DatasetCatalog(rows, snap["repo"])


def delete(name: str, *, repo: str | None = None) -> list[str]:
    """Delete a shared dataset by flat name; returns the repo paths removed.

    A folder dataset removes the whole folder; a file dataset removes the data
    file and its ``.json`` sidecar (same stem). Refuses to act when the name
    matches more than one dataset so a delete never nukes the wrong bucket. The
    CLI adds a re-type-to-confirm prompt; this function deletes immediately.
    """
    hf = _hf()
    repo_id = _resolve_repo(repo)
    files = hf.list_repo_files(repo_id=repo_id, repo_type="dataset")
    target_rel, kind = find_dataset(files, name)
    deleted = []
    if kind == "dir":
        hf.delete_folder(path_in_repo=target_rel, repo_id=repo_id, repo_type="dataset")
        deleted.append(f"{target_rel}/")
    else:
        bucket = target_rel.split("/", 1)[0]
        for f in files:  # the data file AND its .json calibration sidecar (same stem) go together
            parts = f.split("/")
            if (f == target_rel or (len(parts) == 2 and parts[0] == bucket
                    and Path(parts[1]).stem == Path(target_rel).stem
                    and Path(parts[1]).suffix.lower() in SIDECAR_SUFFIXES)):
                hf.delete_file(path_in_repo=f, repo_id=repo_id, repo_type="dataset")
                deleted.append(f)
    return deleted


def status(*, repo: str | None = None) -> dict:
    """Snapshot of the shared repo: auth, datasets + sizes, and local cache size.

    Answers the operator's "where does my data live, can I upload, what is shared,
    what do I already have locally" in one call. No token needed for the dataset
    listing; auth is reported as whoever is logged in (or ``None`` = download-only).

    Examples
    --------
    >>> snap = status()
    >>> snap["logged_in_as"], len(snap["datasets"])
    ('bobleesj', 8)
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
    sizes: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    entries = list(api.list_repo_tree(repo_id, repo_type="dataset", recursive=True))
    folders = {
        record[1] for entry in entries if getattr(entry, "size", None) is not None
        and (record := dataset_entry(entry.path.split("/"))) and record[2] == "dir"
    }
    for entry in entries:
        size = getattr(entry, "size", None)
        if size is None:
            continue  # folder entry, not a file
        classified = dataset_entry(entry.path.split("/"))
        folder = "/".join(entry.path.split("/")[:2])
        if folder in folders:
            classified = (folder, folder, "dir")
        if classified is None:
            continue  # not a dataset file (wrong bucket, placeholder, or a top-level sidecar)
        key = classified[0]
        sizes[key] += size
        counts[key] += 1
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
    at a glance, so a user can pick a dataset without opening Hugging Face in a browser.

    Examples
    --------
    >>> tree()
    bobleesj/quantem-data  (12.3 GB total, 8 datasets)
    ...
    """
    print(_tree_text(repo=repo))


def _tree_text(*, repo: str | None = None) -> str:
    """The dataset tree as text, shared by ``tree()`` and ``browse().__repr__()``."""
    groups, snap = _by_bucket(repo=repo)
    lines = [f"{snap['repo']}  ({snap['total_mb'] / 1000:.1f} GB total, {len(snap['datasets'])} datasets)"]
    for bucket in sorted(groups):
        items = sorted(groups[bucket], key=lambda d: d["name"])
        gb = sum(d["size_mb"] for d in items) / 1000
        lines.append(f"\n{bucket}/   ({len(items)} datasets, {gb:.2f} GB)")
        for d in items:
            lines.append(f"  {d['size_mb']:>9.1f} MB  {d['name'].split('/', 1)[1]}")
    return "\n".join(lines)


THUMB_DIR = ".thumbnails"  # top-level tree, OUTSIDE the data buckets, so a thumbnail
# never looks like a dataset folder (a `<bucket>/<name>/thumbnail.png` made single-file
# datasets ambiguous with their data file in `download`).


def _thumb_data_uri(full_name: str, repo_id: str) -> str | None:
    """Fetch only ``.thumbnails/<bucket>/<name>.png`` (KB, not the data) and return it as a
    base64 data URI for inline display, or None if the dataset has no thumbnail yet."""
    import base64  # noqa: PLC0415
    hf = _hf()
    was_disabled = False
    try:
        from huggingface_hub.utils import (  # noqa: PLC0415
            are_progress_bars_disabled,
            disable_progress_bars,
            enable_progress_bars,
        )
        was_disabled = are_progress_bars_disabled()
        disable_progress_bars()
    except ImportError:
        enable_progress_bars = None
    try:
        path = hf.hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=f"{THUMB_DIR}/{full_name}.png",
        )
    except (hf.errors.EntryNotFoundError, hf.errors.HfHubHTTPError):
        return None  # no thumbnail uploaded for this dataset yet
    finally:
        if enable_progress_bars is not None and not was_disabled:
            enable_progress_bars()
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
            f'<div style="font-size:10px;color:#999">download("{full}")</div></div>')


class _Gallery:
    """Renders a thumbnail grid inline in Jupyter (``_repr_html_``); prints the tree in a
    terminal (``__repr__``). Returned by ``browse()`` so the notebook displays it directly."""

    def __init__(self, repo: str | None):
        self._groups, self._snap = _by_bucket(repo=repo)
        self._repo_id = _resolve_repo(repo)

    def _repr_html_(self) -> str:
        parts = [f'<div style="font-family:sans-serif"><b>{self._snap["repo"]}</b> '
                 f'- {self._snap["total_mb"]/1000:.1f} GB, {len(self._snap["datasets"])} datasets']
        for bucket in sorted(self._groups):
            items = sorted(self._groups[bucket], key=lambda d: d["name"])
            parts.append(f'<h4 style="margin:12px 0 0">{bucket}/ '
                         f'<span style="font-weight:400;color:#888">({len(items)})</span></h4>')
            parts.append("".join(_card_html(d, self._repo_id) for d in items))
        parts.append("</div>")
        return "".join(parts)

    def __repr__(self) -> str:
        repo = self._repo_id if self._repo_id != DEFAULT_REPO else None
        return _tree_text(repo=repo)


def browse(*, repo: str | None = None) -> "_Gallery":
    """Visual dataset picker: a grid of thumbnails + names + sizes you can scan before
    downloading anything. In Jupyter it renders inline; in a terminal it prints the tree.
    Thumbnails are tiny (KB) and fetched on the fly - the multi-GB data is never touched.
    Pick a name and pass it to ``download(...)``.

    Examples
    --------
    >>> browse()   # in Jupyter: a thumbnail grid; in a terminal: the tree
    """
    return _Gallery(repo)


def _image_payload(path: Path) -> Path | None:
    """Return the image file inside a folder dataset, if it is an image folder.

    Hugging Face stores some single images as folders so their metadata can ride
    next to ``data.npy``. Those are still HAADF image datasets from a user's
    perspective, not 4D-STEM acquisitions.
    """
    image_suffixes = {".npy", ".tif", ".tiff", ".emd", ".png", ".jpg", ".jpeg"}
    if path.is_file():
        return path
    preferred = path / "data.npy"
    if preferred.exists():
        return preferred
    payloads = [
        child
        for child in path.iterdir()
        if child.is_file() and child.suffix.lower() in image_suffixes
    ]
    return payloads[0] if len(payloads) == 1 else None


def _apply_image_meta(dataset, meta: dict | None):
    """Attach sidecar calibration to an image dataset when the object supports it."""
    if not meta:
        return dataset
    for attr in ("name", "sampling", "units", "signal_units"):
        if attr in meta and hasattr(dataset, attr):
            try:
                setattr(dataset, attr, meta[attr])
            except (AttributeError, TypeError):
                pass
    if hasattr(dataset, "metadata"):
        try:
            dataset.metadata.update(meta)
        except (AttributeError, TypeError):
            pass
    return dataset


def load(
    name: str,
    *,
    repo: str | None = None,
    out: str | Path | None = None,
    verbose: bool = True,
    revision: str | None = None,
    **kwargs,
):
    """Download a dataset AND return it ready to use, in one call.

    A single image returns a ``Dataset2d`` through ``quantem.widget.io.read_image``.
    Native 4D-STEM inputs use ``quantem.gpu.io.load``: CUDA/MPS acquisitions
    retain the loader's encoded storage and native detector geometry. Extra
    keyword arguments pass to that loader, such as ``backend="cuda"`` or
    ``hot_pixel_correction="none"``. Read bounded regions from the returned
    acquisition and close it when finished. On CUDA and MPS a master-file folder
    returns a list of independent acquisitions; close each one. With
    ``backend="cpu"`` the loader stacks them into one dataset unless
    ``stack=False`` is passed. Both optional packages are
    imported only for their respective input type; download alone needs neither.
    ``revision`` pins the downloaded input and metadata together and is not
    forwarded to either loader. Set ``verbose=False`` to suppress the transfer
    summary.

    Examples
    --------
    >>> ds = load("gold_drift_0deg")
    >>> ds.array.shape
    (2048, 2048)
    """
    path = download(name, repo=repo, out=out, verbose=verbose, revision=revision)
    if path.is_dir() and path.parent.name == "widget":
        raise ValueError(
            f"{name!r} is a widget input collection, not one acquisition. "
            "Use download() and the matching quantem.widget.datasets tutorial helper "
            "to open its monitor or image session."
        )
    image_path = _image_payload(path)
    if path.parent.name == "4dstem" or image_path is None:
        try:
            from quantem.gpu import io as gpu_io  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Loading native 4D-STEM inputs requires quantem.gpu. "
                "Install it with `pip install quantem.gpu` and use a CUDA or MPS "
                "device; download() alone returns the path without loading data."
            ) from exc
        source = path / "data.npy" if path.is_dir() and (path / "data.npy").is_file() else path
        if source.is_dir():
            source = gpu_io.discover(str(source), verbose=verbose)
        return gpu_io.load(source, **kwargs)
    try:
        from quantem.widget import io as widget_io  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Loading images requires quantem.widget: pip install quantem.widget "
            "(download() alone returns the file path without it)."
        ) from exc
    return _apply_image_meta(
        widget_io.read_image(image_path), read_meta(name, repo=repo, revision=revision),
    )
