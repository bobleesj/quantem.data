"""Dataset registry and loader backed by Hugging Face Hub.

Metadata lives in JSON sidecar files on HF Hub, not hardcoded here.
The registry is discovered at runtime by listing the repo contents.
"""

import json
import os
import tempfile

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

from quantem.data.schema import SCHEMA_VERSION, validate, make_template

REPO_ID = "bobleesj/quantem-data"


def _api() -> HfApi:
    return HfApi()


def available(technique: str | None = None) -> list[str]:
    """List available dataset names.

    Parameters
    ----------
    technique : str, optional
        Filter by technique (e.g. ``"4dstem"``, ``"hrtem"``).

    Returns
    -------
    list of str
        Sorted dataset names (excluding placeholders).
    """
    api = _api()
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    names = []
    for f in files:
        if f.endswith(".json") and f != "README.md":
            name = os.path.basename(f).removesuffix(".json")
            if name.startswith("placeholder_"):
                continue
            if technique and not f.startswith(f"{technique}/"):
                continue
            names.append(name)
    return sorted(names)


def info(name: str) -> dict:
    """Return metadata for a dataset (downloads the JSON sidecar).

    Parameters
    ----------
    name : str
        Dataset name (see ``available()``).
    """
    meta = _download_metadata(name)
    return meta


def load(name: str, metadata: bool = False):
    """Download (with caching) and return a dataset as a NumPy array.

    Files are cached in ``~/.cache/huggingface/`` and only downloaded once.

    Parameters
    ----------
    name : str
        Dataset name (see ``available()``).
    metadata : bool
        If True, return ``(array, metadata_dict)`` instead of just the array.

    Returns
    -------
    np.ndarray or (np.ndarray, dict)
    """
    meta = _download_metadata(name)
    npy_path = meta["_npy_path"]
    local = hf_hub_download(
        repo_id=REPO_ID,
        filename=npy_path,
        repo_type="dataset",
    )
    arr = np.load(local)
    if metadata:
        return arr, meta
    return arr


def load_raw(name: str) -> str:
    """Download an original instrument file and return the local path.

    Raw files (e.g. ``.h5``, ``.dm4``, ``.mrc``) are stored in the ``raw/``
    folder. This returns the cached local path for use with h5py, hyperspy, etc.

    Parameters
    ----------
    name : str
        Raw file name (without extension), e.g. ``"arina_lamella_master"``.

    Returns
    -------
    str
        Local file path (cached in ``~/.cache/huggingface/``).
    """
    api = _api()
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")

    # Find matching file in raw/
    matches = [f for f in files if f.startswith("raw/") and os.path.basename(f).startswith(name)]
    if not matches:
        raw_files = [f for f in files if f.startswith("raw/") and not f.endswith(".json")]
        raise KeyError(
            f"Raw file {name!r} not found. Available raw files: "
            f"{[os.path.basename(f) for f in raw_files]}"
        )

    # Prefer exact match (name + extension)
    path = matches[0]
    local = hf_hub_download(
        repo_id=REPO_ID,
        filename=path,
        repo_type="dataset",
    )
    return local


def upload(
    data,
    name: str,
    technique: str,
    metadata: dict | str | None = None,
    description: str = "",
    contributor: str = "",
    license: str = "CC-BY-4.0",
    token: str | None = None,
):
    """Upload a dataset with metadata to HF Hub.

    Parameters
    ----------
    data : array_like or str
        NumPy array, or path to a ``.npy`` file.
    name : str
        Dataset name (becomes the filename, e.g. ``"arina_lamella_32x32"``).
    technique : str
        Category folder (``"4dstem"``, ``"hrtem"``, ``"eels"``, etc.).
    metadata : dict or str, optional
        Full metadata dict, or path to a JSON file. If None, a template
        is created from the other parameters.
    description : str
        One-line description (used if metadata is None).
    contributor : str
        Who is uploading (used if metadata is None).
    license : str
        License string (default ``"CC-BY-4.0"``).
    token : str, optional
        HF token. If None, uses cached login.
    """
    # Resolve data
    if isinstance(data, str):
        arr = np.load(data)
    else:
        arr = np.asarray(data, dtype=np.float32)

    # Resolve metadata
    if metadata is None:
        meta = make_template(
            name=name,
            technique=technique,
            shape=arr.shape,
            dtype=str(arr.dtype),
            description=description,
            contributor=contributor,
            license=license,
        )
    elif isinstance(metadata, str):
        with open(metadata) as f:
            meta = json.load(f)
    else:
        meta = dict(metadata)

    # Ensure required fields
    meta.setdefault("schema_version", SCHEMA_VERSION)
    meta.setdefault("name", name)
    meta.setdefault("technique", technique)

    # Validate
    errors = validate(meta)
    if errors:
        raise ValueError(
            "Metadata validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    # Check shape matches
    meta_shape = meta.get("data", {}).get("shape")
    if meta_shape and tuple(meta_shape) != tuple(arr.shape):
        raise ValueError(
            f"Metadata shape {meta_shape} does not match array shape {list(arr.shape)}"
        )

    # Upload
    api = _api()
    npy_remote = f"{technique}/{name}.npy"
    json_remote = f"{technique}/{name}.json"

    with tempfile.TemporaryDirectory() as tmpdir:
        npy_local = os.path.join(tmpdir, f"{name}.npy")
        json_local = os.path.join(tmpdir, f"{name}.json")

        np.save(npy_local, arr)
        with open(json_local, "w") as f:
            json.dump(meta, f, indent=2)

        api.upload_file(
            path_or_fileobj=npy_local,
            path_in_repo=npy_remote,
            repo_id=REPO_ID,
            repo_type="dataset",
            token=token,
        )
        api.upload_file(
            path_or_fileobj=json_local,
            path_in_repo=json_remote,
            repo_id=REPO_ID,
            repo_type="dataset",
            token=token,
        )

    print(f"Uploaded {name} ({arr.nbytes / (1024**2):.1f} MB) to {REPO_ID}")


def update_metadata(name: str, updates: dict, token: str | None = None):
    """Update metadata fields for an existing dataset.

    Downloads the current JSON, merges your changes, re-uploads.

    Parameters
    ----------
    name : str
        Dataset name.
    updates : dict
        Fields to update. Nested dicts are merged (not replaced).
    token : str, optional
        HF token. If None, uses cached login.
    """
    meta = _download_metadata(name)
    technique = meta["technique"]
    json_remote = f"{technique}/{name}.json"

    # Deep merge
    _deep_merge(meta, updates)

    # Remove internal fields
    meta.pop("_npy_path", None)

    # Validate
    errors = validate(meta)
    if errors:
        raise ValueError(
            "Updated metadata is invalid:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    api = _api()
    with tempfile.TemporaryDirectory() as tmpdir:
        json_local = os.path.join(tmpdir, f"{name}.json")
        with open(json_local, "w") as f:
            json.dump(meta, f, indent=2)
        api.upload_file(
            path_or_fileobj=json_local,
            path_in_repo=json_remote,
            repo_id=REPO_ID,
            repo_type="dataset",
            token=token,
        )

    print(f"Updated metadata for {name}")


def list_files(technique: str | None = None) -> list[dict]:
    """List all files on HF Hub with details.

    Parameters
    ----------
    technique : str, optional
        Filter by technique folder (e.g. ``"4dstem"``). If None, lists all.

    Returns
    -------
    list of dict
        Each dict has ``path``, ``size_mb``, ``type`` (``"data"``/``"metadata"``).
    """
    api = _api()
    repo_info = api.repo_info(repo_id=REPO_ID, repo_type="dataset", files_metadata=True)
    results = []
    for item in repo_info.siblings:
        path = item.rfilename
        if path in (".gitattributes", "README.md"):
            continue
        if technique and not path.startswith(f"{technique}/"):
            continue
        ext = os.path.splitext(path)[1]
        file_type = "metadata" if ext == ".json" else "data"
        size_mb = item.size / (1024 * 1024) if item.size else 0
        results.append({
            "path": path,
            "size_mb": round(size_mb, 2),
            "type": file_type,
        })
    return sorted(results, key=lambda x: x["path"])


def _deep_merge(base: dict, updates: dict):
    """Merge updates into base, recursing into nested dicts."""
    for key, val in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


def _download_metadata(name: str) -> dict:
    """Download and parse the JSON sidecar for a dataset."""
    # Try each technique folder
    api = _api()
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")

    json_file = None
    npy_file = None
    for f in files:
        basename = os.path.basename(f)
        if basename == f"{name}.json":
            json_file = f
        elif basename == f"{name}.npy":
            npy_file = f

    if json_file is None:
        # Fall back: maybe only .npy exists (legacy, no sidecar)
        if npy_file is not None:
            return {
                "name": name,
                "description": "",
                "_npy_path": npy_file,
            }
        names = available()
        raise KeyError(
            f"Unknown dataset {name!r}. Available: {names}"
        )

    local = hf_hub_download(
        repo_id=REPO_ID,
        filename=json_file,
        repo_type="dataset",
    )
    with open(local) as f:
        meta = json.load(f)

    # Attach the npy path for load()
    if npy_file:
        meta["_npy_path"] = npy_file
    else:
        # Infer from json path
        meta["_npy_path"] = json_file.replace(".json", ".npy")

    return meta
