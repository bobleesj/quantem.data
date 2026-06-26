from importlib.metadata import PackageNotFoundError, version

from quantem.data.metadata import parse_velox_emd_metadata
from quantem.data.repository import datasets_dir, load_dataset_metadata
from quantem.data.sync import sync_dataset_metadata
# Hugging Face transfer verbs. The package-level API is the REMOTE (shared-repo) view so
# every package-level verb means the same thing: `list_datasets` lists what's on Hugging
# Face (matching status/download/read_meta). The LOCAL registry listing stays available as
# `quantem.data.repository.list_datasets` for callers that want the synced-on-disk view.
from quantem.data._remote import upload, download, read_meta, status, list_datasets, tree, browse, load

try:
    __version__ = version("quantem.data")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "__version__",
    "datasets_dir",
    "list_datasets",
    "load_dataset_metadata",
    "parse_velox_emd_metadata",
    "sync_dataset_metadata",
    "upload",
    "download",
    "read_meta",
    "status",
    "tree",
    "browse",
    "load",
]
