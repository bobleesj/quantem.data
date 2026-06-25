from importlib.metadata import PackageNotFoundError, version

from quantem.data.metadata import parse_velox_emd_metadata
from quantem.data.repository import datasets_dir, list_datasets, load_dataset_metadata
from quantem.data.sync import sync_dataset_metadata
# Hugging Face transfer verbs. hub.list_datasets/delete are the REMOTE-repo variants
# (they live on the ``quantem.data.hub`` submodule to avoid colliding with the local
# registry's ``list_datasets`` above); the package-level API exposes the common ones.
from quantem.data.hub import upload, download, read_meta, status

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
]
