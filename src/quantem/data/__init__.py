from importlib.metadata import PackageNotFoundError, version

from quantem.data.metadata import parse_velox_emd_metadata
from quantem.data.repository import datasets_dir, list_datasets, load_dataset_metadata
from quantem.data.sync import sync_dataset_metadata

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
]
