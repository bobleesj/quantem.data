"""quantem.data: share 4D-STEM / HAADF datasets through one Hugging Face repo.

The whole API is a handful of verbs on this package. The most common one:

    from quantem.data import load
    ds = load("gold_drift_0deg")   # downloads, then returns the data ready to use

Implementation lives in ``quantem.data.huggingface``; this module is just the facade.
"""
from importlib.metadata import PackageNotFoundError, version

from quantem.data.huggingface import (
    browse,
    download,
    list_datasets,
    load,
    read_meta,
    status,
    tree,
    upload,
)

try:
    __version__ = version("quantem.data")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "__version__",
    "browse",
    "download",
    "list_datasets",
    "load",
    "read_meta",
    "status",
    "tree",
    "upload",
]
