"""
quantem.data — Real electron microscopy datasets for quantem widgets.

Downloads and caches data from Hugging Face Hub. Works on Google Colab
out of the box (huggingface_hub is pre-installed).

Usage
-----
>>> from quantem.data import load, available
>>> available()
['korean_sample_c1']
>>> data = load("korean_sample_c1")
>>> data.shape
(256, 256)
"""

__version__ = "0.0.2a2"

from quantem.data.registry import (
    available,
    info,
    load,
    load_raw,
    preview_upload,
    upload,
    update_metadata,
    list_files,
)
from quantem.data.schema import validate, make_template
from quantem.data.browser import DataBrowser

__all__ = [
    "__version__",
    "available",
    "info",
    "load",
    "load_raw",
    "preview_upload",
    "upload",
    "update_metadata",
    "list_files",
    "validate",
    "make_template",
    "DataBrowser",
]
