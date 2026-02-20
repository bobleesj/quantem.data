"""
quantem.data — Real electron microscopy datasets for quantem widgets.

Downloads and caches data from Hugging Face Hub. Works on Google Colab
out of the box (huggingface_hub is pre-installed).

Usage
-----
>>> from quantem.data import load, available
>>> available()
['srtio3_bf', 'srtio3_lamella', 'srtio3_lamella_hr', 'srtio3_mean_dp']
>>> data = load("srtio3_lamella")
>>> data.shape
(32, 32, 48, 48)
"""

__version__ = "0.0.1"

from quantem.data.registry import (
    available,
    info,
    load,
    load_raw,
    upload,
    update_metadata,
    list_files,
)
from quantem.data.schema import validate, make_template

__all__ = [
    "__version__",
    "available",
    "info",
    "load",
    "load_raw",
    "upload",
    "update_metadata",
    "list_files",
    "validate",
    "make_template",
]
