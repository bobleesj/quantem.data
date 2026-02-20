"""DataBrowser — Interactive Jupyter widget for browsing quantem.data datasets."""

import json
import pathlib

import anywidget
import numpy as np
import traitlets

from quantem.data.registry import available, info, load
from quantem.data.schema import VALID_TECHNIQUES


class DataBrowser(anywidget.AnyWidget):
    """Interactive browser for quantem.data datasets on HF Hub.

    Displays a filterable table of available datasets. Click a dataset
    to see its metadata, then click Load to download it into memory.

    Parameters
    ----------
    technique : str, optional
        Initial technique filter (e.g. ``"4dstem"``, ``"hrtem"``).

    Examples
    --------
    >>> from quantem.data import DataBrowser
    >>> browser = DataBrowser()
    >>> browser  # displays widget in notebook
    >>> # After selecting and loading a dataset in the UI:
    >>> browser.data.shape
    (256, 256)
    """

    _esm = pathlib.Path(__file__).parent / "static" / "browser.js"

    # Python → JS: full catalog as JSON
    catalog_json = traitlets.Unicode("[]").tag(sync=True)
    # JS → Python: which dataset the user clicked
    selected_name = traitlets.Unicode("").tag(sync=True)
    # Python → JS: metadata for selected dataset
    selected_info_json = traitlets.Unicode("").tag(sync=True)
    # JS ↔ Python: technique filter
    technique_filter = traitlets.Unicode("").tag(sync=True)
    # Python → JS: loading indicator
    loading = traitlets.Bool(False).tag(sync=True)
    # Python → JS: name of loaded dataset
    loaded_name = traitlets.Unicode("").tag(sync=True)
    # JS → Python: trigger refresh
    _refresh_requested = traitlets.Bool(False).tag(sync=True)
    # JS → Python: trigger load
    _load_requested = traitlets.Bool(False).tag(sync=True)

    def __init__(self, technique: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self._data: np.ndarray | None = None
        self._metadata: dict | None = None

        if technique:
            self.technique_filter = technique

        self.observe(self._on_selected_name_change, names=["selected_name"])
        self.observe(self._on_refresh_requested, names=["_refresh_requested"])
        self.observe(self._on_load_requested, names=["_load_requested"])

        self.refresh()

    def refresh(self):
        """Re-fetch the dataset catalog from HF Hub."""
        self.loading = True
        try:
            names = available()
            catalog = []
            for name in names:
                try:
                    meta = info(name)
                    entry = {
                        "name": name,
                        "technique": meta.get("technique", ""),
                        "shape": meta.get("data", {}).get("shape", []),
                        "dtype": meta.get("data", {}).get("dtype", ""),
                        "description": meta.get("description", ""),
                        "size_mb": _estimate_size(meta),
                    }
                    catalog.append(entry)
                except Exception:
                    catalog.append({
                        "name": name,
                        "technique": "",
                        "shape": [],
                        "dtype": "",
                        "description": "",
                        "size_mb": 0,
                    })
            self.catalog_json = json.dumps(catalog)
        finally:
            self.loading = False

    def _on_selected_name_change(self, change):
        name = change["new"]
        if not name:
            self.selected_info_json = ""
            return
        try:
            meta = info(name)
            meta.pop("_npy_path", None)
            self.selected_info_json = json.dumps(meta)
        except Exception:
            self.selected_info_json = ""

    def _on_refresh_requested(self, change):
        if change["new"]:
            self._refresh_requested = False
            self.refresh()

    def _on_load_requested(self, change):
        if change["new"]:
            self._load_requested = False
            name = self.selected_name
            if name:
                self._load_dataset(name)

    def _load_dataset(self, name: str):
        self.loading = True
        try:
            arr, meta = load(name, metadata=True)
            meta.pop("_npy_path", None)
            self._data = arr
            self._metadata = meta
            self.loaded_name = name
        except Exception as e:
            self._data = None
            self._metadata = None
            self.loaded_name = ""
            print(f"Failed to load {name}: {e}")
        finally:
            self.loading = False

    @property
    def data(self) -> np.ndarray | None:
        """The loaded dataset as a NumPy array, or None if nothing is loaded."""
        return self._data

    @property
    def metadata(self) -> dict | None:
        """Metadata dict for the loaded dataset, or None."""
        return self._metadata

    @property
    def techniques(self) -> list[str]:
        """Valid technique names."""
        return list(VALID_TECHNIQUES)

    def summary(self):
        """Print a human-readable summary of the browser state."""
        catalog = json.loads(self.catalog_json)
        print(f"DataBrowser")
        print(f"  Datasets: {len(catalog)}")
        if self.technique_filter:
            filtered = [d for d in catalog if d["technique"] == self.technique_filter]
            print(f"  Filter: {self.technique_filter} ({len(filtered)} shown)")
        if self.loaded_name:
            print(f"  Loaded: {self.loaded_name}")
            if self._data is not None:
                print(f"  Shape: {self._data.shape}")
                print(f"  Dtype: {self._data.dtype}")

    def __repr__(self) -> str:
        catalog = json.loads(self.catalog_json)
        parts = [f"DataBrowser({len(catalog)} datasets"]
        if self.technique_filter:
            parts.append(f"filter={self.technique_filter!r}")
        if self.loaded_name:
            parts.append(f"loaded={self.loaded_name!r}")
        return ", ".join(parts) + ")"


def _estimate_size(meta: dict) -> float:
    """Estimate dataset size in MB from metadata shape + dtype."""
    data = meta.get("data", {})
    shape = data.get("shape", [])
    dtype = data.get("dtype", "float32")
    if not shape:
        return 0
    try:
        nbytes = np.prod(shape) * np.dtype(dtype).itemsize
        return round(nbytes / (1024 * 1024), 2)
    except Exception:
        return 0
