# DataBrowser widget

The `DataBrowser` is an interactive Jupyter widget for browsing, filtering, and loading datasets from Hugging Face Hub — without writing any code.

## Quick start

```python
from quantem.data import DataBrowser

browser = DataBrowser()
browser
```

This displays a table of all available datasets. Click a row to see metadata, then click **Load** to download it.

## Access loaded data

After clicking **Load** in the widget:

```python
# NumPy array
browser.data
# → array([[...]], dtype=float32)

# Shape and dtype
browser.data.shape
# → (256, 256)

# Metadata dict
browser.metadata
# → {"name": "korean_sample_c1", "technique": "image", ...}

# Dataset name
browser.loaded_name
# → "korean_sample_c1"
```

## Filter by technique

Use the dropdown in the widget UI, or pass a technique at creation:

```python
browser = DataBrowser(technique="4dstem")
```

## Use with quantem.widget

Load a dataset and visualize it directly:

```python
from quantem.data import DataBrowser
from quantem.widget import Show2D

browser = DataBrowser()
browser
# ... select and load a dataset in the widget ...

Show2D(browser.data, title=browser.loaded_name)
```

## Google Colab

The DataBrowser works on Google Colab. Install from TestPyPI:

```python
%pip install -q -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ quantem-data
```

Then use it normally. The widget renders in Colab's output cells.

## Programmatic alternative

If you prefer code over a widget:

```python
from quantem.data import available, info, load

# List all datasets
available()

# Filter by technique
available(technique="4dstem")

# Get metadata
info("korean_sample_c1")

# Load as NumPy array
data = load("korean_sample_c1")
```
