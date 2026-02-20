# quantem.data

Real electron microscopy datasets for [quantem.widget](https://github.com/bobleesj/quantem.widget).

Data hosted on [Hugging Face Hub](https://huggingface.co/datasets/bobleesj/quantem-data). Works on Google Colab out of the box.

| Notebook | Description |
|----------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bobleesj/quantem.data/blob/main/notebooks/browser.ipynb) | **Browse & download** — interactive DataBrowser widget |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bobleesj/quantem.data/blob/main/notebooks/upload.ipynb) | **Upload via PR** — contribute data to HF Hub |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bobleesj/quantem.data/blob/main/notebooks/demo.ipynb) | **Quick demo** — API usage with quantem.widget |

## Install

```bash
pip install --pre -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ quantem-data
```

To also visualize datasets with [quantem.widget](https://github.com/bobleesj/quantem.widget) (recommended):

```bash
pip install --pre -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ quantem-data quantem-widget
```

## DataBrowser widget

Browse, filter, and load datasets interactively in a notebook:

```python
from quantem.data import DataBrowser

browser = DataBrowser()
browser  # displays interactive widget

# After selecting and loading a dataset:
browser.data       # NumPy array
browser.metadata   # metadata dict
browser.loaded_name  # dataset name
```

## Usage with quantem.widget

```python
from quantem.data import load
from quantem.widget import Show2D

# load a 2D image (downloads once, cached locally)
Show2D(load("korean_sample_c1"))

# load with metadata
data, meta = load("korean_sample_c1", metadata=True)
Show2D(data, title=meta["description"])
```

## API

```python
from quantem.data import available, info, load, list_files

# List datasets (optionally filter by technique)
available()                          # all datasets
available(technique="image")         # only images

# Dataset metadata (no data download)
info("korean_sample_c1")

# Load processed data as NumPy array
data = load("korean_sample_c1")
data, meta = load("korean_sample_c1", metadata=True)

# List all files on HF Hub
list_files()
list_files("image")
```

## Upload (creates a PR for review)

Uploads create a Pull Request on Hugging Face Hub by default. The data is reviewed before merging.

```python
from quantem.data import preview_upload, upload

# Step 1: preview — validates naming, metadata, checks for duplicates
preview_upload(
    my_array,
    name="gold_nanoparticle",
    technique="hrtem",
    description="Gold nanoparticle HRTEM at 200 kV",
    contributor="Jane Doe",
)

# Step 2: upload — creates PR on HF Hub
upload(
    my_array,
    name="gold_nanoparticle",
    technique="hrtem",
    description="Gold nanoparticle HRTEM at 200 kV",
    contributor="Jane Doe",
)
# → Created PR to add gold_nanoparticle (0.2 MB)
# → Review: https://huggingface.co/datasets/bobleesj/quantem-data/discussions/1
```

## CLI

```bash
# List datasets
quantem-data list
quantem-data list --technique image

# Show metadata
quantem-data info korean_sample_c1

# List files on HF Hub
quantem-data files

# Download
quantem-data download korean_sample_c1

# Upload (creates PR by default)
quantem-data upload my_data.npy --name silicon_110 --technique hrtem \
    --description "Silicon [110] HRTEM" --contributor "Jane Doe"
```

## Naming Convention

Dataset names follow a **material-first** convention: `{material}_{descriptor}`.

- **Lowercase, underscores only** — no hyphens, spaces, or special characters
- **Material first** — chemical formula or common name (`srtio3`, `silicon`, `graphene`, `gold`)
- **Descriptor second** — morphology, orientation, or qualifier (`lamella`, `monolayer`, `110`, `nanoparticle`)
- **Lab suffix only when needed** — to disambiguate (`srtio3_lamella_ncem` vs `srtio3_lamella_oxford`)
- **No resolution, binning, or year** — those belong in the JSON metadata

## Technique Folders

| Folder | Data type | quantem.widget |
|--------|-----------|----------------|
| `4dstem/` | 4D-STEM diffraction | Show4DSTEM, Show4D |
| `hrtem/` | High-resolution TEM | Show2D, Mark2D |
| `eels/` | Electron energy loss | Show1D |
| `tomo/` | Tomography | Show3DVolume |
| `diffraction/` | Diffraction patterns | Show2D |
| `image/` | Virtual/derived images | Show2D, Mark2D |
| `complex/` | Ptychography | ShowComplex2D |
| `raw/` | Original instrument files | — |

## Versioning

Follows [PEP 440](https://packaging.python.org/en/latest/discussions/versioning/) with semantic versioning.

Pre-release progression: `0.0.2a1` → `0.0.2a2` → `0.0.2b1` → `0.0.2rc1` → `0.0.2` (stable).

Each tag is immutable — never force-push or re-tag. Install pre-releases with `--pre`.

## Contributing

We welcome contributions of real electron microscopy data. Use `preview_upload()` to validate, then `upload()` to create a PR on HF Hub.

Requirements:
- Data must be shareable under an open license (CC-BY-4.0 recommended)
- Include instrument and sample information in the metadata JSON
- Follow the naming convention above
- All uploads go through PR review
