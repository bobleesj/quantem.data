# Uploading datasets

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bobleesj/quantem.data/blob/main/notebooks/upload.ipynb)

All datasets in quantem.data are hosted on [Hugging Face Hub](https://huggingface.co/datasets/bobleesj/quantem-data).
Uploads create a **Pull Request** by default — the data is reviewed before merging into the public catalog.

## Prerequisites

1. Create a free [Hugging Face account](https://huggingface.co/join)
2. Create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (needs write access)
3. Log in from your terminal:

```bash
huggingface-cli login
```

## Install

```bash
pip install --pre -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ quantem-data
```

## Naming convention

Dataset names follow a **material-first** convention: `{material}_{descriptor}`.

| Rule | Example | Bad example |
|------|---------|-------------|
| Lowercase, underscores only | `srtio3_lamella` | `SrTiO3-Lamella` |
| Material first | `gold_nanoparticle` | `nanoparticle_gold` |
| Descriptor second (morphology, orientation) | `silicon_110` | `110_silicon` |
| Lab suffix only to disambiguate | `srtio3_lamella_ncem` | `ncem_srtio3` |
| No resolution, binning, or year in name | `graphene_monolayer` | `graphene_256x256_2024` |

Resolution, binning, year, and instrument details go in the JSON metadata — not the name.

## Upload from Python

```python
import numpy as np
from quantem.data import upload

# Your data (NumPy array)
data = np.load("my_hrtem_image.npy")

# Upload — creates a PR on Hugging Face Hub
upload(
    data,
    name="silicon_110_hrtem",
    technique="hrtem",
    description="Silicon [110] zone axis, HRTEM at 200 kV",
    contributor="Jane Doe",
)
```

Output:

```
Created PR to add silicon_110_hrtem (0.2 MB)
Review: https://huggingface.co/datasets/bobleesj/quantem-data/discussions/1
```

The PR link takes you to the Hugging Face discussion page where the maintainer can review your data and metadata, then merge it.

## Preview before uploading

Use `preview_upload()` to validate naming, metadata, and check for duplicates before submitting:

```python
from quantem.data import preview_upload

errors = preview_upload(
    data,
    name="silicon_110_hrtem",
    technique="hrtem",
    description="Silicon [110] zone axis, HRTEM at 200 kV",
    contributor="Jane Doe",
)

if errors:
    for e in errors:
        print(f"  - {e}")
else:
    print("Ready to upload!")
```

`preview_upload()` checks:
- Naming convention (lowercase, underscores, material-first)
- Valid technique folder
- Metadata schema compliance
- Array shape consistency
- Duplicate name detection on HF Hub

Fix any errors before calling `upload()`.

## Upload from the command line

```bash
quantem-data upload my_data.npy \
    --name silicon_110_hrtem \
    --technique hrtem \
    --description "Silicon [110] zone axis" \
    --contributor "Jane Doe"
```

By default this creates a PR. Add `--direct` to commit directly (requires write access to the repo).

## Valid techniques

Each dataset belongs to a technique folder:

| technique | data type | widget |
|-----------|-----------|--------|
| `4dstem` | 4D-STEM diffraction | Show4DSTEM, Show4D |
| `hrtem` | high-resolution TEM | Show2D, Mark2D |
| `eels` | electron energy loss | Show1D |
| `tomo` | tomography | Show3DVolume |
| `diffraction` | diffraction patterns | Show2D |
| `image` | virtual/derived images | Show2D, Mark2D |
| `complex` | ptychography | ShowComplex2D |

## Metadata schema

Every uploaded dataset gets a paired `.json` sidecar with metadata.

**Required fields:**

| field | description |
|-------|-------------|
| `schema_version` | current: `"1.0"` |
| `name` | must match the dataset name |
| `technique` | must be one of the valid techniques above |
| `description` | one-line human description |
| `data.shape` | must match the actual array shape |
| `data.dtype` | e.g. `"float32"` |
| `attribution.contributor` | who uploaded the data |
| `attribution.license` | must be open, e.g. `"CC-BY-4.0"` |

**Optional fields:**

| field | description |
|-------|-------------|
| `instrument.microscope` | e.g. `"JEOL JEM-2100F"` |
| `instrument.voltage_kv` | accelerating voltage |
| `instrument.detector` | e.g. `"Gatan OneView"` |
| `calibration.pixel_size` | with `pixel_size_unit` |
| `processing.source` | provenance info |
| `attribution.institution` | lab or university |
| `attribution.date` | upload date |
| `attribution.doi` | publication DOI if applicable |

## Custom metadata

By default, `upload()` generates a metadata template. For full control, pass your own metadata dict or JSON file:

```python
from quantem.data import upload, make_template

# Generate a template and customize it
meta = make_template(
    name="silicon_110_hrtem",
    technique="hrtem",
    shape=(512, 512),
    description="Silicon [110] zone axis, HRTEM at 200 kV",
    contributor="Jane Doe",
)
meta["instrument"] = {
    "microscope": "JEOL JEM-2100F",
    "voltage_kv": 200,
    "detector": "Gatan OneView",
}
meta["calibration"] = {
    "pixel_size": 0.15,
    "pixel_size_unit": "angstrom",
}

upload(data, name="silicon_110_hrtem", technique="hrtem", metadata=meta)
```

Or from a JSON file:

```python
upload(data, name="silicon_110_hrtem", technique="hrtem", metadata="metadata.json")
```

## Validate before uploading

```python
from quantem.data import validate, make_template

meta = make_template(name="test", technique="hrtem", shape=(256, 256))
errors = validate(meta)
if errors:
    for e in errors:
        print(f"  - {e}")
else:
    print("Valid!")
```

## Update existing metadata

To update metadata for a dataset that's already uploaded:

```python
from quantem.data import update_metadata

update_metadata("silicon_110_hrtem", {
    "calibration": {"pixel_size": 0.148, "pixel_size_unit": "angstrom"},
})
```

This also creates a PR by default.

## Direct commits (maintainers only)

If you have write access to the repo, you can skip the PR:

```python
upload(data, name="...", technique="...", create_pr=False)
update_metadata("...", {...}, create_pr=False)
```

```bash
quantem-data upload data.npy --name ... --technique ... --direct
```
