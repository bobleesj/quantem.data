# quantem.data

Real electron microscopy datasets for [quantem.widget](https://github.com/bobleesj/quantem.widget).

Data hosted on [Hugging Face Hub](https://huggingface.co/datasets/bobleesj/quantem-data). Works on Google Colab out of the box.

## Install

```bash
pip install quantem-data
```

## Usage with quantem.widget

```python
from quantem.data import load
from quantem.widget import Show4DSTEM, Show2D

# 4D-STEM dataset (9 MB, downloads once, cached locally)
data = load("srtio3_lamella")
Show4DSTEM(data, scan_shape=(32, 32))

# Bright-field image
Show2D(load("srtio3_bf"))

# Load with metadata for calibrated display
data, meta = load("srtio3_lamella_hr", metadata=True)
Show4DSTEM(data, scan_shape=meta["data"]["scan_shape"])
```

## API

```python
from quantem.data import available, info, load, load_raw, list_files

# List datasets (optionally filter by technique)
available()                          # all datasets
available(technique="4dstem")        # only 4D-STEM

# Dataset metadata (no data download)
info("srtio3_lamella")

# Load processed data as NumPy array
data = load("srtio3_lamella")
data, meta = load("srtio3_lamella", metadata=True)

# Load original instrument files (returns local path)
path = load_raw("arina_lamella_master")  # -> ~/.cache/huggingface/.../*.h5

# List all files on HF Hub
list_files()
list_files("4dstem")
```

## CLI

```bash
# List datasets
quantem-data list
quantem-data list --technique 4dstem

# Show metadata
quantem-data info srtio3_lamella

# List files on HF Hub
quantem-data files

# Download
quantem-data download srtio3_lamella

# Upload
quantem-data upload my_data.npy --name silicon_110 --technique hrtem \
    --description "Silicon [110] HRTEM" --contributor "Jane Doe"
```

## Upload (Python API)

```python
from quantem.data import upload, update_metadata

# Upload with auto-generated metadata
upload(
    my_array,
    name="mos2_monolayer",
    technique="4dstem",
    description="Monolayer MoS2, 80 keV, Medipix3",
    contributor="Jane Doe",
)

# Upload with pre-built metadata JSON
upload(my_array, name="mos2_monolayer", technique="4dstem", metadata="mos2.json")

# Edit metadata for an existing dataset
update_metadata("mos2_monolayer", {
    "calibration": {"pixel_size": 0.98, "pixel_size_unit": "nm"},
})
```

## Naming Convention

Dataset names follow a **material-first** convention: `{material}_{descriptor}`.

- **Lowercase, underscores only** — no hyphens, spaces, or special characters
- **Material first** — chemical formula or common name (`srtio3`, `silicon`, `graphene`, `gold`)
- **Descriptor second** — morphology, orientation, or qualifier (`lamella`, `monolayer`, `110`, `nanoparticle`)
- **Lab suffix only when needed** — to disambiguate (`srtio3_lamella_ncem` vs `srtio3_lamella_oxford`)
- **No resolution, binning, or year** — those belong in the JSON metadata

| Name | Technique | What it is |
|------|-----------|------------|
| `srtio3_lamella` | 4dstem | STO FIB lamella |
| `mos2_monolayer` | 4dstem | Monolayer MoS2 |
| `silicon_110` | hrtem | Silicon [110] zone axis |
| `graphene_pristine` | hrtem | Pristine graphene |
| `silicon_k_edge` | eels | Silicon K-edge spectrum |
| `gold_nanoparticle` | tomo | Au nanoparticle tilt series |

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

## Metadata JSON Schema

Every dataset has a JSON sidecar with structured metadata. Required fields:

- `schema_version`, `name`, `technique`, `description`
- `data.shape`, `data.dtype`
- `attribution.contributor`, `attribution.license`

Optional: `instrument`, `calibration`, `processing` — see [CLAUDE.md](CLAUDE.md) for full spec.

## Contributing

We welcome contributions of real electron microscopy data. See the upload API above, or contact us via [GitHub Issues](https://github.com/bobleesj/quantem.data/issues).

Requirements:
- Data must be shareable under an open license (CC-BY-4.0 recommended)
- Include instrument and sample information in the metadata JSON
- Follow the naming convention above
