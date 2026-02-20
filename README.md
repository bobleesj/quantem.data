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
| `korean_sample_c1` | image | Virtual image from 4D-STEM focal series |

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

Optional: `instrument`, `calibration`, `processing`.

## Contributing

We welcome contributions of real electron microscopy data. See the upload API above, or contact us via [GitHub Issues](https://github.com/bobleesj/quantem.data/issues).

Requirements:
- Data must be shareable under an open license (CC-BY-4.0 recommended)
- Include instrument and sample information in the metadata JSON
- Follow the naming convention above
