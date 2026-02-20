# quantem.data

Real electron microscopy datasets hosted on [Hugging Face Hub](https://huggingface.co/datasets/bobleesj/quantem-data).
Works with [quantem.widget](https://bobleesj.github.io/quantem.widget/) out of the box.

## install

```bash
pip install quantem-data
```

## quick start

```python
from quantem.data import load
from quantem.widget import Show4DSTEM, Show2D

# 4D-STEM dataset (9 MB, downloads once, cached locally)
data = load("srtio3_lamella")
Show4DSTEM(data, scan_shape=(32, 32))

# bright-field image
Show2D(load("srtio3_bf"))

# load with metadata for calibrated display
data, meta = load("srtio3_lamella_hr", metadata=True)
Show4DSTEM(data, scan_shape=meta["data"]["scan_shape"])
```

## technique folders

| folder | data type | quantem.widget |
|--------|-----------|----------------|
| `4dstem/` | 4D-STEM diffraction | Show4DSTEM, Show4D |
| `hrtem/` | high-resolution TEM | Show2D, Mark2D |
| `eels/` | electron energy loss | Show1D |
| `tomo/` | tomography | Show3DVolume |
| `diffraction/` | diffraction patterns | Show2D |
| `image/` | virtual/derived images | Show2D, Mark2D |
| `complex/` | ptychography | ShowComplex2D |
| `raw/` | original instrument files | — |

```{toctree}
:maxdepth: 2
:hidden:

api/index
changelog
```
