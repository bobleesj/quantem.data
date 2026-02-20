# quantem.data

Real electron microscopy datasets hosted on [Hugging Face Hub](https://huggingface.co/datasets/bobleesj/quantem-data).
Works with [quantem.widget](https://bobleesj.github.io/quantem.widget/) out of the box.

## install

```bash
pip install --pre -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ quantem-data
```

## quick start

```python
from quantem.data import load
from quantem.widget import Show2D

# load a 2D image (downloads once, cached locally)
Show2D(load("korean_sample_c1"))

# load with metadata
data, meta = load("korean_sample_c1", metadata=True)
Show2D(data, title=meta["description"])
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

guides/index
api/index
changelog
```
