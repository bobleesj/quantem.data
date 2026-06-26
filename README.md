# quantem.data

Share 4D-STEM / HAADF datasets through one Hugging Face repo, with one-line download and load.

The datasets live here: <https://huggingface.co/datasets/bobleesj/quantem-data>

## Install

```bash
pip install quantem.data       # browse / download / load (no token needed)
```

Uploading also needs a Hugging Face write token (`huggingface-cli login` or `HF_TOKEN`).

## Use

```python
from quantem.data import browse, load, read_meta

browse()                        # thumbnail gallery in Jupyter, tree in a terminal
ds = load("gold_drift_0deg")    # download, then return the data ready to use
read_meta("gold_512")           # calibration: voltage, semiangle, FOV, ...
```

`load` returns a `Dataset2d` for a single image, or the loaded 4D data for an acquisition.
Want only the file path? Use `download(name)`.

## Command line

```bash
quantem-data list                          # every shared dataset name
quantem-data tree                          # grouped by bucket, with sizes
quantem-data meta gold_512                 # calibration sidecar
quantem-data download gold_512 --out ./d   # pull one dataset
```

## Publish (owner)

From the terminal, just point at the data - `upload` asks for the calibration (with an
example for each field) and shows you the result before writing:

```bash
quantem-data upload ./gold_512        # prompts for voltage, semiangle, sample, date, ...
quantem-data upload scan.emd --meta cal.json   # or script it with a JSON sidecar
quantem-data upload scan.emd --no-input        # or attach nothing
```

Or from Python:

```python
from quantem.data import upload

upload("scan.emd", name="gold_run", folder="haadf", meta={"voltage_kV": 300})
```

Calibration travels with the data. A Velox `.emd` auto-fills voltage / semiangle /
magnification / FOV / scan shape; an Arina folder auto-fills detector and scan shape. Explicit
`meta=` always wins.

The repo defaults to `bobleesj/quantem-data` (override with `QUANTEM_DATA_REPO` or `repo=`).
