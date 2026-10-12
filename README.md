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
read_meta("gold_512_npy_bin4")  # calibration: voltage, sampling, units, ...
```

`load` uses `quantem.widget` for images and `quantem.gpu.io.load` for native
4D-STEM acquisitions. Install the corresponding optional package. CUDA/MPS
acquisitions retain encoded storage and native detector geometry; read bounded
regions from each returned acquisition and close it when finished. On CUDA and
MPS a master-file folder returns a list of independent acquisitions; with
`backend="cpu"` the loader stacks them into one dataset unless `stack=False`
is passed.
Want only the file path? Use `download(name)`.

## Reuse inputs across tutorials

One input dataset can serve several algorithms and viewers. The current layout
has four data folders: shared acquisitions under `4dstem/` and `haadf/`,
synthetic denoising inputs under `denoise/`, and viewer input collections under
`widget/`. Revision `5925e4b49b2ca28ff82d6d5a6b9a8bfea9edfb20` holds this
layout, and quantem.widget pins it for its tutorial inputs; pin the same one
in a new tutorial. Tutorial code belongs in its software repository; Hugging
Face holds data, data cards, descriptive metadata and checksums only.

Use a qualified dataset name and an immutable revision in tutorial input
specifications. Unique flat names remain supported for interactive use:

```python
from quantem.data import download

DATA_REVISION = "5925e4b49b2ca28ff82d6d5a6b9a8bfea9edfb20"
folder = download("denoise/simulated_au_nanobeam_pg_dose1", revision=DATA_REVISION)
master = folder / "truth-events-dose-1.0_master.h5"
```

`download` returns the existing dataset folder with its companion files.
Pass `master` to `quantem.gpu.io.load` for accelerated acquisition loading,
then use the loaded data with the algorithm or widget. Several tutorials can
reuse this synthetic expectation; noisy observations and training results are
made locally. No viewer-specific copy or upload is needed. Use the same
`revision` with `read_meta` to retrieve matching metadata. Every listing takes
it too: `list_datasets`, `datasets`, `tree`, `browse` and `status`, and
`--revision` on `quantem-data list`, `tree` and `status`. The calls that
`datasets` and `browse` print then carry that revision.

Qualified names, `revision`, `cache_dir` and `force_download` are newer than
the 0.0.1rc4 release on TestPyPI, which also has no `load`. Until a release
includes them, install from this repository:

```bash
python -m pip install git+https://github.com/bobleesj/quantem.data.git
```

`widget/ducky` is a synthetic reconstruction monitor with snapshot companions;
`widget/gold_session` is an image session. The same revision contains both:

```python
folder = download("widget/ducky", revision=DATA_REVISION)
```

`load` does not open these collections; it raises a `ValueError` that names
`quantem.widget.datasets`.

A revision fixes every path. Do not combine files from two revisions in one
input.

`download(..., cache_dir="./hf-cache")` keeps the Hub's revision-specific cache.
Repeated calls reuse cached files but may still check the network. Use
`force_download=True` to replace cached copies. `load` takes the same
`revision`, `cache_dir` and `force_download` and passes every other keyword to
the reader; a keyword the reader does not take raises `TypeError`. `out=`
exports the selected dataset and metadata under their repository paths; use
separate output directories for different revisions. Transfer and
file-presence checks do not rehash cached scientific files or establish their
calibration.

### Older layout

Two older revisions hold an earlier layout that still has `survey/` and
`widget-tutorials/` folders, which the current revision does not have:

- `475260be22b5a00704f58382517a44aff94e252e` has no `widget/` folder.
- `b30a28c295494b51139415d540819c4cbfde49b6` has `widget/` next to the two
  older folders. The saved outputs of `notebooks/shared_datasets_tutorial.ipynb`
  were produced at this revision.

quantem.widget 0.0.1rc41 requests `widget-tutorials/` paths; the `widget/`
paths need a later quantem.widget. `quantem.data` lists and downloads the four
current folders at any revision; a `survey/` or `widget-tutorials/` path is
not a dataset to it.

## Command line

```bash
quantem-data list                          # every shared dataset name
quantem-data tree                          # grouped by bucket, with sizes
quantem-data meta gold_512_npy_bin4        # calibration sidecar
quantem-data download gold_512 --out ./d   # pull one dataset
quantem-data list --revision 5925e4b49b2ca28ff82d6d5a6b9a8bfea9edfb20   # or: tree / status
```

## Publish (owner)

The source-owned contribution checker preserves the provenance, permission,
shape, sampling and units checks formerly hosted with the public data:

```bash
cp scripts/templates/dataset-meta.json /path/to/new-dataset/meta.json
# Replace examples with documented acquisition details and permission.
python scripts/check_dataset_metadata.py /path/to/new-dataset
```

This is an opt-in check for new contributions, separate from the calibration
sidecars read at runtime. It does not retroactively require legacy datasets to
use this schema. A `scan_sampling_A`, when given, must be the `[row, col]` pair
in angstrom; one number is refused. Leave unknown provenance unknown; do not
invent permissions, dates or calibration to make validation pass. The checker
and template live in this source checkout, not in the installed runtime or
Hugging Face data tree.

From the terminal, just point at the data - `upload` asks for the calibration (with an
example for each field) and shows you the result before writing:

```bash
quantem-data upload ./gold_512        # prompts for voltage, semiangle, sample, date, ...
quantem-data upload scan.emd --meta cal.json   # or script it with a JSON sidecar
quantem-data upload scan.emd --no-input        # or attach nothing
```

### Metadata template

Metadata is human-readable YAML. Don't start from a blank file - `template` prints a filled
example you edit, then pass to `--meta`:

```bash
quantem-data template 4dstem > cal.yaml   # or: template haadf
```

```yaml
modality: 4dstem
voltage_kV: 300
semiangle_mrad: 25
operators:
  - operator name
  - second operator name
pi: principal investigator name
sample: gold nanoparticles
date: '2026-06-25'
scan_sampling_A:  # [row, col], angstrom
  - 0.2
  - 0.2
magnification_MX: 1.3
facility: facility name
```

Most of this is filled automatically when the file already carries it (a Velox `.emd` provides
voltage / semiangle / magnification / FOV / scan shape / date), so an interactive `upload` only
asks for what's missing. Set `QUANTEM_FACILITY` to your facility name to skip the facility
prompt too. `--meta` accepts YAML or JSON.

Or from Python:

```python
from quantem.data import upload

upload("scan.emd", name="gold_run", folder="haadf", meta={"voltage_kV": 300})
```

Calibration travels with the data. A Velox `.emd` auto-fills voltage / semiangle /
magnification / FOV / scan shape; an Arina folder auto-fills detector and scan shape. Explicit
`meta=` always wins.

The repo defaults to `bobleesj/quantem-data` (override with `QUANTEM_DATA_REPO` or `repo=`).
