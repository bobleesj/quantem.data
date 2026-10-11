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

`load` uses `quantem.widget` for images and `quantem.gpu.io.load` for native
4D-STEM acquisitions. Install the corresponding optional package. CUDA/MPS
acquisitions retain encoded storage and native detector geometry; read bounded
regions from each returned acquisition and close it when finished. Master-file
folders return a list of independent acquisitions.
Want only the file path? Use `download(name)`.

## Reuse inputs across tutorials

One input dataset can serve several algorithms and viewers. Shared acquisitions
stay under `4dstem/` and `haadf/`, synthetic denoising inputs under `denoise/`,
and viewer input collections under `widget/`. Tutorial code belongs in its software
repository; Hugging Face holds data, descriptive metadata and checksums only.

Use a qualified dataset name and an immutable revision in tutorial input
specifications. Unique flat names remain supported for interactive use:

```python
from quantem.data import download

folder = download(
    "denoise/simulated_au_nanobeam_pg_dose1",
    revision="475260be22b5a00704f58382517a44aff94e252e",
)
master = folder / "truth-events-dose-1.0_master.h5"
```

`download` returns the existing dataset folder with its companion files.
Pass `master` to `quantem.gpu.io.load` for accelerated acquisition loading,
then use the loaded data with the algorithm or widget. TV and SHINE reuse the
same synthetic expectation; noisy observations and training results are made
locally. No viewer-specific copy or upload is needed. Use the same `revision`
with `read_meta` to retrieve matching metadata. This API is a development
change; use this checkout until a release includes it.

`widget/ducky` is a synthetic reconstruction monitor with snapshot companions;
`widget/gold_session` is an image session. These collections are available at
the verified R1 revision:

```python
folder = download(
    "widget/ducky",
    revision="b30a28c295494b51139415d540819c4cbfde49b6",
)
```

Open them through the matching `quantem.widget.datasets` tutorial helpers.
Existing public paths remain for older clients. The unchanged denoising input
above still uses R0; use R1 for the new `widget/` paths.

`download(..., cache_dir="./hf-cache")` keeps the Hub's revision-specific cache.
Repeated calls reuse cached files but may still check the network. Use
`force_download=True` to replace cached copies. `out=` exports the selected
dataset and metadata under their repository paths; use separate output
directories for different revisions. Transfer and file-presence checks do not
rehash cached scientific files or establish their calibration.

## Command line

```bash
quantem-data list                          # every shared dataset name
quantem-data tree                          # grouped by bucket, with sizes
quantem-data meta gold_512                 # calibration sidecar
quantem-data download gold_512 --out ./d   # pull one dataset
quantem-data list --revision 475260be22b5a00704f58382517a44aff94e252e
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
use this schema. Leave unknown provenance unknown; do not invent permissions,
dates or calibration to make validation pass. The checker and template live in
this source checkout, not in the installed runtime or Hugging Face data tree.

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
  - Jane Doe
  - Sangjoon Bob Lee
pi: Colin Ophus
sample: gold nanoparticles
date: '2026-06-25'
scan_sampling_A: 0.2
magnification_MX: 1.3
facility: ncem
```

Most of this is filled automatically when the file already carries it (a Velox `.emd` provides
voltage / semiangle / magnification / FOV / scan shape / date), so an interactive `upload` only
asks for what's missing. Set `QUANTEM_FACILITY=ncem` to skip the facility prompt too. `--meta`
accepts YAML or JSON.

Or from Python:

```python
from quantem.data import upload

upload("scan.emd", name="gold_run", folder="haadf", meta={"voltage_kV": 300})
```

Calibration travels with the data. A Velox `.emd` auto-fills voltage / semiangle /
magnification / FOV / scan shape; an Arina folder auto-fills detector and scan shape. Explicit
`meta=` always wins.

The repo defaults to `bobleesj/quantem-data` (override with `QUANTEM_DATA_REPO` or `repo=`).
