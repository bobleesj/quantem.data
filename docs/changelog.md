# changelog

## 0.0.2

- `DataBrowser` interactive Jupyter widget for browsing and loading datasets
- JS build infrastructure (esbuild + React + MUI + anywidget)
- `upload()` and `update_metadata()` create PRs by default (`create_pr=True`)
- atomic uploads: `.npy` + `.json` committed in a single HF Hub commit
- CLI: `--direct` flag for direct commits (skips PR)
- light/dark theme support in DataBrowser widget
- demo notebook with Google Colab badge

## 0.0.1

- initial release
- `load()`, `available()`, `info()`, `load_raw()`, `upload()`, `update_metadata()`, `list_files()`
- JSON metadata schema with validation
- CLI: `quantem-data list`, `info`, `files`, `download`, `upload`
- Hugging Face Hub backend with local caching
- technique folders: `4dstem`, `hrtem`, `eels`, `tomo`, `diffraction`, `image`, `complex`, `raw`
