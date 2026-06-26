"""Backfill thumbnails for every dataset in bobleesj/quantem-data.

Cheap: for each dataset, render a small preview from a file it ALREADY has - a 4D-STEM
Virtual_8bit.tif, a binned .npy (virtual bright-field = mean over the detector), or the
HAADF image itself - never the multi-GB raw data. Uploads to ``.thumbnails/<bucket>/
<name>.png``, a top-level tree OUTSIDE the data buckets so a thumbnail never collides with
a dataset's own file (an earlier ``<bucket>/<name>/thumbnail.png`` made single-file
datasets ambiguous in ``download``). Also deletes that old in-folder thumbnail if present.
Adds/removes only thumbnails; never touches data. Needs an HF write token (owner action)."""
import io as _io
import os
import sys
import tempfile

import numpy as np
from PIL import Image

from quantem.data import huggingface
from quantem.widget import io as wio

THUMB = 256
hf = huggingface._hf()
repo = huggingface.DEFAULT_REPO
TOKEN = hf.get_token()
if not TOKEN:
    raise SystemExit("No HF token found. Run `huggingface-cli login` with a WRITE token first.")
all_files = hf.list_repo_files(repo_id=repo, repo_type="dataset")


def preview_source(full_name: str) -> str | None:
    """Cheapest existing file to render a thumbnail from. Falls back to a binned-npy sibling
    (e.g. gold_512 -> gold_512_npy_bin4) when a dataset is raw-h5 only."""
    here = [f for f in all_files
            if (f.startswith(full_name + "/") or f.startswith(full_name + "."))
            and "/thumbnail" not in f and not f.endswith(".png")]
    for f in here:  # a ready-made 8-bit virtual image is ideal
        if "virtual00_8bit" in f.lower() and f.lower().endswith((".tif", ".tiff")):
            return f
    for f in here:
        if "virtual" in f.lower() and f.lower().endswith((".tif", ".tiff")):
            return f
    for exts in ((".tif", ".tiff"), (".emd",), (".npy",)):
        cands = [f for f in here if f.lower().endswith(exts)]
        if cands:
            return min(cands, key=len)
    bucket, short = full_name.split("/", 1)  # raw-only: borrow a binned sibling
    sib = [f for f in all_files if f.startswith(f"{bucket}/{short}_npy_bin") and f.endswith(".npy")]
    return min(sib, key=len) if sib else None


def thumb_png(local: str, is_npy: bool) -> bytes:
    """Render a 256 px 8-bit PNG. 4D arrays become a virtual bright-field (mean over the
    detector -> a real-space scan image); 2D images are used directly. Contrast-stretched."""
    arr = np.load(local, mmap_mode="r") if is_npy else np.asarray(wio.read_image(local).array)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr.mean(axis=(-1, -2))  # virtual BF: average the detector -> (scan_y, scan_x)
    while arr.ndim > 2:
        arr = arr.mean(axis=0)
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img.thumbnail((THUMB, THUMB))
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> int:
    datasets = huggingface.list_datasets()
    print(f"{len(datasets)} datasets\n")
    done, skipped = [], []
    for full in datasets:
        try:
            src = preview_source(full)
            if src is None:
                print(f"  SKIP {full}: no preview source")
                skipped.append(full)
                continue
            local = hf.hf_hub_download(repo_id=repo, repo_type="dataset", filename=src)
            png = thumb_png(local, src.lower().endswith(".npy"))
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            with open(tmp, "wb") as fh:
                fh.write(png)
            hf.upload_file(path_or_fileobj=tmp, path_in_repo=f"{huggingface.THUMB_DIR}/{full}.png",
                           repo_id=repo, repo_type="dataset", token=TOKEN)
            os.unlink(tmp)
            # remove the old in-folder thumbnail that caused the download collision
            old = f"{full}/thumbnail.png"
            if old in all_files:
                hf.delete_file(path_in_repo=old, repo_id=repo, repo_type="dataset", token=TOKEN)
            print(f"  OK   {full}  <- {src.split('/')[-1]}  ({len(png) // 1024} KB)")
            done.append(full)
        except Exception as err:
            print(f"  FAIL {full}: {type(err).__name__}: {str(err)[:70]}")
            skipped.append(full)
    print(f"\nthumbnails: {len(done)} ok | {len(skipped)} skipped {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
