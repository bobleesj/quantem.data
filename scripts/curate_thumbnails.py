"""Backfill thumbnail.png for every dataset in bobleesj/quantem-data.

Cheap: for each dataset, download only its smallest existing preview image (a 4D-STEM
Virtual_8bit.tif, or the HAADF image itself) - never the multi-GB data - downscale to a
256 px PNG, and upload it as <bucket>/<name>/thumbnail.png. Adds files only; never deletes
or overwrites data. Needs an HF write token (owner action)."""
import io as _io
import sys
import numpy as np
from PIL import Image
from quantem.data import hub
from quantem.widget import io as wio

THUMB = 256
hf = hub._hub()
repo = hub.DEFAULT_REPO
# _hub() sets HF_HUB_DISABLE_IMPLICIT_TOKEN=1 (quiet public downloads), which stops the
# stored token from being attached automatically - so pass it EXPLICITLY for uploads.
TOKEN = hf.get_token()
if not TOKEN:
    raise SystemExit("No HF token found. Run `huggingface-cli login` with a WRITE token first.")
all_files = hf.list_repo_files(repo_id=repo, repo_type="dataset")


def preview_source(full_name: str) -> str | None:
    """Pick the cheapest file in a dataset folder to render a thumbnail from."""
    # match folder datasets (<name>/...) AND single-file datasets (<name>.emd/.tif),
    # but not a sibling whose name is a prefix (gold_haadf must not grab gold_haadf_npy).
    here = [f for f in all_files
            if (f.startswith(full_name + "/") or f.startswith(full_name + "."))
            and not f.endswith("thumbnail.png")]
    # 1) an 8-bit virtual image (tiny, already display-ready) for 4D-STEM
    for f in here:
        if "virtual00_8bit" in f.lower() and f.lower().endswith((".tif", ".tiff")):
            return f
    # 2) any virtual tif
    for f in here:
        if "virtual" in f.lower() and f.lower().endswith((".tif", ".tiff")):
            return f
    # 3) a plain image (HAADF .tif / .png), else an .emd / .npy we can load
    for exts in ((".tif", ".tiff", ".png"), (".emd", ".npy")):
        cands = [f for f in here if f.lower().endswith(exts)]
        if cands:
            return min(cands, key=len)
    return None


def make_thumb(array: np.ndarray) -> bytes:
    """2D array -> 256 px 8-bit PNG bytes, contrast-stretched (1-99 percentile)."""
    arr = np.asarray(array, dtype=np.float32)
    while arr.ndim > 2:  # collapse any extra dims to a 2D image
        arr = arr.mean(axis=0)
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img.thumbnail((THUMB, THUMB))
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> int:
    datasets = hub.list_datasets()  # data buckets only (notebooks filtered)
    print(f"{len(datasets)} datasets to thumbnail\n")
    done, skipped = [], []
    import os
    import tempfile
    for full in datasets:
        try:
            src = preview_source(full)
            if src is None:
                print(f"  SKIP {full}: no preview source found")
                skipped.append(full)
                continue
            local = hf.hf_hub_download(repo_id=repo, repo_type="dataset", filename=src)
            # .npy may be a raw 4D array (read_image only handles 2D) - load it directly and
            # let make_thumb collapse extra dims; everything else goes through read_image.
            arr = np.load(local) if src.lower().endswith(".npy") else wio.read_image(local).array
            png = make_thumb(arr)
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            with open(tmp, "wb") as fh:
                fh.write(png)
            hf.upload_file(path_or_fileobj=tmp, path_in_repo=f"{full}/thumbnail.png",
                           repo_id=repo, repo_type="dataset", token=TOKEN)
            os.unlink(tmp)
            print(f"  OK   {full}  <- {src.split('/')[-1]}  ({len(png)//1024} KB)")
            done.append(full)
        except Exception as err:
            print(f"  FAIL {full}: {type(err).__name__}: {str(err)[:70]}")
            skipped.append(full)
    print(f"\nthumbnails uploaded: {len(done)} | skipped: {len(skipped)} {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
