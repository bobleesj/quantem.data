"""``quantem-data`` command line: share 4D-STEM / HAADF datasets through the project
Hugging Face repo without writing any ``huggingface_hub`` code.

    quantem-data list                          # names of every shared dataset
    quantem-data status                        # repo summary (datasets, file counts, sizes)
    quantem-data meta gold_512                 # print a dataset's calibration sidecar
    quantem-data download gold_512 --out ./d   # pull one dataset by flat name
    quantem-data upload scan.h5 --folder 4dstem --name gold_run --meta cal.json

`list` / `status` / `meta` / `download` are read-only and need no token (the repo is
public). `upload` writes, so it needs an HF token (``huggingface-cli login`` or ``HF_TOKEN``).
Every command is a thin wrapper over ``quantem.data.hub`` so the CLI and the Python API
never drift."""
import argparse
import json
import sys
from pathlib import Path

from quantem.data import hub


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``quantem-data`` console script. Parse args, dispatch to a
    subcommand, return a process exit code."""
    parser = argparse.ArgumentParser(
        prog="quantem-data",
        description="Share 4D-STEM / HAADF datasets through the project Hugging Face repo.",
    )
    parser.add_argument("--repo", default=None,
                        help="Override the dataset repo (else $QUANTEM_DATA_REPO, else the default).")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List every shared dataset name (flat, one per line).")
    sub.add_parser("tree", help="Datasets grouped by bucket (data type) with sizes - the structure at a glance.")
    sub.add_parser("status", help="Repo summary: datasets, file counts, sizes.")

    p_meta = sub.add_parser("meta", help="Print a dataset's calibration sidecar (or 'no metadata').")
    p_meta.add_argument("name", help="Dataset name (flat, e.g. gold_512).")

    p_dl = sub.add_parser("download", help="Download one dataset by flat name.")
    p_dl.add_argument("name", help="Dataset name (flat, e.g. gold_512).")
    p_dl.add_argument("--out", default=None, help="Directory to download into (default: HF cache).")

    p_up = sub.add_parser("upload", help="Upload a file or folder (needs an HF write token).")
    p_up.add_argument("path", help="Local file or folder to upload.")
    p_up.add_argument("--name", default=None, help="Dataset name (default: the file/folder stem).")
    p_up.add_argument("--folder", default=None, help="Bucket: 4dstem or haadf (default: auto by content).")
    p_up.add_argument("--meta", default=None, help="Path to a JSON calibration sidecar to attach.")

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        return _dispatch(args)
    except (FileNotFoundError, ValueError) as err:
        print(f"quantem-data: {err}", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    """Route a parsed command to the matching ``quantem.data.hub`` verb."""
    if args.command == "list":
        names = hub.list_datasets(repo=args.repo)
        print("\n".join(names) if names else "(no datasets)")
        return 0
    if args.command == "tree":
        hub.tree(repo=args.repo)
        return 0
    if args.command == "status":
        s = hub.status(repo=args.repo)
        print(f"repo {s['repo']}  |  logged in as {s['logged_in_as']}  |  {s['total_mb']:.0f} MB total")
        for d in s["datasets"]:
            print(f"  {d['size_mb']:>10.1f} MB  {d['files']:>3} files  {d['name']}")
        return 0
    if args.command == "meta":
        meta = hub.read_meta(args.name, repo=args.repo)
        print(json.dumps(meta, indent=2) if meta else f"{args.name!r}: no metadata")
        return 0
    if args.command == "download":
        path = hub.download(args.name, repo=args.repo, out=args.out)
        print(path)
        return 0
    if args.command == "upload":
        meta = json.loads(Path(args.meta).read_text()) if args.meta else None
        url = hub.upload(args.path, args.name, folder=args.folder, repo=args.repo, meta=meta)
        print(url)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
