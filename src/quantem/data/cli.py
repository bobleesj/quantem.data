"""``quantem-data`` command line: share 4D-STEM / HAADF datasets through the project
Hugging Face repo without writing any ``huggingface_hub`` code.

    quantem-data list                          # names of every shared dataset
    quantem-data status                        # repo summary (datasets, file counts, sizes)
    quantem-data meta gold_512                 # print a dataset's calibration sidecar
    quantem-data download gold_512 --out ./d   # pull one dataset by flat name
    quantem-data template 4dstem > cal.yaml    # an example sidecar to fill in, not start blank
    quantem-data upload ./gold_512             # asks for calibration, then uploads

`list` / `status` / `meta` / `download` are read-only and need no token (the repo is
public). `upload` writes, so it needs an HF token (``huggingface-cli login`` or ``HF_TOKEN``).
Run interactively, `upload` asks for the calibration the raw file cannot store (with an
example for each field) so the user never has to hand-write a sidecar; pass ``--meta cal.yaml``
to script it, or ``--no-input`` to skip the prompts entirely. Metadata is human-readable YAML.
Every command is a thin wrapper over the ``quantem.data`` verbs so the CLI and API never drift."""
import argparse
import os
import sys
from pathlib import Path

from quantem.data import download, list_datasets, read_meta, status, tree, upload
from quantem.data.huggingface import auto_meta, parse_sidecar

def _names(raw: str) -> list[str]:
    """Comma-separated people -> a list, so a sidecar stores ``operators: [Jane Doe, Bob Lee]``
    rather than one run-together string. A single name yields a one-element list."""
    return [part.strip() for part in raw.split(",") if part.strip()]


# The calibration a user is asked for at an interactive upload, per modality: (key, example,
# cast, required). Auto-derived fields (modality, scan_shape/det_shape/shape, dtype, and what a
# Velox .emd already carries: voltage/semiangle/magnification/FOV/date) are never prompted.
# Roughly mirrors the canonical sidecar schema in quantem.widget.io.meta (kept in sync by hand
# so quantem.data stays a standalone, widget-free transfer layer; `operators`/`pi` replace the
# vague `source` here first - the widget schema is a follow-up).
_PROMPT_FIELDS: dict[str, list[tuple[str, object, object, bool]]] = {
    "4dstem": [
        ("voltage_kV", 300, float, True),
        ("semiangle_mrad", 25, float, True),
        ("operators", ["Jane Doe", "Sangjoon Bob Lee"], _names, True),
        ("pi", "Colin Ophus", str, True),
        ("sample", "gold nanoparticles", str, True),
        ("date", "2026-06-25", str, True),
        ("scan_sampling_A", 0.2, float, False),
        ("magnification_MX", 1.3, float, False),
        ("facility", "ncem", str, False),
    ],
    "haadf": [
        ("voltage_kV", 300, float, True),
        ("operators", ["Jane Doe", "Sangjoon Bob Lee"], _names, True),
        ("pi", "Colin Ophus", str, True),
        ("sample", "gold nanoparticles", str, True),
        ("date", "2026-06-25", str, True),
        ("pixel_size_nm", 0.01, float, False),
        ("facility", "ncem", str, False),
    ],
}


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

    p_tmpl = sub.add_parser("template", help="Print an example calibration sidecar to fill in and pass to --meta.")
    p_tmpl.add_argument("modality", nargs="?", default="4dstem", choices=sorted(_PROMPT_FIELDS),
                        help="Which schema (default: 4dstem).")

    p_dl = sub.add_parser("download", help="Download one dataset by flat name.")
    p_dl.add_argument("name", help="Dataset name (flat, e.g. gold_512).")
    p_dl.add_argument("--out", default=None, help="Directory to download into (default: HF cache).")

    p_up = sub.add_parser("upload", help="Upload a file or folder (needs an HF write token).")
    p_up.add_argument("path", help="Local file or folder to upload.")
    p_up.add_argument("--name", default=None, help="Dataset name (default: the file/folder stem).")
    p_up.add_argument("--folder", default=None, help="Bucket: 4dstem or haadf (default: auto by content).")
    p_up.add_argument("--meta", default=None, help="Path to a YAML or JSON calibration sidecar (skips the prompts).")
    p_up.add_argument("--no-input", action="store_true", help="Don't prompt for calibration (scripts/CI).")

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
    """Route a parsed command to the matching ``quantem.data`` verb."""
    if args.command == "list":
        names = list_datasets(repo=args.repo)
        print("\n".join(names) if names else "(no datasets)")
        return 0
    if args.command == "tree":
        tree(repo=args.repo)
        return 0
    if args.command == "status":
        s = status(repo=args.repo)
        print(f"repo {s['repo']}  |  logged in as {s['logged_in_as']}  |  {s['total_mb']:.0f} MB total")
        for d in s["datasets"]:
            print(f"  {d['size_mb']:>10.1f} MB  {d['files']:>3} files  {d['name']}")
        return 0
    if args.command == "meta":
        meta = read_meta(args.name, repo=args.repo)
        if not meta:
            print(f"{args.name!r}: no metadata")
        else:
            import yaml  # noqa: PLC0415
            print(yaml.safe_dump(meta, sort_keys=False).rstrip())
        return 0
    if args.command == "template":
        import yaml  # noqa: PLC0415
        print(yaml.safe_dump(_template_meta(args.modality), sort_keys=False).rstrip())
        return 0
    if args.command == "download":
        path = download(args.name, repo=args.repo, out=args.out)
        print(path)
        return 0
    if args.command == "upload":
        return _upload(args)
    return 0


def _upload(args: argparse.Namespace) -> int:
    """Upload a dataset, building its calibration sidecar from ``--meta``, an interactive
    prompt, or nothing - then confirm before the (token-gated, repo-mutating) write."""
    src = Path(args.path)
    name = args.name or (src.stem if src.is_file() else src.name)
    folder = args.folder or ("4dstem" if src.is_dir() else "haadf")
    if args.meta:
        meta = parse_sidecar(args.meta)  # .yaml or .json, by extension
    elif args.no_input or not sys.stdin.isatty():
        meta = None  # scripted / piped: attach nothing, don't block on a prompt
    else:
        from_file = auto_meta(args.path)  # voltage/semiangle/date/... a Velox .emd already carries
        env = {"facility": os.environ["QUANTEM_FACILITY"]} if os.environ.get("QUANTEM_FACILITY") else {}
        known = {**from_file, **env}  # neither needs asking
        prompted = _prompt_meta(name, folder, known)
        # upload() re-derives only from_file; env defaults must be carried into the sidecar here
        meta = {**env, **prompted}
        print(f"\nUploading '{name}' to {folder}/ with: {({**from_file, **meta}) or '(no calibration)'}")
        if input("Proceed? [y/N]: ").strip().lower() not in ("y", "yes"):
            print("aborted.")
            return 1
    url = upload(args.path, name, folder=folder, repo=args.repo, meta=meta)
    print(url)
    return 0


def _template_meta(modality: str) -> dict:
    """An example sidecar for a modality: every field with a plausible value, so a user edits
    a filled file instead of starting blank. Built from the same field table as the prompts,
    so the two never drift. Auto-derived fields (shapes, dtype) are left out - they come from
    the data file at upload."""
    return {"modality": modality, **{key: example for key, example, _, _ in _PROMPT_FIELDS[modality]}}


def _prompt_meta(name: str, folder: str, known: dict | None = None) -> dict:
    """Ask the operator only for the calibration not already KNOWN, one field at a time with an
    example and a required/optional tag; Enter skips a field (present = known). Fields in
    ``known`` (auto-parsed from a Velox .emd / Arina master, or a ``QUANTEM_FACILITY`` default)
    are shown, not asked again - that is what makes most uploads just-confirm. Missing required
    fields are warned about, not blocked."""
    known = known or {}
    fields = _PROMPT_FIELDS.get(folder) or _PROMPT_FIELDS["haadf"]
    print(f"Describe '{name}' ({folder}). Press Enter to skip any field.")
    if known:
        print(f"  already known (not asked): {known}")
    meta = {key: _ask(key, example, cast, required)
            for key, example, cast, required in fields if key not in known}
    missing = [key for key, _, _, required in fields
               if required and key not in known and meta.get(key) is None]
    if missing:
        print(f"  warning: missing recommended fields {missing} - uploading without them.")
    return {key: value for key, value in meta.items() if value is not None}


def _ask(key: str, example: object, cast, required: bool):
    """Prompt for one field, re-asking until the value casts cleanly or is left blank."""
    tag = "required" if required else "optional"
    shown = ", ".join(example) if isinstance(example, list) else example  # a list example reads as names
    while True:
        raw = input(f"  {key} (e.g. {shown}) [{tag}]: ").strip()
        if not raw:
            return None
        try:
            return cast(raw)
        except ValueError:
            print(f"    '{raw}' is not valid for {key}; try again or press Enter to skip.")


if __name__ == "__main__":
    sys.exit(main())
