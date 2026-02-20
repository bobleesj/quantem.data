"""Command-line interface for quantem.data."""

import argparse
import json
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        prog="quantem-data",
        description="Manage quantem electron microscopy datasets on Hugging Face Hub",
    )
    sub = parser.add_subparsers(dest="command")

    # list
    p_list = sub.add_parser("list", help="List available datasets")
    p_list.add_argument("--technique", help="Filter by technique")

    # info
    p_info = sub.add_parser("info", help="Show dataset metadata")
    p_info.add_argument("name", help="Dataset name")

    # files
    p_files = sub.add_parser("files", help="List all files on HF Hub")
    p_files.add_argument("--technique", help="Filter by technique")

    # upload
    p_upload = sub.add_parser("upload", help="Upload a dataset")
    p_upload.add_argument("file", help="Path to .npy file")
    p_upload.add_argument("--name", required=True, help="Dataset name")
    p_upload.add_argument("--technique", required=True, help="Technique folder")
    p_upload.add_argument("--description", default="", help="Description")
    p_upload.add_argument("--contributor", default="", help="Contributor name")
    p_upload.add_argument("--metadata", help="Path to metadata JSON file")
    p_upload.add_argument("--token", help="HF token")

    # download
    p_download = sub.add_parser("download", help="Download a dataset")
    p_download.add_argument("name", help="Dataset name")
    p_download.add_argument("-o", "--output", help="Output path (default: {name}.npy)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    from quantem.data import available, info, load, upload, list_files

    if args.command == "list":
        names = available(technique=args.technique)
        if not names:
            print("No datasets found.")
        for name in names:
            meta = info(name)
            desc = meta.get("description", "")
            shape = meta.get("data", {}).get("shape", "")
            print(f"  {name:40s}  {str(shape):25s}  {desc}")

    elif args.command == "info":
        meta = info(args.name)
        meta.pop("_npy_path", None)
        print(json.dumps(meta, indent=2))

    elif args.command == "files":
        files = list_files(technique=args.technique)
        for f in files:
            print(f"  {f['path']:50s}  {f['size_mb']:8.2f} MB  [{f['type']}]")

    elif args.command == "upload":
        meta = args.metadata
        if meta:
            with open(meta) as f:
                meta = json.load(f)
        upload(
            args.file,
            name=args.name,
            technique=args.technique,
            metadata=meta,
            description=args.description,
            contributor=args.contributor,
            token=args.token,
        )

    elif args.command == "download":
        data = load(args.name)
        output = args.output or f"{args.name}.npy"
        np.save(output, data)
        print(f"Saved {output} ({data.nbytes / (1024**2):.1f} MB)")


if __name__ == "__main__":
    main()
