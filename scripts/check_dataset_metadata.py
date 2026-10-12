"""Check documented provenance for a new dataset contribution.

Ported from bobleesj/quantem-data/check_meta at
475260be22b5a00704f58382517a44aff94e252e. This developer-only checker
preserves the original rules; it is not a universal schema for legacy inputs.
Never fabricate missing permission, date or calibration values to pass it.

Refuse a folder whose meta.json is missing required keys.

Required: name, source, date, sample, shape, dtype, permission
  (permission.received, permission.from, permission.collectors,
  permission.date). from and collectors are one or more names.
Optional: sampling, units, and optics keys. scan_sampling_A, when given, is
the [row, col] pair in angstrom; one number is refused.
"""

import argparse
import json
import re
import sys
from pathlib import Path

REQUIRED = ("name", "source", "date", "sample", "shape", "dtype", "permission")
PERMISSION_REQUIRED = ("received", "from", "collectors", "date")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _name_list(value: object) -> list[str] | None:
    """Accept one name or a list of names. Empty is missing."""

    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if (
        isinstance(value, list)
        and value
        and all(isinstance(item, str) and item.strip() for item in value)
    ):
        return [item.strip() for item in value]
    return None


def check_meta(folder: Path) -> list[str]:
    """Check the contribution sidecar without modifying it.

    Parameters
    ----------
    folder : pathlib.Path
        New dataset folder containing ``meta.json``.

    Returns
    -------
    list[str]
        Missing or malformed fields. An empty list means the sidecar meets
        this contribution schema, not that its provenance was independently
        verified or that its calibration is measured.

    Examples
    --------
    >>> check_meta(Path("new-acquisition"))
    ['missing new-acquisition/meta.json']
    """

    problems: list[str] = []
    path = folder / "meta.json"
    if not path.is_file():
        return [f"missing {path}"]
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{path} is not JSON: {exc}"]
    if not isinstance(meta, dict):
        return [f"{path} must be a JSON object"]
    for key in REQUIRED:
        value = meta.get(key)
        if value in (None, "", []):
            problems.append(f"missing {key}")
    if "name" in meta and not isinstance(meta.get("name"), str):
        problems.append("name must be a string")
    if "source" in meta and not isinstance(meta.get("source"), str):
        problems.append("source must be a string")
    if "sample" in meta and not isinstance(meta.get("sample"), str):
        problems.append("sample must be a string")
    permission = meta.get("permission")
    if permission not in (None, "", []):
        if not isinstance(permission, dict):
            problems.append("permission must be an object")
        else:
            for key in PERMISSION_REQUIRED:
                value = permission.get(key)
                if value in (None, "", []):
                    problems.append(f"missing permission.{key}")
            if permission.get("received") is not True:
                problems.append("permission.received must be true")
            if (
                permission.get("from") not in (None, "", [])
                and _name_list(permission.get("from")) is None
            ):
                problems.append("permission.from must be one name or a list of names")
            if (
                permission.get("collectors") not in (None, "", [])
                and _name_list(permission.get("collectors")) is None
            ):
                problems.append(
                    "permission.collectors must be one name or a list of names"
                )
            perm_date = permission.get("date")
            if perm_date not in (None, "") and not (
                isinstance(perm_date, str) and DATE_RE.match(perm_date)
            ):
                problems.append("permission.date must be YYYY-MM-DD")
    date = meta.get("date")
    if date not in (None, "", []) and not (
        isinstance(date, str) and DATE_RE.match(date)
    ):
        problems.append("date must be YYYY-MM-DD")
    shape = meta.get("shape")
    if shape not in (None, "", []) and not (
        isinstance(shape, list)
        and len(shape) >= 2
        and all(isinstance(n, int) and not isinstance(n, bool) for n in shape)
    ):
        problems.append("shape must be a list of ints, length >= 2")
    dtype = meta.get("dtype")
    if dtype not in (None, "", []) and not isinstance(dtype, str):
        problems.append("dtype must be a string")
    sampling = meta.get("sampling")
    if sampling not in (None, "", []):
        if not (
            isinstance(sampling, list)
            and all(
                isinstance(n, (int, float)) and not isinstance(n, bool)
                for n in sampling
            )
        ):
            problems.append("sampling must be a list of numbers")
        if (
            isinstance(shape, list)
            and isinstance(sampling, list)
            and len(sampling) != len(shape)
        ):
            problems.append("sampling length must match shape length")
    units = meta.get("units")
    if units not in (None, "", []):
        if not (isinstance(units, list) and all(isinstance(u, str) for u in units)):
            problems.append("units must be a list of strings")
        if (
            isinstance(shape, list)
            and isinstance(units, list)
            and len(units) != len(shape)
        ):
            problems.append("units length must match shape length")
    scan_sampling = meta.get("scan_sampling_A")
    if scan_sampling is not None and not (
        isinstance(scan_sampling, list)
        and len(scan_sampling) == 2
        and all(
            isinstance(n, (int, float)) and not isinstance(n, bool)
            for n in scan_sampling
        )
    ):
        problems.append("scan_sampling_A must be [row, col], two numbers in angstrom")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Report contribution metadata problems and return a shell exit code."""
    parser = argparse.ArgumentParser(
        description="Validate a new dataset contribution's meta.json; never rewrite metadata.",
    )
    parser.add_argument("folder", type=Path, help="folder that contains meta.json")
    args = parser.parse_args(argv)
    folder: Path = args.folder
    if not folder.is_dir():
        print(f"not a folder: {folder}", file=sys.stderr)
        return 2
    problems = check_meta(folder)
    if problems:
        print(f"{folder / 'meta.json'}:")
        for item in problems:
            print(f"  {item}")
        return 1
    print(f"{folder / 'meta.json'}: ok ({', '.join(REQUIRED)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
