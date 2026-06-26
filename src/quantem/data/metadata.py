"""Metadata parsing helpers for quantem-data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py


def _first_image_group(handle: h5py.File) -> str:
    image_root = handle["Data/Image"]
    return next(iter(image_root.keys()))


def _load_raw_metadata(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        image_group = _first_image_group(handle)
        dataset = handle[f"Data/Image/{image_group}/Metadata"]
        raw = bytes(dataset[:, 0].tolist()).rstrip(b"\x00")
    return json.loads(raw.decode("utf-8", errors="replace"))


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_velox_emd_metadata(path: str | Path) -> dict[str, Any]:
    """Parse normalized metadata from a Velox/EMD file."""
    path = Path(path)
    raw = _load_raw_metadata(path)

    optics = raw.get("Optics", {})
    scan = raw.get("Scan", {})
    custom = raw.get("CustomProperties", {})
    instrument = raw.get("Instrument", {})
    acquisition = raw.get("Acquisition", {})

    fov = optics.get("FullScanFieldOfView", {})
    scan_size = scan.get("ScanSize", {})
    magnification = custom.get("StemMagnification", {})

    result = {
        "path": str(path),
        "source_format": "emd",
        "metadata_source": "velox_emd",
        "instrument_manufacturer": str(instrument.get("Manufacturer", "")).lower(),
        "instrument_model": str(instrument.get("InstrumentModel", "")).lower(),
        "instrument_class": str(instrument.get("InstrumentClass", "")).lower(),
        "beam_energy_kv": None,
        "convergence_semiangle_mrad": None,
        "stem_magnification_x": _to_float(magnification.get("value")),
        "full_scan_field_of_view_nm": None,
        "acquisition_date": None,
        "scan_size": {
            "width": int(scan_size["width"]) if "width" in scan_size else None,
            "height": int(scan_size["height"]) if "height" in scan_size else None,
        },
        "dwell_time_s": _to_float(scan.get("DwellTime")),
        "frame_time_s": _to_float(scan.get("FrameTime")),
        "raw_metadata": raw,
    }

    accel = _to_float(optics.get("AccelerationVoltage"))
    if accel is not None:
        result["beam_energy_kv"] = accel / 1000.0

    convergence = _to_float(optics.get("BeamConvergence"))
    if convergence is not None:
        result["convergence_semiangle_mrad"] = convergence * 1000.0

    fov_x = _to_float(fov.get("x"))
    if fov_x is not None:
        result["full_scan_field_of_view_nm"] = fov_x * 1e9

    # Velox stores the acquisition instant as a unix timestamp; surface it as a plain ISO date
    # so an uploaded dataset is dated automatically (the operator never types it).
    timestamp = _to_float(acquisition.get("AcquisitionStartDatetime", {}).get("DateTime"))
    if timestamp is not None:
        from datetime import datetime, timezone  # noqa: PLC0415
        result["acquisition_date"] = datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat()

    return result

