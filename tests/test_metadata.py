"""Velox .emd calibration parsing + its wiring into upload's auto-metadata.

A collaborator downloading a HAADF gets no calibration unless upload extracts it from the
file. We build a tiny synthetic .emd matching Velox's layout (JSON blob under
``Data/Image/<g>/Metadata`` as a column of bytes) and assert both the raw parse and the
unit-normalized values upload would attach."""
import json

import h5py
import numpy as np
import pytest

from quantem.data.huggingface import _build_meta
from quantem.data.metadata import parse_velox_emd_metadata

VELOX_RAW = {
    "Optics": {
        "AccelerationVoltage": "300000",       # volts -> 300 kV
        "BeamConvergence": "0.025",            # radians -> 25 mrad
        "FullScanFieldOfView": {"x": "2e-08"},  # metres -> 20 nm
    },
    "Scan": {"ScanSize": {"width": "512", "height": "512"}, "DwellTime": "1e-05"},
    "CustomProperties": {"StemMagnification": {"value": "1300000"}},  # times -> 1.3 MX
    "Instrument": {"Manufacturer": "FEI", "InstrumentModel": "Titan"},
    "Acquisition": {"AcquisitionStartDatetime": {"DateTime": "1700000000"}},  # 2023-11-14 UTC
}


@pytest.fixture
def velox_emd(tmp_path):
    """Write VELOX_RAW into a minimal .emd the parser can read back."""
    path = tmp_path / "scan.emd"
    blob = json.dumps(VELOX_RAW).encode("utf-8")
    column = np.frombuffer(blob, dtype=np.uint8).reshape(-1, 1)  # parser reads Metadata[:, 0]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Data/Image/0/Metadata", data=column)
    return path


def test_parse_velox_normalizes_units(velox_emd):
    parsed = parse_velox_emd_metadata(velox_emd)
    assert parsed["beam_energy_kv"] == pytest.approx(300.0)
    assert parsed["convergence_semiangle_mrad"] == pytest.approx(25.0)
    assert parsed["full_scan_field_of_view_nm"] == pytest.approx(20.0)
    assert parsed["stem_magnification_x"] == pytest.approx(1_300_000.0)
    assert parsed["scan_size"] == {"width": 512, "height": 512}
    assert parsed["acquisition_date"] == "2023-11-14"  # unix timestamp -> ISO date


def test_upload_auto_meta_from_velox(velox_emd):
    meta = _build_meta(velox_emd, None)
    assert meta["voltage_kV"] == pytest.approx(300.0)
    assert meta["semiangle_mrad"] == pytest.approx(25.0)
    assert meta["magnification_MX"] == pytest.approx(1.3)
    assert meta["scan_fov_nm"] == pytest.approx(20.0)
    assert meta["scan_shape"] == [512, 512]
    assert meta["date"] == "2023-11-14"  # auto-dated from the acquisition timestamp


def test_operator_meta_overrides_auto(velox_emd):
    meta = _build_meta(velox_emd, {"voltage_kV": 200})  # operator knows better
    assert meta["voltage_kV"] == 200
    assert meta["semiangle_mrad"] == pytest.approx(25.0)  # untouched auto value stays
