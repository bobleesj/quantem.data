"""Interactive `upload` prompting: turn an operator's typed answers into a calibration dict,
casting numbers, skipping blanks, and warning (not blocking) on missing required fields. No
network - we feed answers through a fake ``input``."""
import builtins
import sys
from types import SimpleNamespace

import pytest

from quantem.data import cli
from quantem.data.huggingface import DATA_BUCKETS


def _answers(monkeypatch, values):
    """Feed `values` to successive input() calls."""
    stream = iter(values)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(stream))


def test_prompt_builds_meta_casting_numbers_and_names_and_skipping_blanks(monkeypatch):
    # 4dstem order: voltage, semiangle, operators, pi, sample, date, scan_sampling, mag, facility;
    # one number for the scan sampling is refused and asked again, since the key holds [row, col]
    _answers(monkeypatch, [
        "300", "25", "first operator, second operator", "investigator", "gold", "2026-06-25",
        "0.2", "0.2, 0.25", "", "",
    ])
    meta = cli._prompt_meta("gold_512", "4dstem")
    assert meta == {
        "voltage_kV": 300.0,
        "semiangle_mrad": 25.0,
        "operators": ["first operator", "second operator"],  # comma-separated -> list
        "pi": "investigator",
        "sample": "gold",
        "date": "2026-06-25",
        "scan_sampling_A": [0.2, 0.25],
    }  # blank magnification + facility are simply absent (present = known)


def test_prompt_warns_but_keeps_going_when_required_is_skipped(monkeypatch, capsys):
    _answers(monkeypatch, ["", "25", "first operator", "investigator", "gold", "2026-06-25", "", "", ""])  # skip voltage
    meta = cli._prompt_meta("gold_512", "4dstem")
    assert "voltage_kV" not in meta
    assert "warning" in capsys.readouterr().out


def test_prompt_skips_fields_already_read_from_the_file(monkeypatch):
    # voltage + date auto-parsed from a Velox .emd, so the operator is asked only for the rest
    _answers(monkeypatch, ["first operator", "investigator", "gold", "", ""])  # operators, pi, sample, pixel, facility
    meta = cli._prompt_meta("gold", "haadf", known={"voltage_kV": 300.0, "date": "2023-11-14"})
    assert "voltage_kV" not in meta and "date" not in meta  # not asked; from the file
    assert meta["operators"] == ["first operator"]


def test_upload_uses_facility_from_env_without_asking(monkeypatch, tmp_path):
    monkeypatch.setenv("QUANTEM_FACILITY", "facility name")
    img = tmp_path / "gold.tif"
    img.write_bytes(b"x")
    captured = _capture_upload(monkeypatch)
    # haadf order with facility supplied by env: voltage, operators, pi, sample, date, pixel, confirm
    _answers(monkeypatch, ["300", "first operator", "investigator", "gold", "2026-06-25", "", "y"])
    assert cli.main(["upload", str(img)]) == 0
    assert captured["meta"]["facility"] == "facility name"  # came from the env, never prompted


def test_ask_reprompts_until_a_number_parses(monkeypatch):
    _answers(monkeypatch, ["not-a-number", "300"])
    assert cli._ask("voltage_kV", 300, float, required=True) == 300.0


def test_template_is_fillable_example_matching_the_prompts(capsys):
    import yaml
    assert cli.main(["template", "4dstem"]) == 0
    meta = yaml.safe_load(capsys.readouterr().out)  # valid YAML, ready for --meta
    assert meta["modality"] == "4dstem"
    assert meta["voltage_kV"] == 300                      # example value, not blank
    assert meta["operators"] == ["operator name", "second operator name"]  # placeholders, never people
    assert (meta["pi"], meta["facility"]) == ("principal investigator name", "facility name")
    assert meta["scan_sampling_A"] == [0.2, 0.2]          # the [row, col] pair, not one number
    assert set(meta) == {"modality"} | {k for k, *_ in cli._PROMPT_FIELDS["4dstem"]}


def test_upload_help_names_every_data_bucket(capsys):
    with pytest.raises(SystemExit):
        cli.main(["upload", "--help"])
    printed = capsys.readouterr().out
    assert all(bucket in printed for bucket in DATA_BUCKETS)


def _capture_upload(monkeypatch):
    """Stub the real HF upload so the full `upload` command can run with no network."""
    captured = {}
    def fake_upload(path, name=None, *, folder=None, repo=None, meta=None):
        captured.update(path=path, name=name, folder=folder, meta=meta)
        return "https://hf/commit"
    monkeypatch.setattr(cli, "upload", fake_upload)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))  # force the interactive path
    return captured


def test_upload_command_prompts_confirms_then_uploads(monkeypatch, tmp_path):
    img = tmp_path / "gold.tif"
    img.write_bytes(b"x")
    captured = _capture_upload(monkeypatch)
    # haadf order: voltage, operators, pi, sample, date, pixel_size_nm, facility, then the confirm
    _answers(monkeypatch, ["300", "first operator, second operator", "investigator", "gold", "2026-06-25", "", "", "y"])
    assert cli.main(["upload", str(img)]) == 0
    assert captured["folder"] == "haadf"      # auto from a single file
    assert captured["name"] == "gold"          # auto from the stem
    assert captured["meta"]["voltage_kV"] == 300.0
    assert captured["meta"]["operators"] == ["first operator", "second operator"]
    assert captured["meta"]["pi"] == "investigator"


def test_upload_command_aborts_on_no_confirmation(monkeypatch, tmp_path):
    img = tmp_path / "gold.tif"
    img.write_bytes(b"x")
    captured = _capture_upload(monkeypatch)
    _answers(monkeypatch, ["300", "first operator", "investigator", "gold", "2026-06-25", "", "", "n"])  # decline
    assert cli.main(["upload", str(img)]) == 1
    assert captured == {}  # upload never called


def test_pinned_download_and_metadata_commands(monkeypatch, tmp_path, capsys):
    calls = []

    def download(name, **kwargs):
        calls.append((name, kwargs))
        return tmp_path / "widget/ducky"

    monkeypatch.setattr(cli, "download", download)
    assert cli.main([
        "download", "widget/ducky", "--revision", "input-commit",
        "--cache-dir", str(tmp_path), "--force-download",
    ]) == 0
    assert calls == [("widget/ducky", {
        "repo": None, "out": None, "revision": "input-commit",
        "cache_dir": str(tmp_path), "force_download": True,
    })]
    monkeypatch.setattr(cli, "read_meta", lambda name, **kwargs: calls.append((name, kwargs)))
    assert cli.main(["meta", "widget/ducky", "--revision", "input-commit"]) == 0
    assert calls[-1] == ("widget/ducky", {"repo": None, "revision": "input-commit"})
    capsys.readouterr()
