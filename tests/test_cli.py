"""Interactive `upload` prompting: turn an operator's typed answers into a calibration dict,
casting numbers, skipping blanks, and warning (not blocking) on missing required fields. No
network - we feed answers through a fake ``input``."""
import builtins

from quantem.data import cli


def _answers(monkeypatch, values):
    """Feed `values` to successive input() calls."""
    stream = iter(values)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(stream))


def test_prompt_builds_meta_casting_numbers_and_skipping_blanks(monkeypatch):
    # 4dstem order: voltage, semiangle, source, sample, date, scan_sampling, magnification, facility
    _answers(monkeypatch, ["300", "25", "ncem", "gold", "2026-06-25", "0.2", "", ""])
    meta = cli._prompt_meta("gold_512", "4dstem")
    assert meta == {
        "voltage_kV": 300.0,
        "semiangle_mrad": 25.0,
        "source": "ncem",
        "sample": "gold",
        "date": "2026-06-25",
        "scan_sampling_A": 0.2,
    }  # blank magnification + facility are simply absent (present = known)


def test_prompt_warns_but_keeps_going_when_required_is_skipped(monkeypatch, capsys):
    _answers(monkeypatch, ["", "25", "ncem", "gold", "2026-06-25", "", "", ""])  # skip voltage
    meta = cli._prompt_meta("gold_512", "4dstem")
    assert "voltage_kV" not in meta
    assert "warning" in capsys.readouterr().out


def test_ask_reprompts_until_a_number_parses(monkeypatch):
    _answers(monkeypatch, ["not-a-number", "300"])
    assert cli._ask("voltage_kV", 300, float, required=True) == 300.0
