"""Receipt publication must fail closed, including when an earlier attempt succeeded."""

import json
import subprocess

import pytest

from merlin.verify.lean import check


@pytest.mark.parametrize("error", [FileNotFoundError("checker missing"), RuntimeError("audit failed"),
                                 ValueError("malformed checker JSON")])
def test_failed_rerun_replaces_green_receipt(tmp_path, monkeypatch, error):
    check._write(tmp_path, {"verdict": "verified"})

    def fail(*args, **kwargs):
        assert json.loads((tmp_path / "receipt.json").read_text())["verdict"] == "unmeasured"
        raise error

    monkeypatch.setattr(check, "_check_stage", fail)
    result = check.check_stage(tmp_path, m2m_python=tmp_path / "missing")
    assert result["verdict"] == "unmeasured"
    assert str(error) in result["reason"]
    assert json.loads((tmp_path / "receipt.json").read_text()) == result


def test_interrupted_attempt_cannot_leave_old_proof(tmp_path, monkeypatch):
    check._write(tmp_path, {"verdict": "verified"})

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(check, "_check_support", interrupt)
    with pytest.raises(KeyboardInterrupt):
        check.check_support(tmp_path / "program", "f", tmp_path, m2m_python=tmp_path / "python")
    assert json.loads((tmp_path / "receipt.json").read_text())["verdict"] == "unmeasured"


def test_invalid_width_invalidates_previous_receipt(tmp_path):
    check._write(tmp_path, {"verdict": "verified"})
    with pytest.raises(ValueError):
        check.check_support(tmp_path / "program", "f", tmp_path, m2m_python=tmp_path / "python", index_bits=-1)
    assert json.loads((tmp_path / "receipt.json").read_text())["verdict"] == "unmeasured"


@pytest.mark.parametrize("code,verdict", [(1, "verified"), (0, "abstracted"), (2, "verified")])
def test_checker_exit_must_match_verdict(code, verdict):
    with pytest.raises(ValueError, match="exit status"):
        check._check_exit(subprocess.CompletedProcess([], code), {"verdict": verdict})
