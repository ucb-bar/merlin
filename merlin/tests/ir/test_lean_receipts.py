"""Receipt publication must fail closed, including when an earlier attempt succeeded."""

import json
import subprocess

import pytest

from merlin.common.paths import repo_root
from merlin.verify.lean import check

FIXTURE = repo_root() / "merlin/tests/fixtures/lean_dce_forward.mlir"


def _stage(path):
    path.mkdir()
    for name in ("before", "after"):
        (path / f"{name}.mlir").write_bytes(FIXTURE.read_bytes())
    (path / "meta.json").write_text('{"boundary": "identity"}')
    return path


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


def test_checker_consumes_exact_round_trip_exports(tmp_path, monkeypatch, lean_tools):
    stage = _stage(tmp_path / "stage")
    exported, validated = [], []
    export, round_trip = check.export_file, check.round_trip_exported

    def record_export(source, target, **kwargs):
        exported.append(target)
        return export(source, target, **kwargs)

    def record_round_trip(source, target, *args, **kwargs):
        validated.append(target)
        return round_trip(source, target, *args, **kwargs)

    monkeypatch.setattr(check, "export_file", record_export)
    monkeypatch.setattr(check, "round_trip_exported", record_round_trip)
    receipt = check.check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "verified", receipt.get("reason")
    assert exported == validated and len(exported) == 2
    assert {str(p.resolve()) for p in validated} == {ref["path"] for ref in receipt["exports"].values()}
    assert all(check._sha(p) == receipt["exports"][p.stem]["sha256"] for p in validated)


def test_concurrent_source_edit_cannot_get_green_receipt(tmp_path, monkeypatch, lean_tools):
    stage = _stage(tmp_path / "stage")
    export = check.export_file

    def concurrent_edit(source, target, **kwargs):
        (stage / "after.mlir").write_text("changed during verification")
        return export(source, target, **kwargs)

    monkeypatch.setattr(check, "export_file", concurrent_edit)
    result = check.check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "unmeasured"
    assert "changed or missing" in result["reason"]
