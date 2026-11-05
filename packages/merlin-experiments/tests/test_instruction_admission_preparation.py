"""Neutral transport controls; synthetic owners/tools issue no target qualification."""

import contextlib
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from merlin_experiments.phase1.feedback import instruction_admission_preparation as A


def _evaluate(*, elf, evidence_root):
    evidence_root.mkdir()
    report = evidence_root / "report.json"
    data = {"status": "accepted", "elf_sha256": hashlib.sha256(elf.read_bytes()).hexdigest()}
    report.write_text(json.dumps(data))
    return {**data, "report_path": str(report), "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest()}


@pytest.fixture
def selected(tmp_path, monkeypatch):
    source = tmp_path / "reviewed.json"
    source.write_text(json.dumps({"forbidden_roots": [str(tmp_path / "excluded")]}))

    def read(path, *, target):
        return A.D.Declaration(path, path.read_bytes(), (), ())

    def verify(self):
        if self.path.read_bytes() != self.raw:
            raise ValueError("original declaration changed")

    monkeypatch.setattr(A.D, "read", read)
    monkeypatch.setattr(A.D.Declaration, "verify", verify)
    monkeypatch.setattr(A, "shared_compute_pool", lambda **kwargs: contextlib.nullcontext())
    return A.InstructionAdmissionSelection(
        "neutral", source, hashlib.sha256(source.read_bytes()).hexdigest(), tmp_path / "out", tmp_path
    )


def test_check_plan_never_calls_issuer_or_creates_output(selected, monkeypatch):
    monkeypatch.setattr(A.C, "prepare", lambda *a, **k: pytest.fail("check mode issued a service"))
    result = A.check_plan(selected)
    assert result["status"] == "CHECKED_NOT_PREPARED" and result["qualification"] is False
    assert result["service_issued"] is False and not selected.output.exists()


@pytest.mark.parametrize("change", ["hash", "existing", "outside", "excluded", "symlink", "budget"])
def test_plan_refuses_foreign_or_unbounded_selection(selected, change):
    if change == "hash":
        selected = replace(selected, declaration_sha256="0" * 64)
    elif change == "existing":
        selected.output.mkdir()
    elif change == "outside":
        selected = replace(selected, output=selected.task_root.parent / "outside")
    elif change == "excluded":
        selected = replace(selected, output=selected.task_root / "excluded" / "output")
    elif change == "symlink":
        link = selected.task_root / "link"
        link.symlink_to(selected.declaration)
        selected = replace(selected, declaration=link)
    else:
        selected = replace(selected, lease_timeout_s=float("nan"))
    with pytest.raises(ValueError):
        A.check_plan(selected)


def test_saved_report_cannot_be_a_selection_or_live_owner(selected, monkeypatch):
    with pytest.raises(ValueError, match="typed selection"):
        A.check_plan({"status": "accepted"})
    monkeypatch.setattr(A.C, "prepare", lambda *a, **k: {"status": "accepted"})
    with pytest.raises(ValueError, match="live service"):
        A.prepare_live(selected)


def _neutral_original_owner(selected, monkeypatch):
    calls = []
    service = A.LinkedElfAdmissionService(
        "neutral",
        _evaluate,
        ((str(Path(__file__).resolve()), hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),),
    )
    owner = object.__new__(A.C.PreparedInstructionSelection)
    object.__setattr__(owner, "service", service)
    monkeypatch.setattr(A.C.PreparedInstructionSelection, "verify", lambda self: calls.append("verify"))
    monkeypatch.setattr(A.C.PreparedInstructionSelection, "record", lambda self: {"scope": "synthetic transport"})

    def original(declaration, *, target, output):
        assert type(declaration) is A.D.Declaration and target == "neutral"
        calls.append("original_prepare")
        output.mkdir()
        return owner

    monkeypatch.setattr(A.C, "prepare", original)
    return owner, calls


def test_prepare_returns_same_original_owner_and_rechecks(selected, monkeypatch):
    owner, calls = _neutral_original_owner(selected, monkeypatch)
    assert A.prepare_live(selected) is owner
    assert calls == ["original_prepare", "verify"]


def test_current_original_owner_failure_propagates(selected, monkeypatch):
    _neutral_original_owner(selected, monkeypatch)

    def fail(self):
        raise ValueError("actual owner source changed")

    monkeypatch.setattr(A.C.PreparedInstructionSelection, "verify", fail)
    with pytest.raises(ValueError, match="source changed"):
        A.prepare_live(selected)


def _argv(selected):
    return [
        "--target",
        selected.target,
        "--selection",
        str(selected.declaration),
        "--selection-sha256",
        selected.declaration_sha256,
        "--output",
        str(selected.output),
        "--task-root",
        str(selected.task_root),
    ]


def test_cli_default_checks_only_and_missing_input_is_precise(tmp_path, capsys):
    args = [
        "--target",
        "neutral",
        "--selection",
        str(tmp_path / "missing.json"),
        "--selection-sha256",
        "0" * 64,
        "--output",
        str(tmp_path / "out"),
        "--task-root",
        str(tmp_path),
    ]
    assert A.main(args) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "REFUSED" and result["qualification"] is False
    assert not (tmp_path / "out").exists()


def test_cli_real_service_transport_static_only(selected, monkeypatch, capsys):
    _neutral_original_owner(selected, monkeypatch)
    elf = selected.task_root / "existing.elf"
    elf.write_bytes(b"neutral transport fixture")
    assert A.main([*_argv(selected), "--prepare", "--elf", str(elf)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "PREPARED" and result["linked_elf"]["status"] == "accepted"
    assert result["qualification"] is False and result["target_execution"] == "NOT_RUN"
    assert result["timing"] == "UNKNOWN"
    assert (selected.output / "operator_preparation.json").is_file()


def test_static_use_requires_explicit_prepare(selected, monkeypatch, capsys):
    monkeypatch.setattr(A.C, "prepare", lambda *a, **k: pytest.fail("implicit issuer"))
    assert A.main([*_argv(selected), "--elf", str(selected.task_root / "any.elf")]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "REFUSED"


def test_missing_elf_refuses_before_expensive_issuers(selected, monkeypatch, capsys):
    monkeypatch.setattr(A.C, "prepare", lambda *a, **k: pytest.fail("missing ELF issued a service"))
    assert A.main([*_argv(selected), "--prepare", "--elf", str(selected.task_root / "missing.elf")]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "REFUSED"
    assert not selected.output.exists()


def test_existing_cli_dispatches_fixed_operator_module(monkeypatch):
    from merlin_experiments import cli

    seen = []
    monkeypatch.setattr(A, "main", lambda args: seen.append(args) or 2)
    assert cli.main(["instruction-admission", "--target", "neutral"]) == 2
    assert seen == [["--target", "neutral"]]
