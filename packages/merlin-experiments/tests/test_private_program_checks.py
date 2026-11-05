"""Only explicit semantic refusals may leave sibling static checks running."""

from pathlib import Path

import pytest
from merlin_experiments.phase1.feedback import private_compilation_inputs as compilation
from merlin_experiments.phase1.feedback import private_program_checks as checks

from merlin.compile.model_execution_inputs import file_sha256
from merlin.verify.host_compute_audit import HostComputeVeto


def _receipt(tmp_path):
    elf = tmp_path / "program.elf"
    elf.write_bytes(b"owned harmless ELF observation")
    return {"output": {"elf": str(elf), "elf_sha256": file_sha256(elf)}}


@pytest.mark.parametrize(
    "failure", [checks.ProgramCheckFailure("unproved span", check="span"), HostComputeVeto("host arithmetic")]
)
def test_typed_refusal_keeps_failed_observations_and_reopens_identity(tmp_path, monkeypatch, failure):
    receipt = _receipt(tmp_path)
    calls = []
    monkeypatch.setattr(compilation, "verify", lambda *a, **k: calls.append(("recipe", k)))

    def refuse(*a, **k):
        raise failure

    result = checks.verify(refuse, receipt, revalidate=lambda: calls.append("inputs"), program="encode")
    assert calls == ["inputs", ("recipe", {"required": True})]
    assert result["status"] == checks.FAILED and result["linked_device_groups"] == 0
    assert result["observed_elf"]["sha256"] == receipt["output"]["elf_sha256"]
    assert result["check_elapsed_s"] >= 0 and "no linked or numerical credit" in result["scope"]
    assert checks.progress_stage(result) == "program_checks_failed"
    with pytest.raises(checks.ProgramCheckFailure, match="encode"):
        checks.require_complete([result])


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("identity changed"),
        RuntimeError("unexpected"),
        MemoryError("memory"),
        KeyboardInterrupt(),
        SystemExit(1),
    ],
)
def test_untyped_integrity_unexpected_and_interrupt_failures_do_not_continue(tmp_path, failure):
    def refuse(*a, **k):
        raise failure

    with pytest.raises(type(failure)):
        checks.verify(refuse, _receipt(tmp_path), revalidate=lambda: pytest.fail("must not continue"), program="p")


@pytest.mark.parametrize("changed", ["inputs", "elf", "recipe"])
def test_semantic_failure_cannot_mask_changed_inputs_products_or_recipe(tmp_path, monkeypatch, changed):
    receipt = _receipt(tmp_path)

    def refuse(*a, **k):
        if changed == "elf":
            Path(receipt["output"]["elf"]).write_bytes(b"changed")
        raise checks.ProgramCheckFailure("unproved control", check="control")

    def reopen():
        if changed == "inputs":
            raise ValueError("selected input changed")

    def recipe(*a, **k):
        raise ValueError("recipe changed")

    monkeypatch.setattr(compilation, "verify", recipe)
    with pytest.raises(ValueError, match="changed"):
        checks.verify(refuse, receipt, revalidate=reopen, program="p")


def test_success_retains_real_checker_record_without_new_input_or_execution_credit(tmp_path):
    result = checks.verify(
        lambda *a, **k: {"program": "p", "status": checks.VERIFIED},
        _receipt(tmp_path),
        revalidate=lambda: pytest.fail("ordinary final caller owns this check"),
        program="p",
    )
    assert result["status"] == checks.VERIFIED and result["check_elapsed_s"] >= 0
    checks.require_complete([result])
    assert checks.progress_stage(result) == "program_receipt_verified"
