"""Ordinary out-of-model transforms and blocking admitted checks at real scopes."""

import json
import sys
from pathlib import Path

import pytest

from merlin.verify import compiler_transforms as transforms
from merlin.verify import pass_validation
from merlin.verify.speculative_gate import GateFailure, SpeculativeGate

SOURCE = """builtin.module {
func.func @forward(%x:i32, %unused:memref<1xi32>) -> i32 {
  %lb = arith.constant 3 : index
  %ub = arith.constant 31 : index
  %st = arith.constant 2 : index
  %r = scf.for %i = %lb to %ub step %st iter_args(%c = %x) -> (i32) {
    %a = arith.addi %c, %c : i32
    scf.yield %a : i32
  }
  func.return %r : i32
}
}"""


def validation_at(tmp_path, monkeypatch, passes):
    # Tool availability is intentionally disabled to test the compiler policy.
    # An admitted job still runs the actual worker and must refuse these tools.
    monkeypatch.setattr(pass_validation.CheckerTools, "__post_init__", lambda self: None)
    tools = pass_validation.CheckerTools(tmp_path / "missing-sdk", Path(sys.executable))
    gate = SpeculativeGate(tmp_path / "gate", job_timeout=20)
    return pass_validation.PassValidation(gate, tools, frozenset(passes))


@pytest.mark.parametrize("kind", ([transforms.UNROLL, transforms.TILE])[:1])
def test_unsupported_source_runs_ordinary_transform_and_logs_at_end(tmp_path, monkeypatch, caplog, kind):
    options = transforms.Options(unroll_factor=2) if kind == transforms.UNROLL else transforms.Options(tile_size=3)
    baseline, _ = transforms.apply(SOURCE, tmp_path / "ordinary", options=options)
    validation = validation_at(tmp_path, monkeypatch, [kind])
    with pass_validation.validating(validation):
        actual, report = transforms.apply(SOURCE, tmp_path / "selected", options=options)
        assert not caplog.records
    assert actual == baseline
    assert report["changed"] == [kind]
    assert not validation.gate.verdicts
    coverage = json.loads((validation.gate.directory / "coverage.json").read_text())
    assert coverage["compiler_completed"] is True
    assert coverage["verified_boundaries"] == 0
    assert coverage["unverified_boundaries"] == 1
    assert coverage["boundaries"][0]["proof_credit"] is False
    assert "Unverified compiler boundaries" in caplog.text


@pytest.mark.parametrize("kind", ([transforms.UNROLL, transforms.TILE])[:1])
def test_admitted_source_with_unavailable_real_worker_stops_compilation(tmp_path, monkeypatch, kind):
    source = SOURCE.replace(", %unused:memref<1xi32>", "")
    options = transforms.Options(unroll_factor=2) if kind == transforms.UNROLL else transforms.Options(tile_size=3)
    validation = validation_at(tmp_path, monkeypatch, [kind])
    with pytest.raises(GateFailure):
        with pass_validation.validating(validation):
            transforms.apply(source, tmp_path / "selected", options=options)
    coverage = json.loads((validation.gate.directory / "coverage.json").read_text())
    assert coverage["compiler_completed"] is False
    assert coverage["unverified_boundaries"] == 0
    assert coverage["verified_boundaries"] == 0


def test_unexpected_admission_error_blocks_and_quarantines_compiler_output(tmp_path, monkeypatch):
    controller = pass_validation
    validation = validation_at(tmp_path, monkeypatch, [transforms.UNROLL])
    product = tmp_path / "model.ll"
    quarantined = []

    def quarantine(work):
        quarantined.append(work)
        product.unlink()

    def unexpected(raw):
        raise RuntimeError("unexpected source projection failure")

    @pass_validation.validated("work", on_failure=quarantine)
    def compile_with_fault(work, *, pass_validation):
        product.write_text("compiler product before a later source admission")
        return controller.source_admission(transforms.UNROLL, SOURCE, predicate=unexpected)

    with pytest.raises(GateFailure, match="unexpected source projection failure") as failure:
        compile_with_fault(tmp_path, pass_validation=validation)
    assert isinstance(failure.value.__cause__, RuntimeError)
    assert not product.exists() and quarantined == [tmp_path]
    coverage = json.loads((validation.gate.directory / "coverage.json").read_text())
    assert coverage["compiler_completed"] is False
    assert coverage["unverified_boundaries"] == 0
