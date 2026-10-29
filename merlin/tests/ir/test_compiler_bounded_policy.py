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


@pytest.mark.parametrize("kind", [transforms.UNROLL, transforms.TILE])
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


@pytest.mark.parametrize("kind", [transforms.UNROLL, transforms.TILE])
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


@pytest.mark.parametrize("pass_name", ["cse", "prune_dead_pure_tensor_ops", "lower_quant_ext"])
def test_source_rulebook_admission_uses_only_original_source(tmp_path, pass_gate_tools, pass_name):
    from types import SimpleNamespace

    from merlin.verify.compiler_ir_source_profile import admit_source
    from merlin.verify.compiler_profile import SourceOutsideProfile

    validation = SimpleNamespace(gate=SimpleNamespace(directory=tmp_path), tools=pass_gate_tools)
    supported = "builtin.module { func.func @forward(%x:i32) -> i32 { func.return %x : i32 } }"
    profile = admit_source(supported, pass_name=pass_name, validation=validation)
    assert isinstance(profile, str)
    unsupported = "builtin.module { func.func @forward(%x:memref<1xi32>) { func.return } }"
    with pytest.raises(SourceOutsideProfile, match="signature outside"):
        admit_source(unsupported, pass_name=pass_name, validation=validation)
    decisions = [json.loads(path.read_text()) for path in tmp_path.rglob("admission.json")]
    assert sorted(row["supported"] for row in decisions) == [False, True]
    assert all(row["boundary_proof"] is False and row["policy_approval"] is False for row in decisions)


@pytest.mark.parametrize("fmt", ["bf16", "f16"])
def test_draft_macc_source_admission_excludes_nonzero_initialization(tmp_path, pass_gate_tools, fmt):
    from types import SimpleNamespace

    from lean_bf16_f32acc_cases import bmm

    from merlin.verify.compiler_ir_source_profile import MACC, admit_source
    from merlin.verify.compiler_profile import SourceOutsideProfile

    validation = SimpleNamespace(gate=SimpleNamespace(directory=tmp_path), tools=pass_gate_tools)
    assert admit_source(bmm(fmt), pass_name=transforms.MACC, validation=validation) == MACC
    with pytest.raises(SourceOutsideProfile, match=r"not a \+0.0"):
        admit_source(bmm(fmt, init="1.0"), pass_name=transforms.MACC, validation=validation)


@pytest.mark.parametrize("textual", [False, True])
def test_per_channel_i64_dq_runs_ordinary_lowering_with_unverified_end_log(tmp_path, pass_gate_tools, caplog, textual):
    from test_lean_dequant_contract import PER_CHANNEL

    from merlin.llvmlower import lower

    source = PER_CHANNEL.replace("ZERO", "i64").replace("RANGE", "quant_min = -128 : i64, quant_max = 127 : i64")
    ordinary = lower.lower_model(source, tmp_path / "ordinary", targets=(), textual=textual)
    gate = SpeculativeGate(tmp_path / "gate", job_timeout=120)
    validation = pass_validation.PassValidation(gate, pass_gate_tools, frozenset({"lower_quant_ext"}))
    selected = lower.lower_model(source, tmp_path / "selected", targets=(), textual=textual, pass_validation=validation)
    assert selected.ll_path.read_bytes() == ordinary.ll_path.read_bytes()
    assert gate.verdicts == []
    coverage = json.loads((gate.directory / "coverage.json").read_text())
    assert coverage["unverified_boundaries"] == 1
    assert coverage["verified_boundaries"] == 0
    assert coverage["compiler_completed"] is True
    assert "Unverified compiler boundaries" in caplog.text


def test_multiple_selected_host_groups_run_normally_with_unverified_coverage(tmp_path, monkeypatch):
    from dataclasses import replace

    from host_graph_fusion_cases import contract, source

    from merlin.llvmlower.graph_optimizations import optimize_host_graph

    raw = source(context=True)
    options = replace(contract(raw, context=True), fusion_groups=((0, 1, 2, 3), (4, 5, 6, 7)))
    ordinary, _ = optimize_host_graph(raw, options)
    validation = validation_at(tmp_path, monkeypatch, ["host_graph_fusion"])
    with pass_validation.validating(validation):
        selected, _ = optimize_host_graph(raw, options)
    assert selected == ordinary
    assert not validation.gate.verdicts
    coverage = json.loads((validation.gate.directory / "coverage.json").read_text())
    assert coverage["unverified_boundaries"] == 1
    assert coverage["boundaries"][0]["code"] == "fusion-selection"


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
