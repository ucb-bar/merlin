"""Merlin's dead-tensor pruning is checked against MerlinVerify's refinement theorems while lowering.

Real-tool tests need the built checker (``lake build`` in ``merlin/lean``), ``MERLIN_LEAN_SDK_BIN`` and
``MERLIN_COMPILER_PYTHON``; they skip without them, or fail when ``MERLIN_LEAN_REQUIRED`` is set.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from merlin.llvmlower import lower, passes_xdsl
from merlin.verify import gate_checks, pass_validation
from merlin.verify.gate_errors import GateRejected, GateUnsupported
from merlin.verify.lean import check
from merlin.verify.speculative_gate import GateFailure, SpeculativeGate

SDK_BIN = os.environ.get("MERLIN_LEAN_SDK_BIN")
COMPILER_PYTHON = os.environ.get("MERLIN_COMPILER_PYTHON")
CAPTURES = Path(os.environ.get("MERLIN_RECAPTURES", "/scratch/grammy/blast/merlin/out/artifacts/recaptures"))
T = "tensor<4xf32>"


@pytest.fixture
def tools():
    missing = [
        name
        for name, ok in (
            ("MERLIN_LEAN_SDK_BIN", SDK_BIN and (Path(SDK_BIN) / "lake").is_file()),
            ("MERLIN_COMPILER_PYTHON", COMPILER_PYTHON and Path(COMPILER_PYTHON).is_file()),
            ("built merlin-lean", check.lean_exe().is_file()),
        )
        if not ok
    ]
    if missing:
        reason = "Lean pass-validation tools unavailable: " + ", ".join(missing)
        if os.environ.get("MERLIN_LEAN_REQUIRED"):
            pytest.fail(reason)
        pytest.skip(reason)
    return pass_validation.CheckerTools(Path(SDK_BIN), Path(COMPILER_PYTHON))


@pytest.fixture
def validation(tmp_path, tools):
    gate = SpeculativeGate(tmp_path / "gate", interrupt=True, job_timeout=900)
    yield pass_validation.PassValidation(gate, tools)
    gate.close()


def capture(name: str) -> str:
    path = CAPTURES / name / "model.mlir"
    if not path.is_file():
        if os.environ.get("MERLIN_LEAN_REQUIRED"):
            pytest.fail(f"captured model {name} is not present")
        pytest.skip(f"captured model {name} is not present")
    return path.read_text()


def job(tmp_path, before: str, after: str, tools) -> dict:
    directory = tmp_path / "job"
    directory.mkdir()
    (directory / "before.mlir").write_text(before)
    (directory / "after.mlir").write_text(after)
    return {
        "directory": str(directory),
        "label": "prune_dead_pure_tensor_ops",
        "before": str(directory / "before.mlir"),
        "after": str(directory / "after.mlir"),
        "params": {"sdk_bin": str(tools.sdk_bin), "compiler_python": str(tools.compiler_python), "index_bits": 64},
    }


def test_real_lowering_proves_the_prune_pass(tmp_path, validation):
    result = lower.lower_model(
        capture("small_llama_int8_consistent"), tmp_path / "lower", targets=(), pass_validation=validation
    )
    assert result.ll_path.is_file() and result.stats["dead_tensor_ops_pruned"] > 0
    (verdict,) = validation.gate.verdicts
    assert (verdict["label"], verdict["verdict"]) == ("prune_dead_pure_tensor_ops", "accepted")
    assert verdict["detail"]["theorems"] == list(gate_checks.THEOREMS)
    assert set(verdict["detail"]["axioms"]) <= {"propext", "Classical.choice", "Quot.sound"}
    chain = json.loads(Path(verdict["detail"]["chains"]["forward"]).read_text())
    assert chain["refinement_verdict"] == "verified"


def test_an_injected_prune_bug_stops_real_lowering(tmp_path, validation, monkeypatch):
    from xdsl.dialects import arith
    from xdsl.rewriter import Rewriter

    real = passes_xdsl.prune_dead_pure_tensor_ops

    def buggy(module):
        count = real(module)
        for op in module.walk():
            if isinstance(op, arith.AddfOp):  # an optimizer bug: one live add becomes a subtract
                Rewriter.replace_op(op, arith.SubfOp(op.lhs, op.rhs))
                break
        return count

    monkeypatch.setattr(passes_xdsl, "prune_dead_pure_tensor_ops", buggy)
    with pytest.raises(GateFailure) as raised:
        lower.lower_model(
            capture("small_llama_int8_consistent"), tmp_path / "lower", targets=(), pass_validation=validation
        )
    report = raised.value.report
    assert report["label"] == "prune_dead_pure_tensor_ops"
    assert report["verdict"] == "not_established"
    assert "TypedModule.check_sound did not verify" in report["reason"]
    assert "arith.subf" in Path(report["after"]).read_text()


def test_a_prune_that_changes_nothing_submits_nothing(tmp_path, validation):
    lower.lower_model(capture("tiny_llama_fp32_consistent"), tmp_path / "lower", targets=(), pass_validation=validation)
    assert validation.gate.verdicts == []


DEAD = f"""func.func @forward(%a: {T}) -> {T} {{
  %dead = tensor.empty() : {T}
  %x = arith.addf %a, %a : {T}
  return %x : {T}
}}
"""
LIVE = f"""func.func @forward(%a: {T}) -> {T} {{
  %x = arith.addf %a, %a : {T}
  return %x : {T}
}}
"""


def test_a_correct_small_prune_is_accepted(tmp_path, tools):
    detail = gate_checks.tensor_refinement(job(tmp_path, DEAD, LIVE, tools))
    assert detail["functions"] == ["forward"]


def test_a_wrong_rewrite_is_rejected(tmp_path, tools):
    with pytest.raises(GateRejected, match="did not verify"):
        gate_checks.tensor_refinement(job(tmp_path, DEAD, LIVE.replace("arith.addf", "arith.mulf"), tools))


def test_ir_the_mlir_parser_rejects_is_unsupported(tmp_path, tools):
    forward_use = f"""func.func @forward(%a: {T}) -> {T} {{
  %y = arith.addf %x, %a : {T}
  %x = arith.addf %a, %a : {T}
  return %y : {T}
}}
"""
    with pytest.raises(GateUnsupported, match="exporter refused"):
        gate_checks.tensor_refinement(job(tmp_path, forward_use, forward_use + "\n", tools))


def test_missing_tools_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        pass_validation.CheckerTools(tmp_path / "no-sdk", tmp_path / "no-python")


def test_requesting_validation_without_an_sdk_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv(pass_validation.ENV_PASSES, "all")
    monkeypatch.delenv(pass_validation.ENV_SDK_BIN, raising=False)
    with pytest.raises(ValueError, match="SDK"):
        lower.lower_model(LIVE, tmp_path / "lower", targets=())


def test_an_unknown_pass_is_refused(tmp_path, tools):
    gate = SpeculativeGate(tmp_path / "gate")
    try:
        with pytest.raises(ValueError, match="subset"):
            pass_validation.PassValidation(gate, tools, frozenset({"canonicalize"}))
    finally:
        gate.close()


def test_merlin_compile_flags_request_validation(monkeypatch):
    from merlin.compile import command

    ap = command.parser(("int8",), ("none",), "none", ("k1",))
    args = ap.parse_args(["--workload", "w", "--validate-passes", "all", "--lean-sdk-bin", "/sdk/bin"])
    monkeypatch.delenv(pass_validation.ENV_PASSES, raising=False)
    monkeypatch.delenv(pass_validation.ENV_SDK_BIN, raising=False)
    command.apply_pass_validation(args)
    assert os.environ[pass_validation.ENV_PASSES] == "all"
    assert os.environ[pass_validation.ENV_SDK_BIN] == "/sdk/bin"


def test_a_checker_built_from_other_sources_is_refused(tmp_path):
    import sys

    sdk = tmp_path / "sdk"
    sdk.mkdir()
    lake = sdk / "lake"
    lake.write_text("#!/bin/sh\nexit 3\n")  # what `lake build --no-build` reports for a stale build
    lake.chmod(0o755)
    if not check.lean_exe().is_file():
        pytest.skip("the merlin-lean checker is not built")
    with pytest.raises(ValueError, match="out of date"):
        pass_validation.CheckerTools(sdk, Path(sys.executable))
