"""Merlin's dead-code pruning is checked against MerlinVerify's theorems while lowering.

Real lowerings use representative kernel chunks of the retained model capsules (lean_gate_chunks),
so every op family the capsules contain goes through the real passes in seconds.

Real-tool tests need the built checker (``lake build`` in ``merlin/lean``), ``MERLIN_LEAN_SDK_BIN`` and
``MERLIN_COMPILER_PYTHON``; they skip without them, or fail when ``MERLIN_LEAN_REQUIRED`` is set.
"""

from __future__ import annotations

import os
from pathlib import Path

import lean_gate_chunks
import pytest

from merlin.llvmlower import lower, passes_xdsl
from merlin.verify import gate_checks, pass_validation
from merlin.verify.gate_errors import GateRejected
from merlin.verify.lean import check
from merlin.verify.speculative_gate import GateFailure, SpeculativeGate

SDK_BIN = os.environ.get("MERLIN_LEAN_SDK_BIN")
COMPILER_PYTHON = os.environ.get("MERLIN_COMPILER_PYTHON")


T = "tensor<4xf32>"


@pytest.fixture
def tools(pass_gate_tools):
    return pass_gate_tools


@pytest.fixture
def validation(tmp_path, tools):
    gate = SpeculativeGate(tmp_path / "gate", interrupt=True, job_timeout=900)
    yield pass_validation.PassValidation(gate, tools, frozenset({"prune_dead_pure_tensor_ops"}))
    gate.close()


def kernel(variant: str = "text", name: str = "SY_model_tiny_llama:normalization@1015") -> str:
    """A representative chunk: RMSNorm, Q/DQ and the projections around them in TinyLlama."""
    (chunk,) = [c for c in lean_gate_chunks.all_chunks() if c.name == name]
    return getattr(chunk, variant)


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


def test_missing_tools_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        pass_validation.CheckerTools(tmp_path / "no-sdk", tmp_path / "no-python")


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


def test_a_32_bit_index_width_is_refused(tmp_path, tools):
    # LEDGER S6: the Lean model and the pinned native pipeline disagree at index32.
    gate = SpeculativeGate(tmp_path / "gate")
    try:
        with pytest.raises(ValueError, match="index"):
            pass_validation.PassValidation(gate, tools, index_bits=32)
    finally:
        gate.close()


def test_a_failed_check_leaves_no_usable_artifact(tmp_path, tools, monkeypatch):
    from xdsl.dialects import arith
    from xdsl.rewriter import Rewriter

    real = passes_xdsl.prune_dead_pure_tensor_ops

    def buggy(module):
        count = real(module)
        for op in module.walk():
            if isinstance(op, arith.AddfOp):
                Rewriter.replace_op(op, arith.SubfOp(op.lhs, op.rhs))
                break
        return count

    monkeypatch.setattr(passes_xdsl, "prune_dead_pure_tensor_ops", buggy)
    gate = SpeculativeGate(tmp_path / "gate", interrupt=False, job_timeout=900)  # lowering always finishes first
    work = tmp_path / "lower"
    try:
        with pytest.raises(GateFailure):
            lower.lower_model(
                kernel("pruned"),
                work,
                targets=(),
                pass_validation=pass_validation.PassValidation(gate, tools, frozenset({"prune_dead_pure_tensor_ops"})),
            )
    finally:
        gate.close()
    assert not (work / "model.ll").exists()
    assert (work / "model.ll.rejected").is_file()


CHUNKS = [chunk.name for chunk in lean_gate_chunks.all_chunks()]
