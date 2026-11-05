"""Merlin's dead-code pruning is checked against MerlinVerify's theorems while lowering.

Real lowerings use representative kernel chunks of the retained model capsules (lean_gate_chunks),
so every op family the capsules contain goes through the real passes in seconds.

Real-tool tests need the built checker (``lake build`` in ``merlin/lean``), ``MERLIN_LEAN_SDK_BIN`` and
``MERLIN_COMPILER_PYTHON``; they skip without them, or fail when ``MERLIN_LEAN_REQUIRED`` is set.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import lean_gate_chunks
import pytest

from merlin.llvmlower import lower, passes_xdsl
from merlin.verify import gate_checks, pass_validation
from merlin.verify.gate_errors import GateRejected, GateUnsupported
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


def test_real_lowering_proves_pruning(tmp_path, validation):
    result = lower.lower_model(kernel("pruned"), tmp_path / "lower", targets=(), pass_validation=validation)
    assert result.ll_path.is_file() and result.stats["dead_tensor_ops_pruned"] > 0
    assert result.stats["quant_ext_lowered"] > 0  # dead quantization is pruned too
    verdicts = {v["label"]: v for v in validation.gate.verdicts}
    assert {label: v["verdict"] for label, v in verdicts.items()} == {
        "prune_dead_pure_tensor_ops": "accepted",
    }
    verdict = verdicts["prune_dead_pure_tensor_ops"]
    assert verdict["detail"]["theorems"] == list(gate_checks.THEOREMS)
    assert set(verdict["detail"]["axioms"]) <= {"propext", "Classical.choice", "Quot.sound"}
    chain = json.loads(Path(verdict["detail"]["chains"]["forward"]).read_text())
    assert chain["refinement_verdict"] == "verified"
    # The claim says exactly which optional strengthenings were established.
    assert verdict["detail"]["claim"].startswith("strict refinement")
    assert set(verdict["detail"]["established"]) == {"universal_boundary", "definedness"} & set(
        k for k, v in verdict["detail"]["strengthenings"].items() if v == "verified"
    )


def test_all_registers_formal_gates_without_draft_numerical_policy(tmp_path, tools, monkeypatch):
    from merlin.verify import bf16_policy_gate as draft_policy

    def forbidden(*args, **kwargs):
        raise AssertionError("draft policy entered the production compiler gate")

    monkeypatch.setenv(pass_validation.ENV_PASSES, "all")
    monkeypatch.setenv(pass_validation.ENV_SDK_BIN, str(tools.sdk_bin))
    monkeypatch.setenv(pass_validation.ENV_COMPILER_PYTHON, str(tools.compiler_python))
    monkeypatch.setattr(draft_policy, "require_admission", forbidden)
    monkeypatch.setattr(draft_policy, "check_policy", forbidden)
    original = pass_validation.from_environment
    validations = []

    def capture(*args, **kwargs):
        selected = original(*args, **kwargs)
        validations.append(selected)
        return selected

    monkeypatch.setattr(pass_validation, "from_environment", capture)
    result = lower.lower_model(kernel("pruned"), tmp_path / "lower", targets=())
    assert result.ll_path.is_file()
    assert len(validations) == 1
    validation = validations[0]
    assert validation.passes == pass_validation.DEFAULT_CHECKS
    assert validation.passes == frozenset(pass_validation.CHECKS)
    assert {v["label"]: v["verdict"] for v in validation.gate.verdicts} == {
        "prune_dead_pure_tensor_ops": "accepted",
        "lower_quant_ext": "accepted",
    }


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
        lower.lower_model(kernel("pruned"), tmp_path / "lower", targets=(), pass_validation=validation)
    report = raised.value.report
    assert report["label"] == "prune_dead_pure_tensor_ops"
    assert report["verdict"] == "not_established"
    assert "TypedModule.check_sound did not verify" in report["reason"]
    assert "arith.subf" in Path(report["after"]).read_text()


def test_a_prune_that_changes_nothing_submits_nothing(tmp_path, tools):
    gate = SpeculativeGate(tmp_path / "gate", job_timeout=900)
    prune_only = pass_validation.PassValidation(gate, tools, frozenset({"prune_dead_pure_tensor_ops"}))
    try:
        lower.lower_model(kernel(), tmp_path / "lower", targets=(), pass_validation=prune_only)
    finally:
        gate.close()
    assert gate.verdicts == []


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
    monkeypatch.setenv(pass_validation.ENV_PASSES, "prune_dead_pure_tensor_ops")
    monkeypatch.delenv(pass_validation.ENV_SDK_BIN, raising=False)
    with pytest.raises(ValueError, match="SDK"):
        lower.lower_model(LIVE, tmp_path / "lower", targets=())


@pytest.mark.parametrize("pass_name", ["canonicalize", "cse", "linalg-fuse-elementwise-ops", "linalg-generalize-named-ops"])
def test_upstream_pass_selection_is_refused(tmp_path, tools, pass_name):
    gate = SpeculativeGate(tmp_path / "gate")
    try:
        with pytest.raises(ValueError, match="subset"):
            pass_validation.PassValidation(gate, tools, frozenset({pass_name}))
    finally:
        gate.close()


def test_merlin_compile_flags_request_validation(monkeypatch):
    from merlin.compile import command

    ap = command.parser(("int8",), ("none",), "none", ("k1",))
    args = ap.parse_args(["--workload", "w", "--validate-passes", "all", "--lean-sdk-bin", "/sdk/bin"])
    keys = (pass_validation.ENV_PASSES, pass_validation.ENV_SDK_BIN)
    original = {key: os.environ.get(key) for key in keys}
    with monkeypatch.context() as environment:
        # Register restoration even for initially absent variables: delenv
        # records no undo then, and apply_pass_validation sets them directly.
        for key in keys:
            environment.setenv(key, "")
        command.apply_pass_validation(args)
        assert os.environ[pass_validation.ENV_PASSES] == "all"
        assert os.environ[pass_validation.ENV_SDK_BIN] == "/sdk/bin"
    assert {key: os.environ.get(key) for key in keys} == original


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


@pytest.mark.parametrize("nested", [False, True])
def test_explicit_lowering_width_must_match_its_validation(tmp_path, tools, nested):
    gate = SpeculativeGate(tmp_path / "gate", job_timeout=30)
    validation = pass_validation.PassValidation(gate, tools, frozenset({"prune_dead_pure_tensor_ops"}))
    try:
        if nested:
            with pass_validation.validating(validation), pytest.raises(ValueError, match="index width differs"):
                lower.lower_model(LIVE, tmp_path / "output", targets=(), index_bits=32)
        else:
            with pytest.raises(ValueError, match="index width differs"):
                lower.lower_model(LIVE, tmp_path / "output", targets=(), index_bits=32, pass_validation=validation)
    finally:
        gate.close()
    assert gate.verdicts == []


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
                pass_validation=pass_validation.PassValidation(
                    gate, tools, frozenset({"prune_dead_pure_tensor_ops"})
                ),
            )
    finally:
        gate.close()
    assert not (work / "model.ll").exists()
    assert (work / "model.ll.rejected").is_file()


def test_a_nested_lowering_joins_the_active_validation(tmp_path, validation):
    with pass_validation.validating(validation):
        lower.lower_model(kernel(), tmp_path / "lower", targets=())
        assert pass_validation.current() is validation
    assert not (tmp_path / "lower" / "pass-validation").exists()


def test_a_conflicting_validation_is_refused(tmp_path, validation, tools):
    other = pass_validation.PassValidation(SpeculativeGate(tmp_path / "other"), tools)
    try:
        with pass_validation.validating(validation):
            with pytest.raises(ValueError, match="already active"):
                lower.lower_model(LIVE, tmp_path / "lower", targets=(), pass_validation=other)
    finally:
        other.gate.close()


CHUNKS = [chunk.name for chunk in lean_gate_chunks.all_chunks()]


@pytest.mark.parametrize("variant", ["text", "pruned"])
@pytest.mark.parametrize("name", CHUNKS)
def test_representative_kernels_lower_identically_under_the_gate(tmp_path, tools, name, variant, monkeypatch):
    """Every family of every retained capsule through the real passes: each submitted check accepts,
    the gated LLVM IR is byte-identical to the ungated one, and a lowering that fails without the
    gate fails the same way with it (the gate never masks or changes a compile error)."""
    monkeypatch.delenv(pass_validation.ENV_PASSES, raising=False)
    source = kernel(variant, name)
    try:
        ungated = lower.lower_model(source, tmp_path / "ungated", targets=())
    except Exception as error:  # noqa: BLE001 -- the parity check below re-raises nothing
        ungated = error
    assert not (tmp_path / "ungated" / "pass-validation").exists()
    assert not (tmp_path / "ungated" / "leading-cse").exists()
    gate = SpeculativeGate(tmp_path / "gate", job_timeout=900)
    validation = pass_validation.PassValidation(gate, tools, frozenset({"prune_dead_pure_tensor_ops"}))
    try:
        if isinstance(ungated, Exception):
            with pytest.raises(type(ungated)):
                lower.lower_model(source, tmp_path / "gated", targets=(), pass_validation=validation)
            assert all(v["verdict"] == "accepted" for v in gate.verdicts)
            return
        gated = lower.lower_model(source, tmp_path / "gated", targets=(), pass_validation=validation)
    finally:
        gate.close()
    verdicts = {v["label"]: v["verdict"] for v in gate.verdicts}
    assert set(verdicts.values()) <= {"accepted"}, verdicts
    assert "cse" not in verdicts  # Upstream passes are outside Merlin proof selection.
    assert ("prune_dead_pure_tensor_ops" in verdicts) == (gated.stats["dead_tensor_ops_pruned"] > 0)
    if variant == "pruned":
        assert "prune_dead_pure_tensor_ops" in verdicts
    assert gated.ll_path.read_bytes() == ungated.ll_path.read_bytes()


def test_the_chunks_cover_every_family_the_capsules_use():
    for capsule in lean_gate_chunks.CAPSULES:
        source = lean_gate_chunks.repo_root() / "merlin/contract/capsules/model" / capsule / "capsule.interface.mlir"
        families = {
            line.split('prov.family = "', 1)[1].split('"', 1)[0]
            for line in source.read_text().splitlines()
            if 'prov.family = "' in line
        }
        chunks = lean_gate_chunks.chunks(capsule)
        covered = {f for chunk in chunks for f in families if f'prov.family = "{f}"' in chunk.text}
        assert covered == families, (capsule, families - covered)
