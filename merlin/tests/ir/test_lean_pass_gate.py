"""Merlin's dead-code pruning and leading cse are checked against MerlinVerify's theorems while lowering.

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

from merlin.llvmlower import leading_cse, lower, passes_xdsl
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
    yield pass_validation.PassValidation(gate, tools)
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


def test_real_lowering_proves_pruning_and_cse(tmp_path, validation):
    result = lower.lower_model(kernel("pruned"), tmp_path / "lower", targets=(), pass_validation=validation)
    assert result.ll_path.is_file() and result.stats["dead_tensor_ops_pruned"] > 0
    assert result.stats["quant_ext_lowered"] > 0  # dead quantization is pruned too
    verdicts = {v["label"]: v for v in validation.gate.verdicts}
    assert {label: v["verdict"] for label, v in verdicts.items()} == {
        "prune_dead_pure_tensor_ops": "accepted",
        "cse": "accepted",
    }
    # The pipeline's own leading canonicalize,cse then changed nothing (no second cse check).
    assert [v["label"] for v in validation.gate.verdicts].count("cse") == 1
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
    validation = pass_validation.PassValidation(gate, tools)
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
                pass_validation=pass_validation.PassValidation(gate, tools),
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


F23 = "tensor<2x3xf32>"
DUPLICATES = f"""func.func @forward(%a: {F23}) -> (tensor<6xf32>, tensor<6xf32>, {F23}, {F23}) {{
  %c = arith.constant 1.5 : f32
  %d = arith.constant 2.5 : f32
  %e0 = tensor.empty() : {F23}
  %f0 = linalg.fill ins(%c : f32) outs(%e0 : {F23}) -> {F23}
  %s0 = tensor.collapse_shape %a [[0, 1]] : {F23} into tensor<6xf32>
  %e1 = tensor.empty() : {F23}
  %f1 = linalg.fill ins(%CONSTANT : f32) outs(%e1 : {F23}) -> {F23}
  %s1 = tensor.collapse_shape %a [[0, 1]] : {F23} into tensor<6xf32>
  return %s0, %s1, %f0, %f1 : tensor<6xf32>, tensor<6xf32>, {F23}, {F23}
}}
"""


def _cse(tmp_path, text: str) -> str:
    directory = tmp_path / "apply"
    directory.mkdir(exist_ok=True)
    return leading_cse.apply(text, "cse", directory, "cse")


def test_upstream_cse_of_empties_fills_and_reshapes_is_proved(tmp_path, tools):
    before = DUPLICATES.replace("%CONSTANT", "%c")
    after = _cse(tmp_path, before)
    assert after.count("linalg.fill") == 1 and after.count("tensor.empty") == 1 and after.count("collapse_shape") == 1
    detail = gate_checks.tensor_refinement(job(tmp_path, before, after, tools))
    assert detail["claim"].startswith("strict refinement")


def test_a_cse_that_merges_different_fills_is_rejected(tmp_path, tools):
    before = DUPLICATES.replace("%CONSTANT", "%d")  # the second fill writes another value
    merged = _cse(tmp_path, DUPLICATES.replace("%CONSTANT", "%c"))  # what merging both fills would give
    with pytest.raises(GateRejected, match="did not verify"):
        gate_checks.tensor_refinement(job(tmp_path, before, merged, tools))


# A cse bug inside the build's own runner: after the real cse, one multiply in the module the build
# continues from becomes an add.
CSE_BUG = """        from torch_mlir import ir as _bug_ir
        from torch_mlir.dialects import arith as _bug_arith

        def _bug_find(op):
            for region in op.regions:
                for block in region.blocks:
                    for child in block.operations:
                        if child.operation.name == 'arith.mulf':
                            return child.operation
                        found = _bug_find(child.operation)
                        if found is not None:
                            return found
            return None

        _bug = _bug_find(operation)
        with operation.context, _bug_ir.Location.unknown(), _bug_ir.InsertionPoint(_bug):
            _bug_add = _bug_arith.AddFOp(_bug.operands[0], _bug.operands[1])
        _bug.results[0].replace_all_uses_with(_bug_add.result)
        _bug.erase()
        after = operation.get_asm(print_generic_op_form=True)"""


def test_an_injected_cse_bug_stops_real_lowering(tmp_path, validation, monkeypatch):
    line = "        after = operation.get_asm(print_generic_op_form=True)"
    assert leading_cse._PRELUDE.count(line) == 1
    monkeypatch.setattr(leading_cse, "_PRELUDE", leading_cse._PRELUDE.replace(line, CSE_BUG))
    work = tmp_path / "lower"
    with pytest.raises(GateFailure) as raised:
        lower.lower_model(kernel(), work, targets=(), pass_validation=validation)
    assert raised.value.report["label"] == "cse"
    assert raised.value.report["verdict"] == "not_established"
    assert not (work / "model.ll").exists()


@pytest.mark.parametrize(
    "feature",
    ["erase_self_copy", "fuse_transpose_b"],  # one edits a late stage, one rewrites before the pass manager
)
def test_cse_validates_whatever_the_runner_does_around_it(tmp_path, validation, feature):
    work = tmp_path / "lower"
    lower.lower_model(
        kernel(),
        work,
        targets=(),
        features=frozenset({feature}),
        pass_validation=validation,
    )
    assert ("cse", "accepted") in [(v["label"], v["verdict"]) for v in validation.gate.verdicts]
    assert "_LCPassManager" in (work / "run_lowering.py").read_text()  # recorded inside the build's own runner
    assert (work / "model.ll").is_file()


def test_the_runner_records_nothing_without_cse_validation(tmp_path, tools):
    gate = SpeculativeGate(tmp_path / "gate", job_timeout=900)
    prune_only = pass_validation.PassValidation(gate, tools, frozenset({"prune_dead_pure_tensor_ops"}))
    try:
        lower.lower_model(kernel(), tmp_path / "lower", targets=(), pass_validation=prune_only)
    finally:
        gate.close()
    assert "_LCPassManager" not in (tmp_path / "lower" / "run_lowering.py").read_text()


def test_validated_cse_with_nothing_recorded_fails_closed(tmp_path, validation):
    (tmp_path / leading_cse.DIRECTORY).mkdir()
    (tmp_path / leading_cse.DIRECTORY / "index.json").write_text(json.dumps([{"skipped": "bufferized module"}]))
    with pass_validation.validating(validation):
        with pytest.raises(GateFailure, match="no tensor-level canonicalize,cse"):
            leading_cse.submit(tmp_path)


MAP = "affine_map<(d0) -> (d0)>"


def _generic(name: str, source: str, empty: str, body: str) -> str:
    return (
        f"  %{empty} = tensor.empty() : {T}\n"
        f'  %{name} = linalg.generic {{indexing_maps = [{MAP}, {MAP}], iterator_types = ["parallel"]}}'
        f" ins(%{source} : {T}) outs(%{empty} : {T}) {{\n"
        f"  ^bb0(%i: f32, %o: f32):\n    {body}\n    linalg.yield %r : f32\n  }} -> {T}\n"
    )


NEGATE = "%r = arith.negf %i : f32"
# Each repeat becomes visible only after the one before it merges: empty -> g -> h.
CASCADE = (
    f"func.func @forward(%x: {T}) -> ({T}, {T}) {{\n"
    + _generic("g0", "x", "e0", NEGATE)
    + _generic("g1", "x", "e1", NEGATE)
    + _generic("h0", "g0", "e2", NEGATE)
    + _generic("h1", "g1", "e3", NEGATE)
    + f"  return %h0, %h1 : {T}, {T}\n}}\n"
)


def _capturing(first: str, second: str) -> str:
    """Two generics whose bodies read outside constants (not operands)."""
    return (
        f"func.func @forward(%x: {T}) -> ({T}, {T}) {{\n  %c = arith.constant 1.5 : f32\n"
        "  %d = arith.constant 2.5 : f32\n"
        + _generic("g0", "x", "e0", f"%r = arith.addf %i, %{first} : f32")
        + _generic("g1", "x", "e1", f"%r = arith.addf %i, %{second} : f32")
        + f"  return %g0, %g1 : {T}, {T}\n}}\n"
    )


@pytest.mark.parametrize("before", [CASCADE, _capturing("c", "c")], ids=["cascade", "captured"])
def test_upstream_cse_cascades_and_captured_values_are_proved(tmp_path, tools, before):
    after = _cse(tmp_path, before)
    assert after.count("linalg.generic") < before.count("linalg.generic")
    assert gate_checks.tensor_refinement(job(tmp_path, before, after, tools))["claim"].startswith("strict refinement")


def test_a_cse_merging_generics_that_read_different_values_is_rejected(tmp_path, tools):
    merged = _cse(tmp_path, _capturing("c", "c"))  # what merging both would give
    with pytest.raises(GateRejected, match="did not verify"):
        gate_checks.tensor_refinement(job(tmp_path, _capturing("c", "d"), merged, tools))


SQUARE = "%r = arith.mulf %i, %i : f32"
# Upstream cse erases what is unused when it reaches it: the dead first square (so the later one is
# kept, not merged into it) and the unused negation, but not the negation's producer.
ERASES_UNUSED = (
    f"func.func @forward(%x: {T}) -> ({T}, {T}) {{\n"
    + _generic("a", "x", "e0", SQUARE)
    + _generic("b", "x", "e1", NEGATE)
    + _generic("c", "b", "e2", NEGATE)
    + _generic("d", "x", "e3", SQUARE)
    + f"  return %d, %d : {T}, {T}\n}}\n"
)


def test_upstream_cse_erasing_unused_operations_is_proved(tmp_path, tools):
    after = _cse(tmp_path, ERASES_UNUSED)
    assert after.count("linalg.generic") == 2  # b and d survive
    assert gate_checks.tensor_refinement(job(tmp_path, ERASES_UNUSED, after, tools))["claim"].startswith("strict")


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
    assert not (tmp_path / "ungated" / leading_cse.DIRECTORY).exists()
    gate = SpeculativeGate(tmp_path / "gate", job_timeout=900)
    validation = pass_validation.PassValidation(gate, tools)
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
    if variant == "text":
        assert "cse" in verdicts  # every live chunk repeats tensor.empty, so the leading cse changes it
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
