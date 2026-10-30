"""Only a projector's typed model refusal may become a source exclusion.

Host fault injections: the unchanged source is first shown to be admitted, then
only its projector is replaced. No Lean, native or proof process runs.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.verify import buffer_reuse_bridge, lean_graph_state_scalar_capture, pass_validation
from merlin.verify import compiler_ir_source_profile as memory_profile
from merlin.verify import compiler_loop_generic_profile as generic
from merlin.verify import compiler_loop_source_profile as loop_profile
from merlin.verify import compiler_transforms as transforms
from merlin.verify.compiler_profile import SourceOutsideProfile
from merlin.verify.lean_graph_input_admission import ModelRefusal
from merlin.verify.speculative_gate import GateFailure, SpeculativeGate

GENERIC = """builtin.module {
func.func private @prefix(tensor<4xf32>)->tensor<4xf32>
func.func private @tick(index,tensor<4xf32>)
func.func @forward(%left:tensor<4xf32>,%right:tensor<4xf32>)->tensor<4xf32> {
%r=func.call @prefix(%left):(tensor<4xf32>)->tensor<4xf32>
%l=arith.constant 0:index
%u=arith.constant 5:index
%s=arith.constant 1:index
scf.for %i=%l to %u step %s {func.call @tick(%i,%right):(index,tensor<4xf32>)->() scf.yield}
func.return %r:tensor<4xf32>
}}
"""
WORD = """builtin.module {func.func @forward(%x:i32)->i32 {
%zero=arith.constant 0:index %five=arith.constant 5:index %one=arith.constant 1:index
%r=scf.for %i=%zero to %five step %one iter_args(%c=%x)->(i32) {
  %n=arith.addi %c,%c:i32 scf.yield %n:i32
}
func.return %r:i32}}
"""
MEMORY = """builtin.module {func.func @f(%view:memref<3xi32>)->i32 {
%dead=arith.constant 0:index %r=arith.constant 0:i32 func.return %r:i32}}
"""


def _apply_generic(tmp_path, monkeypatch, error):
    assert loop_profile.admit_source(GENERIC, kind=loop_profile.UNROLL) == loop_profile.GENERIC_PROFILE
    ordinary, _ = transforms.apply(GENERIC, tmp_path / "ordinary", options=transforms.Options(unroll_factor=2))
    tools = object.__new__(pass_validation.CheckerTools)
    object.__setattr__(tools, "sdk_bin", tmp_path / "intentionally-unused-sdk")
    object.__setattr__(tools, "compiler_python", Path(sys.executable))
    gate = SpeculativeGate(tmp_path / "gate", interrupt=False)
    validation = pass_validation.PassValidation(gate, tools, frozenset({transforms.UNROLL}))

    def broken_projection(*args, **kwargs):
        raise error

    monkeypatch.setattr(generic, "project_pair", broken_projection)
    caught, actual = None, None
    try:
        with pass_validation.validating(validation):
            actual, _ = transforms.apply(GENERIC, tmp_path / "selected", options=transforms.Options(unroll_factor=2))
    except Exception as failure:
        caught = failure
    finally:
        gate.close()
    coverage = json.loads((gate.directory / "coverage.json").read_text())
    return caught, actual, ordinary, gate, coverage


@pytest.mark.parametrize("error_type", [ValueError, KeyError, TypeError, IndexError, AttributeError, RuntimeError])
def test_unexpected_generic_projection_error_blocks_compilation(tmp_path, monkeypatch, error_type):
    caught, actual, _, gate, coverage = _apply_generic(
        tmp_path, monkeypatch, error_type("unexpected projection implementation fault")
    )
    assert isinstance(caught, GateFailure) and actual is None
    assert caught.report["verdict"] == "unavailable"
    assert not gate.verdicts
    assert coverage["compiler_completed"] is False
    assert coverage["unverified_boundaries"] == 0


def test_typed_generic_model_refusal_remains_an_unverified_exclusion(tmp_path, monkeypatch):
    caught, actual, ordinary, gate, coverage = _apply_generic(
        tmp_path, monkeypatch, ModelRefusal("whole function CFG is unsupported")
    )
    assert caught is None and actual == ordinary
    assert not gate.verdicts
    assert coverage["compiler_completed"] is True
    assert coverage["unverified_boundaries"] == 1 and coverage["verified_boundaries"] == 0


def _admit(tmp_path, monkeypatch, family, error):
    if family == "word":
        raw, target, function = WORD, lean_graph_state_scalar_capture, "project"

        def predicate(text):
            return loop_profile.admit_source(text, kind=loop_profile.UNROLL)

        pass_name = "ordered_loop_unrolling"
    else:
        raw, target, function = MEMORY, buffer_reuse_bridge, "decode_pair"
        predicate = memory_profile.admit_memory_dce
        pass_name = "prune_dead_pure_tensor_ops"
    assert type(predicate(raw)) is str

    def broken_projection(*args, **kwargs):
        raise error

    monkeypatch.setattr(target, function, broken_projection)
    validation = SimpleNamespace(gate=SimpleNamespace(directory=tmp_path))
    token = pass_validation._ACTIVE.set(validation)
    try:
        return pass_validation.source_admission(pass_name, raw, predicate=predicate)
    finally:
        pass_validation._ACTIVE.reset(token)


@pytest.mark.parametrize("family", ["word", "memory"])
@pytest.mark.parametrize("error_type", [ValueError, KeyError, TypeError, RuntimeError])
def test_unexpected_word_or_memory_projection_error_blocks(tmp_path, monkeypatch, family, error_type):
    with pytest.raises(GateFailure) as caught:
        _admit(tmp_path, monkeypatch, family, error_type("unexpected projection implementation fault"))
    assert caught.value.report["verdict"] == "unavailable"


def test_typed_memory_model_refusal_remains_an_exclusion(tmp_path, monkeypatch):
    result = _admit(tmp_path, monkeypatch, "memory", ModelRefusal("dimension axis outside its declared shape"))
    assert isinstance(result, SourceOutsideProfile) and result.code == "source-model"


def test_typed_word_model_refusal_still_falls_back_to_the_generic_profile(tmp_path, monkeypatch):
    # The word refusal is intentional: a non-word source is retried under the
    # generic model, whose own refusal yields one combined exclusion.
    result = _admit(tmp_path, monkeypatch, "word", ModelRefusal("unsupported region semantics"))
    assert isinstance(result, SourceOutsideProfile) and result.code == "outside-word-and-generic-model"
    assert loop_profile.admit_source(GENERIC, kind=loop_profile.UNROLL) == loop_profile.GENERIC_PROFILE
