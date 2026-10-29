"""An invariant broadcast operand makes whole-body adjacent reuse exact."""

from __future__ import annotations

import numpy as np
import pytest
import test_adjacent_exp as scalar
import test_invocation_exp_cache as shared

from merlin.llvmlower import adjacent_exp as A

environment = pytest.fixture(scalar.environment.__wrapped__)
binding = pytest.fixture(shared.binding.__wrapped__)

SOURCE = """module {func.func @forward(%x:tensor<4x8xf32>,%s:tensor<4x1xf32>)
 ->tensor<4x8xf32> attributes {llvm.emit_c_interface STRICT} {
 %e=tensor.empty():tensor<4x8xf32>
 %r=linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,
 affine_map<(i,j)->(i,0)>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]}
 ins(%x,%s:tensor<4x8xf32>,tensor<4x1xf32>) outs(%e:tensor<4x8xf32>){
 ^bb0(%a:f32,%b:f32,%out:f32):
 %d=arith.subf %a,%b:f32
 %v=math.exp %d:f32
 linalg.yield %v:f32 }->tensor<4x8xf32>
 return %r:tensor<4x8xf32> }}"""


@pytest.mark.parametrize("feature", [A.INPUT_FEATURE, A.INPUT_UNROLL_FEATURE, A.INPUT_UNROLL4_FEATURE])
@pytest.mark.parametrize("case", ["finite", "nan", "infinity", "subnormal", "changed_rows"])
def test_complete_words_modes_and_effects(tmp_path, environment, binding, case, feature):
    control, _ = scalar.build(tmp_path / "control", True, text=SOURCE, feature=A.INVOCATION_FEATURE)
    candidate, llvm = scalar.build(tmp_path / "candidate", True, text=SOURCE, feature=feature)
    x = np.repeat(np.array([-2, -1, 0, 1, 2, 3, 4, 5], dtype="f4"), 4).reshape(4, 8)
    s = np.array([0.125, 0.25, 0.5, 1], dtype="f4").reshape(4, 1)
    if case == "nan":
        x.view("u4")[:] = np.repeat(np.array([0x7F800001, 0x7FC00123, 0xFF800001, 0xFFC00123], dtype="u4"), 8).reshape(
            4, 8
        )
    if case == "infinity":
        x[:] = np.repeat(np.array([np.inf, -np.inf, 0, 1], dtype="f4"), 8).reshape(4, 8)
    if case == "subnormal":
        x.view("u4")[:] = np.repeat(np.array([1, 0x80000001, 0, 0x80000000], dtype="u4"), 8).reshape(4, 8)
    if case == "changed_rows":
        x[:] = 1
    if feature == A.INPUT_UNROLL_FEATURE:
        assert '"llvm.loop.unroll.count", i32 2' in llvm
    if feature == A.INPUT_UNROLL4_FEATURE:
        assert '"llvm.loop.unroll.count", i32 4' in llvm
    assert llvm != scalar.build(tmp_path / "ordinary-again", True, text=SOURCE, feature=A.INVOCATION_FEATURE)[1]
    for mode in range(4):
        for errno in [0, 34, 42]:
            assert environment.mode(mode) == 0
            environment.reset(errno)
            expected = scalar.execute(control, x, s).view("u4").copy()
            expected_state = environment.flags(), environment.err()
            environment.reset(errno)
            actual = scalar.execute(candidate, x, s).view("u4").copy()
            actual_state = environment.flags(), environment.err()
            np.testing.assert_array_equal(actual, expected)
            assert actual_state == expected_state


def test_two_varying_inputs_retain_the_original_cache(tmp_path, binding):
    _, ordinary = scalar.build(tmp_path / "control", True, feature=A.INVOCATION_FEATURE)
    _, selected = scalar.build(tmp_path / "candidate", True, feature=A.INPUT_FEATURE)
    assert selected == ordinary


@pytest.mark.parametrize("width", [1, 7, 9])
def test_unrolled_input_cache_preserves_tails(tmp_path, environment, binding, width):
    source = SOURCE.replace("4x8", f"4x{width}")
    control, _ = scalar.build(tmp_path / "control", True, text=source, feature=A.INVOCATION_FEATURE)
    candidate, _ = scalar.build(tmp_path / "candidate", True, text=source, feature=A.INPUT_UNROLL4_FEATURE)
    x = np.resize(np.array([-2, -2, 0, 0, 1, 1, 1], dtype="f4"), (4, width))
    shift = np.array([0.125, 0.25, 0.5, 1], dtype="f4").reshape(4, 1)
    for mode in range(4):
        assert environment.mode(mode) == 0
        environment.reset(42)
        expected = scalar.execute(control, x, shift).view("u4").copy()
        expected_state = environment.flags(), environment.err()
        environment.reset(42)
        actual = scalar.execute(candidate, x, shift).view("u4").copy()
        actual_state = environment.flags(), environment.err()
        np.testing.assert_array_equal(actual, expected)
        assert actual_state == expected_state


def test_strict_context_keeps_original_body(tmp_path, binding):
    _, ordinary = scalar.build(tmp_path / "control", True, strict=True, text=SOURCE, feature=A.INVOCATION_FEATURE)
    _, selected = scalar.build(tmp_path / "candidate", True, strict=True, text=SOURCE, feature=A.INPUT_FEATURE)
    assert selected == ordinary


def test_exp_input_reuse_requires_live_selected_callee(tmp_path):
    from merlin.llvmlower.pipeline import lower_to_llvm_ir

    with pytest.raises(ValueError, match="explicit live callee grant"):
        lower_to_llvm_ir(SOURCE.replace("STRICT", ""), workdir=tmp_path, features={A.INPUT_FEATURE})
