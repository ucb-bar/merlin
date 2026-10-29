"""Invariant divisor extraction preserves original divide bits and floating flags."""

import ctypes
import subprocess

import numpy as np
import pytest

from merlin.llvmlower.broadcast_divisor_hoisting import rewrite_text


def source(*, broadcast=True, scalar="arith.divf", seed=False, strict=False, fast=False):
    denominator = "1" if broadcast else "8"
    mapping = "0" if broadcast else "j"
    return f"""module {{func.func @forward(%x:tensor<4x8xf32>,%y:tensor<4x{denominator}xf32>)
      ->tensor<4x8xf32> attributes {{llvm.emit_c_interface{', strictfp' if strict else ''}}} {{
      %e=tensor.empty():tensor<4x8xf32>
      %r=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,
        affine_map<(i,j)->(i,{mapping})>,affine_map<(i,j)->(i,j)>],
        iterator_types=["parallel","parallel"]}}
        ins(%x,%y:tensor<4x8xf32>,tensor<4x{denominator}xf32>) outs(%e:tensor<4x8xf32>){{
        ^bb0(%a:f32,%b:f32,%initial:f32):
        %v={scalar} %a,%b {'fastmath<fast>' if fast else ''}:f32
        linalg.yield {'%initial' if seed else '%v'}:f32 }}->tensor<4x8xf32>
      return %r:tensor<4x8xf32> }}}} """


def test_only_closed_broadcast_division_is_rewritten():
    text, report = rewrite_text(source())
    assert report == {"row_invariant_divisors_hoisted": 1}
    assert text.count('"scf.for"') == 2 and text.count('"arith.divf"') == 1
    assert rewrite_text(text)[1] == {}


@pytest.mark.parametrize("kwargs", [dict(broadcast=False), dict(scalar="arith.mulf"),
                                     dict(seed=True), dict(strict=True), dict(fast=True)])
def test_unsupported_source_retains_identical_text(kwargs):
    text = source(**kwargs)
    assert rewrite_text(text) == (text, {})


@pytest.fixture
def environment(tmp_path):
    from merlin.llvmlower.toolchain import clang

    path = tmp_path / "environment.c"
    path.write_text("""#include <fenv.h>
      int mode(int n){const int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};return fesetround(modes[n]);}
      int flags(void){return fetestexcept(FE_ALL_EXCEPT);}
      int clear(void){return feclearexcept(FE_ALL_EXCEPT);}
      int sticky(void){return feraiseexcept(FE_INEXACT);}
    """)
    so = tmp_path / "environment.so"
    subprocess.run([str(clang()), "-shared", "-fPIC", str(path), "-lm", "-o", str(so)],
                   check=True, capture_output=True)
    env = ctypes.CDLL(str(so))
    yield env
    assert env.mode(0) == 0


def build(tmp_path, text):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import build_host_shared
    from merlin.llvmlower.pipeline import lower_to_llvm_ir

    tmp_path.mkdir()
    llvm = lower_to_llvm_ir(text, workdir=tmp_path)
    path = tmp_path / "model.ll"
    path.write_text(llvm)
    shared = build_host_shared(path, tmp_path / "model.so")
    return HostModel.load(str(shared))


def execute(model, x, y, output):
    model([(value.ctypes.data, value.shape) for value in (x, y, output)])


@pytest.mark.parametrize("unit_guard", [False, True])
def test_complete_native_edge_words_flags_modes_dirty_repeats_and_canaries(tmp_path, environment, unit_guard):
    control = build(tmp_path / "control", source())
    text, _ = rewrite_text(source(), unit_guard=unit_guard)
    candidate = build(tmp_path / "candidate", text)
    words = np.array([
        0, 0x80000000, 0x3F800000, 0xBF800000, 0x00800000, 0x80800000, 0x7F7FFFFF, 0xFF7FFFFF,
        1, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x7F800000, 0xFF800000, 0x7FC01234, 0x7F800001,
    ], dtype="u4")
    x = np.tile(words, 2).view("f4").reshape(4, 8)
    denominator_words = [0x3F800000, 0x40000000, 0x00800000, 0x7F7FFFFF,
                         1, 0x007FFFFF, 0x80000001, 0, 0x80000000,
                         0xBF800000, 0x7F800000, 0x7FC01234, 0x7F800001]
    for mode in range(4):
        assert environment.mode(mode) == 0
        for denominator in denominator_words:
            y = np.full((4, 1), denominator, dtype="u4").view("f4")
            stores = [np.full(48, 0xA5A5A5A5, dtype="u4") for _ in range(2)]
            outputs = [store[8:40].view("f4").reshape(4, 8) for store in stores]
            for sticky in (False, True):
                environment.clear()
                if sticky:
                    environment.sticky()
                execute(control, x, y, outputs[0])
                flags = environment.flags()
                environment.clear()
                if sticky:
                    environment.sticky()
                execute(candidate, x, y, outputs[1])
                assert environment.flags() == flags
                np.testing.assert_array_equal(outputs[1].view("u4"), outputs[0].view("u4"))
            for store in stores:
                np.testing.assert_array_equal(store[:8], np.full(8, 0xA5A5A5A5, dtype="u4"))
                np.testing.assert_array_equal(store[40:], np.full(8, 0xA5A5A5A5, dtype="u4"))
    assert environment.mode(0) == 0


def test_normal_lower_model_selects_explicit_feature_and_retains_source(tmp_path):
    import json

    from merlin.llvmlower.broadcast_divisor_hoisting import FEATURE
    from merlin.llvmlower.lower import lower_model

    control = lower_model(source(), tmp_path / "control", targets=(), features=frozenset())
    candidate = lower_model(source(), tmp_path / "candidate", targets=(), features=frozenset({FEATURE}))
    assert "broadcast_divisor_hoisting" not in control.stats
    report = json.loads((tmp_path / "candidate" / "broadcast-divisor-hoisting.json").read_text())
    assert report["rewrites"] == {"row_invariant_divisors_hoisted": 1}
    assert candidate.stats["broadcast_divisor_hoisting"] == report
    assert "fdiv float" in candidate.ll_path.read_text()


def test_aliasing_numerator_divisor_and_output_retains_tensor_values(tmp_path):
    control = build(tmp_path / "control", source())
    selected, _ = rewrite_text(source())
    candidate = build(tmp_path / "candidate", selected)
    original = np.arange(1, 33, dtype="f4").reshape(4, 8)
    outputs = []
    for model in (control, candidate):
        shared = original.copy()
        divisor = shared.reshape(-1)[:4].reshape(4, 1)
        model([(shared.ctypes.data, shared.shape), (divisor.ctypes.data, divisor.shape),
               (shared.ctypes.data, shared.shape)])
        outputs.append(shared.copy())
    np.testing.assert_array_equal(outputs[1].view("u4"), outputs[0].view("u4"))
    expected = np.float32(original / original.reshape(-1)[:4, None])
    np.testing.assert_array_equal(outputs[1].view("u4"), expected.view("u4"))


@pytest.mark.parametrize("kwargs", [dict(broadcast=False), dict(scalar="arith.mulf"),
                                     dict(seed=True), dict(strict=True), dict(fast=True)])
def test_unit_guard_unsupported_source_retains_identical_text(kwargs):
    text = source(**kwargs)
    assert rewrite_text(text, unit_guard=True) == (text, {})


def test_unit_guard_retains_literal_divide_fallback_and_is_idempotent():
    text, report = rewrite_text(source(), unit_guard=True)
    assert report == {"row_invariant_divisors_hoisted": 1, "unit_divisor_rows_guarded": 1}
    assert text.count('"arith.divf"') == 1
    assert text.count('"scf.if"') == 2
    assert text.count('"scf.for"') == 4
    assert rewrite_text(text, unit_guard=True) == (text, {})


def test_supported_staged_readonly_alias_and_writable_overlap_refusal(tmp_path):
    from merlin.llvmlower.abi import StridedMemRefArg
    text, _ = rewrite_text(source(), unit_guard=True)
    candidate = build(tmp_path / "staged_candidate", text)
    x = np.arange(1, 33, dtype="f4").reshape(4, 8)
    x.reshape(-1)[:4] = [2, 1, 1, 1]
    y = x.reshape(-1)[:4].reshape(4, 1)
    out = np.empty_like(x)
    a = StridedMemRefArg(x.ctypes.data, (4, 8), (8, 1), 32, "f32", "input")
    b = StridedMemRefArg(y.ctypes.data, (4, 1), (1, 1), 4, "f32", "input")
    c = StridedMemRefArg(out.ctypes.data, (4, 8), (8, 1), 32, "f32", "output")
    candidate([a, b, c])
    np.testing.assert_array_equal(out.view("u4"), np.float32(x / y).view("u4"))
    bad = StridedMemRefArg(x.ctypes.data, (4, 8), (8, 1), 32, "f32", "output")
    before = x.copy()
    with pytest.raises(ValueError, match="overlap"):
        candidate([a, b, bad])
    np.testing.assert_array_equal(x.view("u4"), before.view("u4"))


def test_explicit_unit_feature_and_tensor_source_alias(tmp_path):
    from merlin.llvmlower.broadcast_divisor_hoisting import UNIT_FEATURE
    from merlin.llvmlower.lower import lower_model
    result = lower_model(source(), tmp_path / "ordinary_unit", targets=(), features=frozenset({UNIT_FEATURE}))
    assert result.stats["broadcast_divisor_hoisting"]["rewrites"]["unit_divisor_rows_guarded"] == 1
    # A single original tensor SSA value supplies both read-only operands.
    text = source().replace("%x:tensor<4x8xf32>,%y:tensor<4x1xf32>", "%x:tensor<4x8xf32>")
    text = text.replace(
        "%e=tensor.empty():tensor<4x8xf32>",
        "%y=tensor.extract_slice %x[0,0][4,1][1,1]:tensor<4x8xf32> to tensor<4x1xf32>\n"
        "%e=tensor.empty():tensor<4x8xf32>",
    )
    control = build(tmp_path / "ssa_control", text)
    selected, _ = rewrite_text(text, unit_guard=True)
    candidate = build(tmp_path / "ssa_candidate", selected)
    x = np.arange(1, 33, dtype="f4").reshape(4, 8)
    x[:, 0] = [1, 2, 1, 4]
    expected = np.float32(x / x[:, :1])
    for model in [control, candidate]:
        output = np.empty_like(x)
        model([(x.ctypes.data, x.shape), (output.ctypes.data, output.shape)])
        np.testing.assert_array_equal(output.view("u4"), expected.view("u4"))


def sum_source(*, scalar="arith.addf", other=False, fast=False):
    value, typ = ("%y", "tensor<4x1xf32>") if other else ("%x", "tensor<4x8xf32>")
    relaxed = "fastmath<fast>" if fast else ""
    prefix = f"""
    %zero=arith.constant 0.0:f32
    %seed=tensor.splat %zero:tensor<4xf32>
    %sum="linalg.reduce"({value},%seed) <{{dimensions=array<i64:1>}}> ({{
      ^bb0(%a:f32,%b:f32):
      %c={scalar} %a,%b {relaxed}:f32
      linalg.yield %c:f32
    }}):({typ},tensor<4xf32>)->tensor<4xf32>
    %den="tensor.expand_shape"(%sum)
      <{{reassociation=[[0:i64,1:i64]],static_output_shape=array<i64:4,1>}}>:(tensor<4xf32>)->tensor<4x1xf32>
    """
    return source().replace("%e=tensor.empty():tensor<4x8xf32>", prefix + "%e=tensor.empty():tensor<4x8xf32>").replace(
        "ins(%x,%y:", "ins(%x,%den:")


def test_finite_sum_witness_uses_exact_same_complete_sum():
    text, report = rewrite_text(sum_source(), unit_guard=True, sum_unit_proof=True)
    assert report["same_tensor_row_sum_witnesses"] == 1
    assert text.count('"scf.for"') == 3
    assert text.count('"arith.divf"') == 1
    assert text.count('"scf.if"') == 1
    for kwargs in [dict(scalar="arith.maxnumf"), dict(other=True), dict(fast=True)]:
        text, report = rewrite_text(sum_source(**kwargs), unit_guard=True, sum_unit_proof=True)
        assert report["same_tensor_row_sum_witnesses"] == 0
        assert text.count('"scf.for"') == 4
    strict = sum_source().replace("llvm.emit_c_interface", "llvm.emit_c_interface, strictfp")
    assert rewrite_text(strict, unit_guard=True, sum_unit_proof=True) == (strict, {})


@pytest.mark.parametrize("kwargs", [{}, dict(scalar="arith.maxnumf"), dict(other=True)])
def test_same_sum_fdiv_modes_flags_and_words(tmp_path, environment, kwargs):
    control = build(tmp_path / "control", sum_source(**kwargs))
    text, _ = rewrite_text(sum_source(**kwargs), unit_guard=True, sum_unit_proof=True)
    candidate = build(tmp_path / "candidate", text)
    words = np.array([
        0, 0x80000000, 0x3F800000, 0xBF800000, 0x00800000, 0x80800000, 0x7F7FFFFF, 0xFF7FFFFF,
        1, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x7F800000, 0xFF800000, 0x7FC01234, 0x7F800001,
    ], dtype="u4")
    patterns = [np.tile(words, 2).reshape(4, 8), np.array([
        [0x3F800000, 0, 0, 0, 0, 0, 0, 0],
        [0x3F000000, 0x3F000000, 0, 0, 0, 0, 0, 0],
        [0x7F7FFFFF, 0xFF7FFFFF, 0x3F800000, 0, 0, 0, 0, 0],
        [0x3E800000] * 4 + [0] * 4,
    ], dtype="u4")]
    for mode in range(4):
        environment.mode(mode)
        for pattern in patterns:
            for sticky in [False, True]:
                stores = [np.full(48, 0xA5A5A5A5, dtype="u4") for _ in range(2)]
                outputs = [store[8:40].view("f4").reshape(4, 8) for store in stores]
                flags = []
                for model, output in zip([control, candidate], outputs):
                    environment.clear()
                    if sticky:
                        environment.sticky()
                    execute(model, pattern.view("f4"), np.ones((4, 1), dtype="f4"), output)
                    flags.append(environment.flags())
                assert flags[0] == flags[1]
                np.testing.assert_array_equal(stores[0], stores[1])
                for store in stores:
                    assert np.all(store[:8] == 0xA5A5A5A5) and np.all(store[40:] == 0xA5A5A5A5)
    environment.mode(0)


def test_explicit_sum_unit_feature(tmp_path):
    from merlin.llvmlower.broadcast_divisor_hoisting import SUM_UNIT_FEATURE
    from merlin.llvmlower.lower import lower_model

    result = lower_model(sum_source(), tmp_path / "ordinary", targets=(), features=frozenset({SUM_UNIT_FEATURE}))
    assert result.stats["broadcast_divisor_hoisting"]["rewrites"]["same_tensor_row_sum_witnesses"] == 1



def test_power_two_feature_requires_complete_same_row_sum(tmp_path):
    from merlin.llvmlower.broadcast_divisor_hoisting import POW2_FEATURE
    from merlin.llvmlower.lower import lower_model

    result = lower_model(sum_source(), tmp_path / "ordinary_power_two", targets=(),
                         features=frozenset({POW2_FEATURE}))
    report = result.stats["broadcast_divisor_hoisting"]
    assert report["schema"] == "merlin.broadcast_power_two_divisor_guard.v1"
    assert report["rewrites"]["power_two_row_sum_witnesses"] == 1
    for kwargs in [dict(scalar="arith.maxnumf"), dict(other=True), dict(fast=True)]:
        text, report = rewrite_text(sum_source(**kwargs), unit_guard=True,
                                    sum_unit_proof=True, power_two_guard=True)
        assert report["power_two_row_sum_witnesses"] == 0
        assert '"arith.mulf"' not in text


def test_power_two_native_endpoints_flags_and_all_host_modes(tmp_path, environment):
    control = build(tmp_path / "control", sum_source())
    text, _ = rewrite_text(sum_source(), unit_guard=True, sum_unit_proof=True, power_two_guard=True)
    candidate = build(tmp_path / "candidate", text)
    divisors = [
        0x00800000, 0x01000000, 0x08000000, 0x3E800000, 0x3F000000, 0x3F800000,
        0x40000000, 0x40800000, 0x41000000, 0x7E800000, 0x7F000000, 0x7F800000,
        0xFF800000, 0x7FC01234, 0x7F800001, 0x3F800001, 0x007FFFFF, 1, 0, 0x80000000,
    ]
    for mode in range(4):
        environment.mode(mode)
        for divisor in divisors:
            words = np.zeros((4, 8), dtype="u4")
            words[0, 0] = divisor
            words[1, :2] = [divisor, 1]
            words[2, :3] = [0x7F7FFFFF, 0xFF7FFFFF, divisor]
            words[3, 0] = divisor ^ 0x80000000
            for sticky in [False, True]:
                stores = [np.full(48, 0xA5A5A5A5, dtype="u4") for _ in range(2)]
                flags = []
                for model, store in zip([control, candidate], stores):
                    environment.clear()
                    if sticky:
                        environment.sticky()
                    execute(model, words.view("f4"), np.zeros((4, 1), dtype="f4"),
                            store[8:40].view("f4").reshape(4, 8))
                    flags.append(environment.flags())
                assert flags[0] == flags[1]
                np.testing.assert_array_equal(stores[0], stores[1])
                for store in stores:
                    assert np.all(store[:8] == 0xA5A5A5A5) and np.all(store[40:] == 0xA5A5A5A5)
    environment.mode(0)
