"""Actual ordered reduction words, flags, maps, seeds and strict refusals."""

from __future__ import annotations

import ctypes
import subprocess

import numpy as np
import pytest

from merlin.common.cpu_pool import shared_compute_pool
from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
from merlin.llvmlower.codegen import mlir_runtime_c
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.scalar_ordered_reduction import FEATURE, MARKER, apply_for_test, edit_pipeline
from merlin.llvmlower.toolchain import clang


def source(rows, width, *, operation="arith.addf", reverse=False, transpose=False, live=False):
    inp = f"tensor<{width}x{rows}xf32>" if transpose else f"tensor<{rows}x{width}xf32>"
    out = f"tensor<{rows}xf32>"
    mapping = "(k,i)" if transpose else "(i,k)"
    result = f"({out},{out})" if live else out
    returned = f"%r,%seed:{out},{out}" if live else f"%r:{out}"
    return f"""module {{func.func @forward(%a:{inp},%seed:{out})->{result} attributes {{llvm.emit_c_interface}} {{
      %r=linalg.generic {{indexing_maps=[affine_map<(i,k)->{mapping}>,affine_map<(i,k)->(i)>],
      iterator_types=["parallel","reduction"],prov.label="reduction"}}
      ins(%a:{inp}) outs(%seed:{out}) {{^bb0(%x:f32,%z:f32):
       %v={operation} {"%z,%x" if reverse else "%x,%z"}:f32
       linalg.yield %v:f32 }}->{out}
      return {returned}
    }} }}"""


def build(root, text, selected):
    root.mkdir()
    with shared_compute_pool(timeout_s=300):
        llvm = lower_to_llvm_ir(text, workdir=root, features={FEATURE} if selected else set(), timeout=300)
    path = root / "model.ll"
    path.write_text(llvm)
    image = root / "model.so"
    with shared_compute_pool(timeout_s=300):
        subprocess.run(
            [
                str(clang()),
                "-O2",
                "-fPIC",
                "-shared",
                "-ffp-contract=off",
                str(path),
                str(mlir_runtime_c()),
                "-lm",
                "-o",
                str(image),
            ],
            check=True,
            capture_output=True,
        )
    return HostModel.load(str(image), image_policy=PrivateHostImagePolicy(root)), llvm


def execute(model, inputs, rows, *, live=False):
    outputs = [np.full(rows, -999.0, dtype="f4") for _ in range(2 if live else 1)]
    model([(v.ctypes.data, v.shape) for v in [*inputs, *outputs]])
    return [v.view("u4").copy() for v in outputs]


@pytest.fixture
def environment(tmp_path):
    path = tmp_path / "fenv.c"
    path.write_text("""#include <fenv.h>
 int mode(int n){const int m[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};return fesetround(m[n]);}
 int reset(int n){feclearexcept(FE_ALL_EXCEPT);return n?feraiseexcept(n==1?FE_INVALID|FE_INEXACT:FE_ALL_EXCEPT):0;}
 int flags(void){return fetestexcept(FE_ALL_EXCEPT);}
 """)
    image = tmp_path / "fenv.so"
    with shared_compute_pool(timeout_s=300):
        subprocess.run([str(clang()), "-shared", "-fPIC", str(path), "-lm", "-o", str(image)], check=True)
    env = ctypes.CDLL(str(image))
    yield env
    assert env.mode(0) == 0
    env.reset(0)


@pytest.mark.parametrize("operation", ["arith.addf", "arith.maximumf", "arith.maxnumf"])
@pytest.mark.parametrize("reverse", [False, True])
def test_complete_words_flags_operand_order_and_live_seed(tmp_path, environment, operation, reverse):
    text = source(3, 17, operation=operation, reverse=reverse, transpose=True, live=True)
    with shared_compute_pool(timeout_s=300):
        changed, count = apply_for_test(text, directory=tmp_path / "rewrite")
    assert count == 0 and "scalar_ordered_tensor_reductions" not in changed
    assert 'prov.label = "reduction"' in changed
    control, control_llvm = build(tmp_path / "control", text, False)
    candidate, llvm = build(tmp_path / "candidate", text, True)
    assert llvm == control_llvm
    assert all("noalias" not in line for line in llvm.splitlines() if line.startswith("define "))
    words = np.array(
        [
            0,
            0x80000000,
            0x7FC00123,
            0x7F800001,
            0xFF800001,
            0x7F800000,
            0xFF800000,
            1,
            0x80000001,
            0x3F800001,
            0xBF800000,
            0x4B800000,
            0xCB800000,
            0x33800000,
            0x7F7FFFFF,
            0xFF7FFFFF,
            0x3F800000,
        ],
        dtype="u4",
    )
    x = np.ascontiguousarray(np.tile(words.view("f4"), (3, 1)).T)
    seed = np.array([0x80000000, 0x7F800001, 0x3F000001], dtype="u4").view("f4")
    original = [v.view("u4").copy() for v in (x, seed)]
    for mode in range(4):
        for flags in range(3):
            assert environment.mode(mode) == 0
            assert environment.reset(flags) == 0
            expected = execute(control, [x, seed], 3, live=True)
            expected_flags = environment.flags()
            assert environment.reset(flags) == 0
            actual = execute(candidate, [x, seed], 3, live=True)
            assert environment.flags() == expected_flags
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)
    for value, before in zip((x, seed), original):
        np.testing.assert_array_equal(value.view("u4"), before)


@pytest.mark.parametrize("rows,width", [(1, 1), (5, 11), (2, 0), (0, 5)])
def test_tails_empty_domains_and_nonzero_seed(tmp_path, rows, width):
    text = source(rows, width, reverse=True)
    control, _ = build(tmp_path / "control", text, False)
    candidate, _ = build(tmp_path / "candidate", text, True)
    rng = np.random.default_rng(239)
    x = rng.normal(size=(rows, width)).astype("f4")
    seed = rng.normal(size=rows).astype("f4")
    selected_args = [x.copy(), seed.copy()]
    control_args = [x.copy(), seed.copy()]
    actual = execute(candidate, selected_args, rows)[0]
    expected = execute(control, control_args, rows)[0]
    np.testing.assert_array_equal(actual, expected)
    for a, b in zip(selected_args, control_args):
        np.testing.assert_array_equal(a.view("u4"), b.view("u4"))


@pytest.mark.parametrize("change", ["strict", "fastmath", "extra", "duplicate", "unknown_attr", "wrong_op"])
def test_unsupported_domains_remain_original(tmp_path, change):
    text = source(3, 7)
    if change == "strict":
        text = text.replace("llvm.emit_c_interface", "llvm.emit_c_interface,strictfp")
    if change == "fastmath":
        text = text.replace("arith.addf %x,%z:f32", "arith.addf %x,%z fastmath<reassoc>:f32")
    if change == "extra":
        text = text.replace("%v=arith.addf", "%one=arith.constant 1.0:f32\n%v=arith.addf")
    if change == "duplicate":
        text = text.replace("%x,%z:f32", "%x,%x:f32")
    if change == "unknown_attr":
        text = text.replace('prov.label="reduction"', "other.unknown=true")
    if change == "wrong_op":
        text = text.replace("arith.addf", "arith.mulf")
    with shared_compute_pool(timeout_s=300):
        _, count = apply_for_test(text, directory=tmp_path / "rewrite")
    assert count == 0


def test_stage_is_explicit_and_cannot_be_duplicated():
    original = ["canonicalize", "one-shot-bufferize"]
    assert edit_pipeline(original) == ["canonicalize", MARKER, "one-shot-bufferize"]
    assert original == ["canonicalize", "one-shot-bufferize"]
    with pytest.raises(ValueError):
        edit_pipeline([MARKER, "one-shot-bufferize"])


def typed_source(rows, width, *, operation="arith.addf", reverse=False, transpose=False, live=True):
    text = source(rows, width, operation=operation, reverse=reverse, transpose=transpose, live=live)
    inp = f"tensor<{width}x{rows}xf32>" if transpose else f"tensor<{rows}x{width}xf32>"
    out = f"tensor<{rows}xf32>"
    integer_input, integer_seed = inp.replace("f32", "i32"), out.replace("f32", "i32")
    text = text.replace(f"%a:{inp},%seed:{out}", f"%raw:{integer_input},%rawseed:{integer_seed}", 1)
    casts = f"""%ae=tensor.empty():{inp}
      %a=linalg.generic {{indexing_maps=[affine_map<(i,k)->(i,k)>,affine_map<(i,k)->(i,k)>],
      iterator_types=["parallel","parallel"]}}
      ins(%raw:{integer_input}) outs(%ae:{inp}) {{
      ^bb0(%q:i32,%u:f32):%v=arith.sitofp %q:i32 to f32 linalg.yield %v:f32}}->{inp}
      %se=tensor.empty():{out}
      %seed=linalg.generic {{indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->(i)>],
      iterator_types=["parallel"]}}
      ins(%rawseed:{integer_seed}) outs(%se:{out}) {{
      ^bb0(%q:i32,%u:f32):%v=arith.sitofp %q:i32 to f32 linalg.yield %v:f32}}->{out}
    """
    return text.replace("%r=linalg.generic", casts + "%r=linalg.generic", 1)


@pytest.mark.parametrize("operation", ["arith.addf", "arith.maximumf", "arith.maxnumf"])
@pytest.mark.parametrize("reverse", [False, True])
def test_typed_integer_domain_promotes_and_preserves_all_modes(tmp_path, environment, operation, reverse):
    text = typed_source(3, 17, operation=operation, reverse=reverse, transpose=True)
    with shared_compute_pool(timeout_s=300):
        changed, count = apply_for_test(text, directory=tmp_path / "rewrite")
    assert count == 1 and "scalar_ordered_tensor_reductions" in changed
    control, _ = build(tmp_path / "control", text, False)
    candidate, llvm = build(tmp_path / "candidate", text, True)
    assert "phi float" in llvm
    values = np.array(
        [0, 1, -1, 2**31 - 1, -(2**31), 2**24 + 1, -(2**24) - 1, 13, -23, 41, 127, -128, 239, 673, -913, 145, 7],
        dtype="i4",
    )
    x = np.ascontiguousarray(np.tile(values, (3, 1)).T)
    seed = np.array([-123456789, 1, 2**31 - 1], dtype="i4")
    for mode in range(4):
        for flags in range(3):
            environment.mode(mode)
            environment.reset(flags)
            expected = execute(control, [x, seed], 3, live=True)
            before_flags = environment.flags()
            environment.reset(flags)
            actual = execute(candidate, [x, seed], 3, live=True)
            assert environment.flags() == before_flags
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)


def test_named_reduce_and_aliasing_integer_entries(tmp_path):
    text = typed_source(3, 2)
    text = (
        text[: text.index("%r=linalg.generic")]
        + """%r="linalg.reduce"(%a,%seed) <{dimensions=array<i64:1>}> ({
      ^bb0(%x:f32,%z:f32): %v=arith.addf %x,%z:f32 linalg.yield %v:f32
    }):(tensor<3x2xf32>,tensor<3xf32>)->tensor<3xf32>
    return %r,%seed:tensor<3xf32>,tensor<3xf32> } }"""
    )
    assert "linalg.reduce" in text
    with shared_compute_pool(timeout_s=300):
        _, count = apply_for_test(text, directory=tmp_path / "rewrite")
    assert count == 1
    control, _ = build(tmp_path / "control", text, False)
    candidate, _ = build(tmp_path / "candidate", text, True)
    base = np.array([2**24 + 1, -17, 239, -673, 2**31 - 1, -(2**31)], dtype="i4")
    before = base.copy()
    x, seed = base.reshape(3, 2), base[:3]
    expected = execute(control, [x, seed], 3, live=True)
    actual = execute(candidate, [x, seed], 3, live=True)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(base, before)


def test_nonnegative_exp_sum_accepts_unsigned_seed_but_refuses_unknown_sign(tmp_path):
    text = typed_source(3, 7)
    # Every exp operand is a finite converted integer, so the producer is
    # nonnegative/non-NaN even if it overflows. A negative seed is not admitted.
    text = text.replace(
        "%v=arith.sitofp %q:i32 to f32 linalg.yield %v:f32",
        "%v=arith.sitofp %q:i32 to f32 %e=math.exp %v:f32 linalg.yield %e:f32",
        1,
    )
    with shared_compute_pool(timeout_s=300):
        _, count = apply_for_test(text, directory=tmp_path / "unknown-sign")
    assert count == 0
    text = text.replace(
        "%v=arith.sitofp %q:i32 to f32 linalg.yield %v:f32", "%v=arith.uitofp %q:i32 to f32 linalg.yield %v:f32"
    )
    with shared_compute_pool(timeout_s=300):
        _, count = apply_for_test(text, directory=tmp_path / "positive")
    assert count == 1


def test_dense_nan_seed_is_not_admitted(tmp_path):
    text = typed_source(3, 7, operation="arith.maximumf")
    start = text.index("%se=tensor.empty()")
    end = text.index("%r=linalg.generic", start)
    text = text[:start] + "%seed=arith.constant dense<0x7FC00123>:tensor<3xf32>\n" + text[end:]
    with shared_compute_pool(timeout_s=300):
        _, count = apply_for_test(text, directory=tmp_path / "rewrite")
    assert count == 0


@pytest.mark.parametrize("prefix", ["canonicalize", "func.func(linalg-generalize-named-ops)"])
def test_runner_preserves_pre_generalization_hook_on_its_actual_stage(prefix):
    from merlin.llvmlower.scalar_ordered_reduction import RUNNER_PRELUDE

    calls = []
    scope = {"_run_stages": lambda *args: calls.append(args)}
    exec(RUNNER_PRELUDE, scope)
    scope["_scalarize_ordered_reductions"] = lambda *args: 0
    hooks = ("original-source-hook",)
    scope["_run_stages"](
        None, None, prefix + "," + MARKER + ",one-shot-bufferize", 7, ("mid",), ("late",), ("post",), hooks
    )
    assert calls[0][7] == (hooks if "generalize" in prefix else ())
    assert calls[1][7] == (() if "generalize" in prefix else hooks)
    assert calls[1][3:7] == (7, ("mid",), ("late",), ("post",))


def test_unselected_runner_delegates_all_original_arguments():
    from merlin.llvmlower.scalar_ordered_reduction import RUNNER_PRELUDE

    calls = []
    scope = {"_run_stages": lambda *args: calls.append(args)}
    exec(RUNNER_PRELUDE, scope)
    args = (None, None, "canonicalize", 7, ("mid",), ("late",), ("post",), ("pre",))
    scope["_run_stages"](*args)
    assert calls == [args]
