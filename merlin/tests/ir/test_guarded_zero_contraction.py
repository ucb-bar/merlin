"""Complete native values and fenv fallback for optional zero-product skipping."""

from __future__ import annotations

import ctypes
import subprocess

import numpy as np
import pytest

from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.codegen import build_host_shared
from merlin.llvmlower.guarded_zero_contraction import (
    FEATURE,
    INDEX_FEATURE,
    PREFIX_FEATURE,
    RUNTIME_SYMBOL,
    ensure_registered,
)
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.scalar_contraction import RECTANGULAR_FEATURE
from merlin.llvmlower.toolchain import clang

ensure_registered()


def source(m=4, n=8, k=7, *, seed="0.0", integer_rhs=True, strict=False):
    rhs = "i8" if integer_rhs else "f32"
    cast = (
        """%e=tensor.empty():tensor<KxNxf32>
    %b=linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],
      iterator_types=["parallel","parallel"]} ins(%raw:tensor<KxNxi8>) outs(%e:tensor<KxNxf32>){
      ^bb0(%x:i8,%unused:f32): %y=arith.sitofp %x:i8 to f32
      linalg.yield %y:f32 }->tensor<KxNxf32>"""
        if integer_rhs
        else ""
    )
    b = "%b" if integer_rhs else "%raw"
    text = """module {func.func @forward(%p:tensor<MxKxf32>,%raw:tensor<KxNxRHS>)
      ->tensor<MxNxf32> attributes {llvm.emit_c_interface STRICT} {
      CAST
      %z=arith.constant SEED:f32
      %empty=tensor.empty():tensor<MxNxf32>
      %initial=tensor.splat %z:tensor<MxNxf32>
      %r=linalg.generic {indexing_maps=[affine_map<(i,j,k)->(i,k)>,
        affine_map<(i,j,k)->(k,j)>,affine_map<(i,j,k)->(i,j)>],
        iterator_types=["parallel","parallel","reduction"]}
        ins(%p,B:tensor<MxKxf32>,tensor<KxNxf32>) outs(%initial:tensor<MxNxf32>){
        ^bb0(%x:f32,%y:f32,%a:f32): %v=arith.mulf %x,%y:f32
        %s=arith.addf %v,%a:f32
        linalg.yield %s:f32 }->tensor<MxNxf32>
      return %r:tensor<MxNxf32> }}"""
    return (
        text.replace("CAST", cast)
        .replace("RHS", rhs)
        .replace("STRICT", ", strictfp" if strict else "")
        .replace("SEED", seed)
        .replace("B:", b + ":")
        .replace("M", str(m))
        .replace("N", str(n))
        .replace("K", str(k))
    )


def build(tmp_path, text, *, selected, feature=FEATURE):
    tmp_path.mkdir()
    llvm = lower_to_llvm_ir(text, workdir=tmp_path, features={RECTANGULAR_FEATURE} | ({feature} if selected else set()))
    path = tmp_path / "model.ll"
    path.write_text(llvm)
    shared = build_host_shared(path, tmp_path / "model.so")
    return HostModel.load(str(shared)), llvm


def execute(model, p, b):
    output = np.empty((p.shape[0], b.shape[1]), dtype="f4")
    model([(p.ctypes.data, p.shape), (b.ctypes.data, b.shape), (output.ctypes.data, output.shape)])
    return output


@pytest.fixture
def environment(tmp_path):
    c = tmp_path / "environment.c"
    c.write_text("""#include <fenv.h>
      int mode(int n){const int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};return fesetround(modes[n]);}
      int flags(void){return fetestexcept(FE_ALL_EXCEPT);}
      int clear(void){return feclearexcept(FE_ALL_EXCEPT);}
    """)
    so = tmp_path / "environment.so"
    subprocess.run([str(clang()), "-shared", "-fPIC", str(c), "-lm", "-o", str(so)], check=True, capture_output=True)
    env = ctypes.CDLL(str(so))
    yield env
    assert env.mode(0) == 0


@pytest.mark.parametrize("feature", [FEATURE, PREFIX_FEATURE, INDEX_FEATURE])
@pytest.mark.parametrize(
    "case", ["bounded", "over_bound", "infinity", "signaling_nan", "trailing_zeros", "all_zeros", "subnormal"]
)
def test_complete_native_bits_flags_and_all_rounding_modes(tmp_path, monkeypatch, environment, case, feature):
    monkeypatch.setenv("MERLIN_GENERALIZE_BEFORE_FUSE", "1")
    control, _ = build(tmp_path / "control", source(), selected=False)
    candidate, llvm = build(tmp_path / "candidate", source(), selected=True, feature=feature)
    assert RUNTIME_SYMBOL in llvm
    p = np.array([[0, -0.0, 0.25, -0.5, 0, 1, -1]] * 4, dtype="f4")
    if case == "trailing_zeros":
        p[:, 3:] = np.array([0.0, -0.0, 0.0, -0.0], dtype="f4")
    if case == "all_zeros":
        p[:] = 0.0
        p.view("u4")[::2, :] = 0x80000000
    if case == "subnormal":
        p.view("u4")[:, 3:] = np.array([1, 0x80000001, 0, 0x80000000], dtype="u4")
    if case == "over_bound":
        p[1, 2] = np.float32(1.000001)
    if case == "infinity":
        p[1, 2] = np.inf
    if case == "signaling_nan":
        p.view("u4")[1, 2] = 0x7F800001
    b = np.arange(-28, 28, dtype="i1").reshape(7, 8)
    b[:, 0] = 0
    for mode in range(4):
        assert environment.mode(mode) == 0
        environment.clear()
        expected = execute(control, p, b)
        flags = environment.flags()
        environment.clear()
        actual = execute(candidate, p, b)
        actual_flags = environment.flags()
        np.testing.assert_array_equal(actual.view("u4"), expected.view("u4"))
        assert actual_flags == flags
    assert environment.mode(0) == 0


@pytest.mark.parametrize("feature", [FEATURE, PREFIX_FEATURE, INDEX_FEATURE])
@pytest.mark.parametrize("change", ["negative_zero_seed", "nonzero_seed", "unknown_rhs", "strict"])
def test_unproved_source_retains_original_runtime_path(tmp_path, monkeypatch, change, feature):
    monkeypatch.setenv("MERLIN_GENERALIZE_BEFORE_FUSE", "1")
    kwargs = (
        {"seed": "-0.0"}
        if change == "negative_zero_seed"
        else {"seed": "1.0"}
        if change == "nonzero_seed"
        else {"integer_rhs": False}
        if change == "unknown_rhs"
        else {"strict": True}
    )
    _, llvm = build(tmp_path / "candidate", source(**kwargs), selected=True, feature=feature)
    assert RUNTIME_SYMBOL not in llvm


@pytest.mark.parametrize("feature", [FEATURE, PREFIX_FEATURE, INDEX_FEATURE])
def test_signed_i64_extremes_keep_complete_bits_and_flags(tmp_path, monkeypatch, environment, feature):
    monkeypatch.setenv("MERLIN_GENERALIZE_BEFORE_FUSE", "1")
    text = source().replace("i8", "i64")
    control, _ = build(tmp_path / "control", text, selected=False)
    candidate, llvm = build(tmp_path / "candidate", text, selected=True, feature=feature)
    assert RUNTIME_SYMBOL in llvm
    p = np.array([[0.0, -0.0, 1.0, -1.0, 0.5, 0.0, -0.0]] * 4, dtype="f4")
    extremes = np.array([-(1 << 63), (1 << 63) - 1, -1, 1, 0, 3, -3, 17], dtype="i8")
    b = np.tile(extremes, (7, 1))
    for mode in range(4):
        assert environment.mode(mode) == 0
        environment.clear()
        expected = execute(control, p, b)
        flags = environment.flags()
        environment.clear()
        actual = execute(candidate, p, b)
        assert environment.flags() == flags
        np.testing.assert_array_equal(actual.view("u4"), expected.view("u4"))
    assert environment.mode(0) == 0


def test_nonzero_list_capacity_retains_identical_full_recurrence(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_GENERALIZE_BEFORE_FUSE", "1")
    text = source(k=4097)
    control = lower_to_llvm_ir(text, workdir=tmp_path / "control", features={RECTANGULAR_FEATURE})
    candidate = lower_to_llvm_ir(text, workdir=tmp_path / "candidate", features={RECTANGULAR_FEATURE, INDEX_FEATURE})
    assert RUNTIME_SYMBOL not in candidate
    assert candidate == control


def test_nonzero_workspace_total_capacity_keeps_remaining_contraction_full(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_GENERALIZE_BEFORE_FUSE", "1")
    functions = [source(k=4096).replace("@forward", "@first"), source(k=4096).replace("@forward", "@second"), source()]
    text = "module {" + "\n".join(item[len("module {") : -1] for item in functions) + "}"
    llvm = lower_to_llvm_ir(text, workdir=tmp_path, features={RECTANGULAR_FEATURE, INDEX_FEATURE})
    calls = [line for line in llvm.splitlines() if "call i32 @" + RUNTIME_SYMBOL + "(" in line]
    assert len(calls) == 2
