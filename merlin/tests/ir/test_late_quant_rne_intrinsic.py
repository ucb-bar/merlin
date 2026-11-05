"""Earlier shared roundeven legalization still admits exact CPU conversion."""

import ctypes
import subprocess

import numpy as np
import pytest

from merlin.llvmlower.late_quant_rne import rewrite
from merlin.llvmlower.toolchain import clang


def source(bits=8):
    lower, upper = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return f"""define i{bits} @quant(float %x) {{
  %lo = call float @llvm.maximum.f32(float %x, float {float(lower)})
  %hi = call float @llvm.minimum.f32(float %lo, float {float(upper)})
  %rounded = call float @llvm.roundeven.f32(float %hi)
  %result = fptosi float %rounded to i{bits}
  ret i{bits} %result
}}
declare float @llvm.maximum.f32(float, float)
declare float @llvm.minimum.f32(float, float)
declare float @llvm.roundeven.f32(float)
"""


@pytest.mark.parametrize("bits", [8, 16, 24])
@pytest.mark.parametrize("combine", [False, True])
def test_actual_intrinsic_bounds_select_explicit_cpu_conversion(bits, combine):
    text = source(bits)
    selected, report = rewrite(text, host_isa="rv64gc", combine_clamp=combine)
    assert len(report["routes"]) == 1
    assert report["routes"][0]["source_form"] == "bounded_roundeven_conversion"
    assert "fcvt.w.s $0," in selected
    assert ("fmax.s ft0" in selected) == combine
    assert f"trunc i32 %merlin.rne.0 to i{bits}" in selected
    assert rewrite(text)[0] == text


@pytest.mark.parametrize(
    "change",
    ["unbounded", "wrong_low", "wrong_high", "fast", "minnum", "wrong_round", "wrong_type", "wide", "metadata"],
)
def test_incomplete_intrinsic_domains_refuse(change):
    text = source()
    changes = {
        "unbounded": ("float %rounded to i8", "float %x to i8"),
        "wrong_low": ("float -128.0", "float -129.0"),
        "wrong_high": ("float 127.0", "float 128.0"),
        "fast": ("call float @llvm.roundeven", "call fast float @llvm.roundeven"),
        "minnum": ("llvm.minimum", "llvm.minnum"),
        "wrong_round": ("llvm.roundeven", "llvm.round"),
        "wrong_type": ("fptosi float %rounded to i8", "fptoui float %rounded to i8"),
        "wide": ("i8", "i32"),
        "metadata": ("fptosi float %rounded to i8", "fptosi float %rounded to i8, !dbg !1"),
    }
    text = text.replace(*changes[change])
    assert rewrite(text, host_isa="rv64gc")[0] == text


def test_other_rounding_uses_keep_their_exact_source_instructions():
    text = source().replace("  ret i8", "  %other = fadd float %rounded, %x\n  ret i8")
    selected, report = rewrite(text, host_isa="rv64gc")
    assert len(report["routes"]) == 1
    assert "%other = fadd float %rounded, %x" in selected
    assert "%rounded = call float @llvm.roundeven.f32(float %hi)" in selected


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("constant_first", [False, True])
def test_both_literal_clamp_orders_and_operand_positions_are_derived(reverse, constant_first):
    text = source()
    if reverse:
        text = text.replace("maximum.f32(float %x, float -128.0)", "minimum.f32(float %x, float 127.0)")
        text = text.replace("minimum.f32(float %lo, float 127.0)", "maximum.f32(float %lo, float -128.0)")
    if constant_first:
        for value, bound in [("%x", "127.0" if reverse else "-128.0"), ("%lo", "-128.0" if reverse else "127.0")]:
            text = text.replace(f"float {value}, float {bound}", f"float {bound}, float {value}")
    selected, report = rewrite(text, host_isa="rv64gc", combine_clamp=True)
    assert len(report["routes"]) == 1
    assert report["routes"][0]["raw_input"] == "%x"
    assert '"=r,f,f,f,~{ft0}"(float %x,' in selected


def test_normal_lowering_and_automatic_rounding_reach_the_cpu_legalizer(tmp_path):
    from merlin.llvmlower.lower import lower_model
    from merlin.tests.ir.test_automatic_rounding import residual
    from merlin.xdsl_dialects._common import text

    program = text(residual(lo=-128, hi=127), generic=True)
    built = lower_model(program, tmp_path / "ordinary", targets=())
    assert built.stats["automatic_host_rounding"]["rewrites"] == 1
    selected, report = rewrite(built.ll_path.read_text(), host_isa="rv64gc", combine_clamp=True)
    assert len(report["routes"]) == 1
    assert report["routes"][0]["source_form"] == "bounded_roundeven_conversion"
    assert "fcvt.w.s" in selected


@pytest.mark.parametrize("bits", [8, 16])
def test_portable_intrinsic_matches_independent_words_at_rounding_boundaries(tmp_path, bits):
    text = source(bits)
    selected, report = rewrite(text, host_isa="portable")
    assert len(report["routes"]) == 1
    lower, upper = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    half = np.arange(lower - 2, upper + 3, dtype=np.float32) + np.float32(0.5)
    rng = np.random.default_rng(841)
    random = rng.integers(0, 2**32, 20000, dtype=np.uint32).view(np.float32)
    values = np.concatenate(
        [
            half,
            np.nextafter(half, np.float32(np.inf)),
            np.nextafter(half, np.float32(-np.inf)),
            random[~np.isnan(random)],
            np.array([-np.inf, np.inf, 0.0, -0.0, 1e-40, -1e-40], np.float32),
        ]
    )
    expected = np.rint(np.clip(values, lower, upper)).astype(np.int64)
    for label, program in [("control", text), ("portable", selected)]:
        llvm = tmp_path / (label + ".ll")
        library = tmp_path / (label + str(bits) + ".so")
        llvm.write_text(program)
        subprocess.run(
            [clang(), "-O2", "-shared", "-fPIC", str(llvm), "-lm", "-o", str(library)], check=True, capture_output=True
        )
        function = ctypes.CDLL(str(library)).quant
        function.argtypes = [ctypes.c_float]
        function.restype = ctypes.c_int8 if bits == 8 else ctypes.c_int16
        actual = np.fromiter((function(float(value)) for value in values), dtype=np.int64)
        np.testing.assert_array_equal(actual, expected)
