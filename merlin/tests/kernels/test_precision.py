"""bf16 matmul f32-accumulation pass (``merlin-bf16-matmul-f32acc``).

A bf16 ``linalg.matmul`` accumulates in bf16 (rounding every partial sum); the pass rewrites
it to accumulate in f32 and round only the final result, matching hardware/torch. The
structural test runs everywhere xDSL is present; the numerical test compiles both forms and
shows the f32-accumulate result is far closer to a high-precision reference (auto-skips
without the toolchain).
"""
from __future__ import annotations

import numpy as np
import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")


def _bf16_matmul_src(m, k, n):
    return (f"builtin.module {{ func.func @forward(%a: tensor<{m}x{k}xbf16>, "
            f"%b: tensor<{k}x{n}xbf16>) -> tensor<{m}x{n}xbf16> {{ "
            f"%e = tensor.empty() : tensor<{m}x{n}xbf16> "
            f"%c = arith.constant 0.0 : bf16 "
            f"%f = linalg.fill ins(%c : bf16) outs(%e : tensor<{m}x{n}xbf16>) "
            f"-> tensor<{m}x{n}xbf16> "
            f"%r = linalg.matmul ins(%a, %b : tensor<{m}x{k}xbf16>, tensor<{k}x{n}xbf16>) "
            f"outs(%f : tensor<{m}x{n}xbf16>) -> tensor<{m}x{n}xbf16> "
            f"func.return %r : tensor<{m}x{n}xbf16> }} }}")


def _f32_to_bf16(x):
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    bias = ((u >> 16) & 1) + 0x7FFF              # round to nearest even
    return ((u + bias) >> 16).astype(np.uint16)


def _bf16_to_f32(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def test_pass_rewrites_bf16_matmul_to_f32_accumulation():
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.passes_xdsl import lower_bf16_matmul_f32acc

    m = parse_mlir_text(_bf16_matmul_src(4, 8, 6))
    assert lower_bf16_matmul_f32acc(m) == 1
    m.verify()
    names = [op.name for op in m.walk()]
    assert "linalg.matmul" not in names           # replaced
    # the f32-accumulating generic + the truncf-back generic
    assert names.count("linalg.generic") == 2
    assert any(op.name == "arith.extf" for op in m.walk())
    assert any(op.name == "arith.truncf" for op in m.walk())
    # the accumulator generic writes an f32 tensor
    gens = [op for op in m.walk() if op.name == "linalg.generic"]
    assert any(str(g.results[0].type.element_type) == "f32" for g in gens)


def _toolchain():
    from merlin.llvmlower import toolchain

    return toolchain.available()


@pytest.mark.skipif(not _toolchain(), reason="m2m venv / clang-23 missing")
def test_f32_accumulation_is_far_more_accurate(tmp_path):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.lower import lower_model
    from merlin.llvmlower.passes_xdsl import lower_bf16_matmul_f32acc
    from merlin.xdsl_dialects._common import text as to_text

    m, k, n = 8, 512, 64                          # long contraction: bf16 accumulation hurts
    rng = np.random.default_rng(0)
    A = _f32_to_bf16(rng.standard_normal((m, k)))
    B = _f32_to_bf16(rng.standard_normal((k, n)))
    ref = (_bf16_to_f32(A).astype(np.float64) @ _bf16_to_f32(B).astype(np.float64)
           ).astype(np.float32)

    def run(text, tag):
        res = lower_model(text, tmp_path / tag, targets=("host",))
        model = HostModel.load(str(res.host_so))
        out = np.zeros((m, n), np.uint16)
        model([(A.ctypes.data, (m, k)), (B.ctypes.data, (k, n)),
               (out.ctypes.data, (m, n))])
        return _bf16_to_f32(out)

    err_bf16 = float(np.abs(run(_bf16_matmul_src(m, k, n), "bf16acc") - ref).max())

    mod = parse_mlir_text(_bf16_matmul_src(m, k, n))
    lower_bf16_matmul_f32acc(mod)
    err_f32 = float(np.abs(run(to_text(mod), "f32acc") - ref).max())

    assert err_f32 < err_bf16 / 5                 # dramatically better
    assert err_f32 < 0.3                          # ~bf16 output ULP at this magnitude
    assert err_bf16 > 1.0                          # bf16 accumulation is genuinely lossy


# --- fp8 (float8_e4m3fn) weight decode: 1-byte storage -> f32 at load -------------------

def test_f8e4m3fn_decode_matches_reference():
    from merlin.runtime.dispatch_runtime import f8e4m3fn_to_f32

    # canonical e4m3fn byte patterns -> values (OCP: 1s/4e bias-7/3m, no inf, NaN=S.1111.111)
    bytes_ = np.array([0x3C, 0x38, 0xB8, 0x00, 0x80, 0x7E, 0xFE, 0x08], np.uint8)
    expect = [1.5, 1.0, -1.0, 0.0, -0.0, 448.0, -448.0, 2.0 ** -6]
    got = f8e4m3fn_to_f32(bytes_)
    assert np.allclose(got, expect, rtol=0, atol=0), (got, expect)
    # NaN encodings (S.1111.111)
    assert np.isnan(f8e4m3fn_to_f32(np.array([0x7F, 0xFF], np.uint8))).all()


# --- bool->float cast: sitofp(i1) (true -> -1.0) must become uitofp (true -> +1.0) -------

def _bool_mul_src():
    """`out = mask * float(bool)` exactly as model2MLIR emits it: sitofp on an i1."""
    return (
        "builtin.module { func.func @forward(%m: tensor<4xf32>, %b: tensor<4xi1>) "
        "-> tensor<4xf32> { "
        "%e = tensor.empty() : tensor<4xf32> "
        "%r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, "
        "affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "
        "iterator_types = [\"parallel\"]} ins(%m, %b : tensor<4xf32>, tensor<4xi1>) "
        "outs(%e : tensor<4xf32>) { "
        "^bb0(%mv: f32, %bv: i1, %o: f32): "
        "%bf = arith.sitofp %bv : i1 to f32 "
        "%p = arith.mulf %mv, %bf : f32 "
        "linalg.yield %p : f32 } -> tensor<4xf32> "
        "func.return %r : tensor<4xf32> } }")


def test_pass_rewrites_bool_sitofp_to_uitofp():
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.passes_xdsl import fix_bool_sitofp

    m = parse_mlir_text(_bool_mul_src())
    assert any(op.name == "arith.sitofp" for op in m.walk())
    assert fix_bool_sitofp(m) == 1
    m.verify()
    assert not any(op.name == "arith.sitofp" for op in m.walk())   # replaced
    assert any(op.name == "arith.uitofp" for op in m.walk())
    # a genuinely-signed (wider) sitofp must be left untouched
    m2 = parse_mlir_text(_bool_mul_src().replace("i1", "i32"))
    assert fix_bool_sitofp(m2) == 0


@pytest.mark.skipif(not _toolchain(), reason="m2m venv / clang-23 missing")
def test_bool_cast_sign_is_correct_after_fix(tmp_path):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.lower import lower_model
    from merlin.llvmlower.passes_xdsl import fix_bool_sitofp
    from merlin.xdsl_dialects._common import text as to_text

    FLT_MAX = np.float32(3.4028235e38)
    mask = np.array([-FLT_MAX, -FLT_MAX, 0.0, 0.0], np.float32)
    boolv = np.array([1, 0, 1, 0], np.bool_)

    def run(text, tag):
        res = lower_model(text, tmp_path / tag, targets=("host",))
        model = HostModel.load(str(res.host_so))
        out = np.zeros(4, np.float32)
        model([(mask.ctypes.data, (4,)), (boolv.ctypes.data, (4,)), (out.ctypes.data, (4,))])
        return out.copy()

    # unfixed: sitofp(true)=-1 -> -FLT_MAX * -1 = +FLT_MAX (wrong sign)
    bad = run(_bool_mul_src(), "sitofp")
    assert bad[0] > 0                              # sign flipped, as in the molmoact bug

    mod = parse_mlir_text(_bool_mul_src())
    fix_bool_sitofp(mod)
    good = run(to_text(mod), "uitofp")
    assert good[0] == -FLT_MAX                      # mask * 1.0, sign preserved (matches torch)
    assert good[1] == 0.0 and good[3] == 0.0
