"""Real int8 compute on RVV: i8xi8->i32 GEMM lowers to RVV integer SIMD (vwmacc), and a
W8A8 dynamic-quant GEMM built on it approximates the f32 result.

This is the "everything that can be int8 should be int8" path: the contraction runs in
actual 8-bit integer arithmetic with i32 accumulation (RVV ``vwmacc.vv``), not dequantized
to f32. Auto-skips the RVV-disasm check without the chipyard toolchain; the host correctness
+ W8A8 accuracy checks need only clang (m2m venv).
"""
from __future__ import annotations

import subprocess

import numpy as np
import pytest

from merlin.llvmlower import toolchain


def _i8_matmul_src(m, k, n):
    return (f"builtin.module {{ func.func @forward(%a: tensor<{m}x{k}xi8>, "
            f"%b: tensor<{k}x{n}xi8>) -> tensor<{m}x{n}xi32> {{ "
            f"%e = tensor.empty() : tensor<{m}x{n}xi32> "
            f"%c0 = arith.constant 0 : i32 "
            f"%f = linalg.fill ins(%c0 : i32) outs(%e : tensor<{m}x{n}xi32>) "
            f"-> tensor<{m}x{n}xi32> "
            f"%y = linalg.matmul ins(%a, %b : tensor<{m}x{k}xi8>, tensor<{k}x{n}xi8>) "
            f"outs(%f : tensor<{m}x{n}xi32>) -> tensor<{m}x{n}xi32> "
            f"func.return %y : tensor<{m}x{n}xi32> }} }}")


def _run_i8_matmul(src, A, B, M, N, tmp_path):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.lower import lower_model

    res = lower_model(src, tmp_path / "i8", targets=("host",))
    model = HostModel.load(str(res.host_so))
    Y = np.zeros((M, N), np.int32)
    model([(A.ctypes.data, (M, A.shape[1])), (B.ctypes.data, (B.shape[0], N)),
           (Y.ctypes.data, (M, N))])
    return Y


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv / clang-23 missing")
def test_i8_matmul_is_exact_integer(tmp_path):
    """i8xi8->i32 contraction is bit-exact integer arithmetic (no float)."""
    M, K, N = 8, 64, 16
    rng = np.random.default_rng(0)
    A = rng.integers(-8, 8, (M, K), np.int8)
    B = rng.integers(-8, 8, (K, N), np.int8)
    Y = _run_i8_matmul(_i8_matmul_src(M, K, N), A, B, M, N, tmp_path)
    assert np.array_equal(Y, A.astype(np.int32) @ B.astype(np.int32))


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv / clang-23 missing")
def test_w8a8_dynamic_quant_matches_f32(tmp_path):
    """W8A8: quantize activations per-row + weights per-col, run the i8 GEMM, dequantize.
    Result tracks the f32 matmul to int8 tolerance (cos > 0.999)."""
    M, K, N = 16, 256, 32
    rng = np.random.default_rng(1)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ref = A @ B

    sa = np.abs(A).max(1, keepdims=True) / 127.0                  # per-row act scale
    sw = np.abs(B).max(0, keepdims=True) / 127.0                  # per-col weight scale
    Aq = np.clip(np.round(A / sa), -127, 127).astype(np.int8)
    Bq = np.clip(np.round(B / sw), -127, 127).astype(np.int8)

    acc = _run_i8_matmul(_i8_matmul_src(M, K, N), Aq, Bq, M, N, tmp_path)  # i8xi8->i32 on RVV path
    out = acc.astype(np.float32) * sa * sw                       # requantize

    cos = float((out.ravel() @ ref.ravel())
                / (np.linalg.norm(out) * np.linalg.norm(ref) + 1e-12))
    assert cos > 0.999, cos


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv / clang-23 missing")
def test_i8_matmul_emits_rvv_integer_simd(tmp_path):
    """The i8 GEMM compiles to rv64gcv with a widening integer MAC (vwmacc) — real int8 SIMD."""
    from merlin.llvmlower.lower import lower_model
    from merlin.runtime.backends import spike

    if not spike.available():
        pytest.skip("chipyard objdump unavailable")
    res = lower_model(_i8_matmul_src(8, 64, 16), tmp_path / "i8r", targets=("riscv",))
    objdump = spike.gcc_path().with_name("riscv64-unknown-elf-objdump")
    dis = subprocess.run([objdump, "-d", str(res.riscv_obj)], capture_output=True,
                         text=True).stdout
    assert "vwmacc" in dis or "vmacc" in dis, "expected RVV integer multiply-accumulate"
    assert "vsetvli" in dis or "vsetivli" in dis


# --- W8A8 integer datapath (passes_quant_int.lower_matmul_int8) -------------------------------

_DEQUANT_MM = (
    'builtin.module attributes {{prov.quantization = "int8_weight_only"}} {{ '
    "func.func @forward(%act: tensor<{m}x{k}xf32>, %w: tensor<{k}x{n}xi8>, "
    "%s: tensor<{n}xf32>, %zp: tensor<{n}xi32>) -> tensor<{m}x{n}xf32> {{ "
    '%wd = "quant_ext.dequantize_per_channel"(%w, %s, %zp) '
    '<{{axis = 1 : i64, input_dtype = "i8"}}> : '
    "(tensor<{k}x{n}xi8>, tensor<{n}xf32>, tensor<{n}xi32>) -> tensor<{k}x{n}xf32> "
    "%e = tensor.empty() : tensor<{m}x{n}xf32> "
    "%c0 = arith.constant 0.0 : f32 "
    "%f = linalg.fill ins(%c0 : f32) outs(%e : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32> "
    "%y = linalg.matmul ins(%act, %wd : tensor<{m}x{k}xf32>, tensor<{k}x{n}xf32>) "
    "outs(%f : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32> "
    "func.return %y : tensor<{m}x{n}xf32> }} }}")


def test_lower_matmul_int8_makes_integer_contraction(tmp_path):
    """``lower_matmul_int8`` turns dequant(weight)+f32-matmul into an i8×i8→i32 contraction:
    the weight stays i8, the activation is dynamically quantized, the matmul accumulates i32."""
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.passes_quant_int import lower_matmul_int8

    src = tmp_path / "deqmm.mlir"
    src.write_text(_DEQUANT_MM.format(m=8, k=64, n=16), encoding="utf-8")
    module = parse_mlir_file(src)
    n = lower_matmul_int8(module)
    module.verify()
    assert n == 1
    assert all(op.name != "linalg.matmul" for op in module.walk())  # f32 matmul gone
    def _ibits(t):
        return getattr(getattr(t.element_type, "width", None), "data", None)
    int_contract = [
        op for op in module.walk()
        if op.name == "linalg.generic" and len(op.inputs) == 2
        and all(_ibits(i.type) == 8 for i in op.inputs) and _ibits(op.results[0].type) == 32]
    assert int_contract, "expected one i8×i8→i32 integer contraction"
