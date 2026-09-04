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


# conv: a 7-iterator linalg.generic with a stride-16 affine input map (d2*16+d5, d3*16+d6),
# f32 activation + f32 weight (torchao leaves conv weights f32) -> i8×i8→i32 + requant.
_CONV_MOD = (
    "builtin.module {{ func.func @forward(%x: tensor<1x3x64x64xf32>, "
    "%w: tensor<{oc}x3x16x16xf32>) -> tensor<1x{oc}x4x4xf32> {{ "
    "%e = tensor.empty() : tensor<1x{oc}x4x4xf32> "
    "%c0 = arith.constant 0.0 : f32 "
    "%f = linalg.fill ins(%c0 : f32) outs(%e : tensor<1x{oc}x4x4xf32>) -> tensor<1x{oc}x4x4xf32> "
    "%y = linalg.generic {{indexing_maps = ["
    "affine_map<(d0,d1,d2,d3,d4,d5,d6)->(d0,d4,d2*16+d5,d3*16+d6)>, "
    "affine_map<(d0,d1,d2,d3,d4,d5,d6)->(d1,d4,d5,d6)>, "
    "affine_map<(d0,d1,d2,d3,d4,d5,d6)->(d0,d1,d2,d3)>], "
    "iterator_types = [\"parallel\",\"parallel\",\"parallel\",\"parallel\","
    "\"reduction\",\"reduction\",\"reduction\"], "
    'prov.op = "conv2d"}} '
    "ins(%x, %w : tensor<1x3x64x64xf32>, tensor<{oc}x3x16x16xf32>) "
    "outs(%f : tensor<1x{oc}x4x4xf32>) {{ "
    "^bb(%a: f32, %b: f32, %o: f32): %m = arith.mulf %a, %b : f32 "
    "%s = arith.addf %o, %m : f32 linalg.yield %s : f32 }} "
    "-> tensor<1x{oc}x4x4xf32> func.return %y : tensor<1x{oc}x4x4xf32> }} }}")


def test_lower_conv_int8_makes_integer_conv(tmp_path):
    """``lower_conv_int8`` turns an f32 conv generic into an i8×i8→i32 contraction, dynamically
    quantizing both operands and keeping the EXACT stride-affine maps (so the conv structure —
    7 iterators, compound input map — survives, unlike the matmul rebuild which drops it)."""
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.passes_quant_int import lower_conv_int8

    src = tmp_path / "conv.mlir"
    src.write_text(_CONV_MOD.format(oc=192), encoding="utf-8")
    module = parse_mlir_file(src)
    n = lower_conv_int8(module)
    module.verify()
    assert n == 1
    def _ibits(t):
        return getattr(getattr(t.element_type, "width", None), "data", None)
    i8conv = [
        op for op in module.walk()
        if op.name == "linalg.generic" and len(op.inputs) == 2
        and op.indexing_maps.data[0].data.num_dims == 7          # conv iterator space preserved
        and all(_ibits(i.type) == 8 for i in op.inputs) and _ibits(op.results[0].type) == 32]
    assert i8conv, "expected one i8×i8→i32 conv with the stride-affine maps intact"


# softmax-shaped: a (S - rowmax) subtraction feeding the exp (the signature lower_softmax_int
# requires). %m is the per-row max (pass zeros in the numeric test so sub == x).
_EXP_MOD = (
    "builtin.module {{ func.func @forward(%x: tensor<{m}x{l}xf32>, %mx: tensor<{m}xf32>) "
    "-> tensor<{m}x{l}xf32> {{ "
    "%se = tensor.empty() : tensor<{m}x{l}xf32> "
    "%sub = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, "
    "affine_map<(d0,d1)->(d0)>, affine_map<(d0,d1)->(d0,d1)>], "
    "iterator_types = [\"parallel\",\"parallel\"]}} "
    "ins(%x, %mx : tensor<{m}x{l}xf32>, tensor<{m}xf32>) outs(%se : tensor<{m}x{l}xf32>) {{ "
    "^bb(%a: f32, %mm: f32, %o: f32): %s = arith.subf %a, %mm : f32 linalg.yield %s : f32 }} "
    "-> tensor<{m}x{l}xf32> "
    "%e = tensor.empty() : tensor<{m}x{l}xf32> "
    "%r = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, "
    "affine_map<(d0,d1)->(d0,d1)>], iterator_types = [\"parallel\",\"parallel\"]}} "
    "ins(%sub : tensor<{m}x{l}xf32>) outs(%e : tensor<{m}x{l}xf32>) {{ "
    "^bb(%a: f32, %o: f32): %ex = math.exp %a : f32 linalg.yield %ex : f32 }} "
    "-> tensor<{m}x{l}xf32> func.return %r : tensor<{m}x{l}xf32> }} }}")


def test_lower_softmax_int_removes_math_exp(tmp_path):
    """``lower_softmax_int`` replaces the softmax ``math.exp`` with an integer (I-BERT) i-exp:
    no ``math.exp`` remains and an integer (i32/i64) exp body is emitted."""
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.passes_quant_int import lower_softmax_int

    src = tmp_path / "exp.mlir"
    src.write_text(_EXP_MOD.format(m=4, l=8), encoding="utf-8")
    module = parse_mlir_file(src)
    n = lower_softmax_int(module)
    module.verify()
    assert n == 1
    assert sum(1 for op in module.walk() if op.name == "math.exp") == 0
    # the integer exp body uses integer multiply-accumulate / shift ops
    assert any(op.name in ("arith.muli", "arith.shrsi") for op in module.walk())


# SiLU sigmoid generic (logistic 1/(1+exp(-x)), tagged prov.op = sigmoid).
_SIGMOID_MOD = (
    "builtin.module {{ func.func @forward(%x: tensor<{m}x{l}xf32>) -> tensor<{m}x{l}xf32> {{ "
    "%e = tensor.empty() : tensor<{m}x{l}xf32> "
    "%r = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, "
    "affine_map<(d0,d1)->(d0,d1)>], iterator_types = [\"parallel\",\"parallel\"]}} "
    "ins(%x : tensor<{m}x{l}xf32>) outs(%e : tensor<{m}x{l}xf32>) "
    'attrs = {{prov.op = "sigmoid"}} {{ '
    "^bb(%a: f32, %o: f32): %n = arith.negf %a : f32 %ex = math.exp %n : f32 "
    "%c1 = arith.constant 1.0 : f32 %d = arith.addf %c1, %ex : f32 "
    "%s = arith.divf %c1, %d : f32 linalg.yield %s : f32 }} "
    "-> tensor<{m}x{l}xf32> func.return %r : tensor<{m}x{l}xf32> }} }}")


def test_lower_silu_int_removes_math_exp(tmp_path):
    """``lower_silu_int`` replaces the logistic ``sigmoid`` (``math.exp``) with the integer
    I-BERT exp + an f32 logistic — no ``math.exp`` remains, integer poly ops appear."""
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.passes_quant_int import lower_silu_int

    src = tmp_path / "sig.mlir"
    src.write_text(_SIGMOID_MOD.format(m=4, l=8), encoding="utf-8")
    module = parse_mlir_file(src)
    n = lower_silu_int(module)
    module.verify()
    assert n == 1
    assert sum(1 for op in module.walk() if op.name == "math.exp") == 0
    assert any(op.name in ("arith.muli", "arith.shrsi") for op in module.walk())


_RSQRT_MOD = (
    "builtin.module {{ func.func @forward(%x: tensor<{m}x{l}xf32>) -> tensor<{m}x{l}xf32> {{ "
    "%e = tensor.empty() : tensor<{m}x{l}xf32> "
    "%r = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, "
    "affine_map<(d0,d1)->(d0,d1)>], iterator_types = [\"parallel\",\"parallel\"]}} "
    "ins(%x : tensor<{m}x{l}xf32>) outs(%e : tensor<{m}x{l}xf32>) {{ "
    "^bb(%a: f32, %o: f32): %q = math.rsqrt %a : f32 linalg.yield %q : f32 }} "
    "-> tensor<{m}x{l}xf32> func.return %r : tensor<{m}x{l}xf32> }} }}")


def test_lower_rsqrt_int_removes_math_rsqrt(tmp_path):
    """``lower_rsqrt_int`` replaces ``math.rsqrt`` with the fast-inverse-sqrt (bitcast + integer
    sub/shift + f32 Newton) — no ``math.rsqrt`` (no libm), bit-hack integer ops appear."""
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.passes_quant_int import lower_rsqrt_int

    src = tmp_path / "rsq.mlir"
    src.write_text(_RSQRT_MOD.format(m=4, l=8), encoding="utf-8")
    module = parse_mlir_file(src)
    n = lower_rsqrt_int(module)
    module.verify()
    assert n == 1
    assert sum(1 for op in module.walk() if op.name == "math.rsqrt") == 0
    assert any(op.name == "arith.bitcast" for op in module.walk())


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv / clang-23 missing")
def test_silu_sigmoid_matches_float(tmp_path):
    """The integer SiLU sigmoid tracks the f32 logistic over mixed-sign inputs (cos > 0.999)."""
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.passes_quant_int import lower_silu_int
    from merlin.llvmlower.lower import lower_model
    from merlin.llvmlower.abi import HostModel
    from merlin.xdsl_dialects._common import text as to_text

    M, L = 4, 16
    src = tmp_path / "sig.mlir"
    src.write_text(_SIGMOID_MOD.format(m=M, l=L), encoding="utf-8")
    module = parse_mlir_file(src)
    lower_silu_int(module)
    res = lower_model(to_text(module), tmp_path / "b", targets=("host",))
    hm = HostModel.load(str(res.host_so))
    rng = np.random.default_rng(0)
    xs = (rng.standard_normal((M, L)) * 4).astype(np.float32)            # mixed sign
    out = np.zeros((M, L), np.float32)
    hm([(xs.ctypes.data, (M, L)), (out.ctypes.data, (M, L))])
    ref = 1.0 / (1.0 + np.exp(-xs))
    cos = float((out.ravel() @ ref.ravel()) / (np.linalg.norm(out) * np.linalg.norm(ref) + 1e-12))
    assert cos > 0.999, cos


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv / clang-23 missing")
def test_integer_exp_matches_float_exp(tmp_path):
    """The integer i-exp tracks float exp for x<=0 (cos > 0.99 — I-BERT 2nd-order poly)."""
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.passes_quant_int import lower_softmax_int
    from merlin.llvmlower.lower import lower_model
    from merlin.llvmlower.abi import HostModel
    from merlin.xdsl_dialects._common import text as to_text

    M, L = 4, 8
    src = tmp_path / "exp.mlir"
    src.write_text(_EXP_MOD.format(m=M, l=L), encoding="utf-8")
    module = parse_mlir_file(src)
    lower_softmax_int(module)
    res = lower_model(to_text(module), tmp_path / "b", targets=("host",))
    hm = HostModel.load(str(res.host_so))
    rng = np.random.default_rng(0)
    xs = (-np.abs(rng.standard_normal((M, L))) * 3).astype(np.float32)   # x <= 0
    mx = np.zeros((M,), np.float32)                                      # rowmax 0 -> sub == xs
    out = np.zeros((M, L), np.float32)
    hm([(xs.ctypes.data, (M, L)), (mx.ctypes.data, (M,)), (out.ctypes.data, (M, L))])
    ref = np.exp(xs)
    cos = float((out.ravel() @ ref.ravel()) / (np.linalg.norm(out) * np.linalg.norm(ref) + 1e-12))
    assert cos > 0.99, cos


def _run_iexp(tmp_path, xs: np.ndarray) -> np.ndarray:
    """Lower + compile the softmax i-exp and evaluate it on ``xs`` (rowmax passed as 0)."""
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.passes_quant_int import lower_softmax_int
    from merlin.llvmlower.lower import lower_model
    from merlin.llvmlower.abi import HostModel
    from merlin.xdsl_dialects._common import text as to_text

    M, L = xs.shape
    src = tmp_path / f"exp_{M}x{L}.mlir"
    src.write_text(_EXP_MOD.format(m=M, l=L), encoding="utf-8")
    module = parse_mlir_file(src)
    assert lower_softmax_int(module) == 1
    res = lower_model(to_text(module), tmp_path / f"b_{M}x{L}", targets=("host",))
    hm = HostModel.load(str(res.host_so))
    xs = np.ascontiguousarray(xs, dtype=np.float32)
    mx = np.zeros((M,), np.float32)
    out = np.zeros((M, L), np.float32)
    hm([(xs.ctypes.data, (M, L)), (mx.ctypes.data, (M,)), (out.ctypes.data, (M, L))])
    return out


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv / clang-23 missing")
def test_integer_exp_is_elementwise_not_row_scaled(tmp_path):
    """The i-exp of an element must not depend on the OTHER elements in its row.

    It used to. The exponent grid was ``max(-x)/127`` reduced over the row, and softmax's input
    carries the attention mask's -inf (clamped to -30) at every masked position — so the grid was
    set by the mask sentinel, not the data, and a causally-masked row quantized its real scores
    ~15x more coarsely than an unmasked one. Measured end-to-end on the small_llama W8A8
    recapture, that alone took the deviation from the host W8A8 reference from rel 0.0077 to
    0.0148, i.e. it was the single largest term in the W8A8 tier failure.

    Padding a row with masked positions is the sharpest form of the question: the padding's exp is
    zero to any precision, so it must leave every real entry's exp EXACTLY where it was.
    """
    real = np.array([0.0, -0.4, -1.3, -2.9], np.float32)
    plain = np.tile(real, (2, 1))                                  # 4 real entries
    padded = np.tile(np.concatenate([real, np.full(4, -np.inf, np.float32)]), (2, 1))
    got_plain = _run_iexp(tmp_path, plain)[:, :4]
    got_padded = _run_iexp(tmp_path, padded)
    assert np.array_equal(got_plain, got_padded[:, :4]), (got_plain, got_padded[:, :4])
    assert np.all(got_padded[:, 4:] == 0.0), got_padded[:, 4:]     # exp(-inf) -> 0


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv / clang-23 missing")
def test_integer_exp_per_element_relative_error(tmp_path):
    """i-exp tracks ``exp`` to better than 1% PER ELEMENT over the range softmax actually uses.

    The cosine check above cannot see this: cosine is dominated by the largest terms, and the
    row-scaled i-exp passed it at cos > 0.99 while individual terms were up to 12% off. The bound
    here is the 2nd-order polynomial's own accuracy floor (measured max 0.41%), so it holds the
    fixed-point choice (``_IEXP_K`` / ``_IEXP_SH``) to the approximation it claims to implement.
    """
    xs = np.linspace(-10.0, 0.0, 256, dtype=np.float32).reshape(8, 32)
    got = _run_iexp(tmp_path, xs)
    ref = np.exp(xs.astype(np.float64))
    rel = np.abs(got.astype(np.float64) - ref) / ref
    assert rel.max() < 1e-2, (rel.max(), xs.ravel()[int(np.argmax(rel))])
