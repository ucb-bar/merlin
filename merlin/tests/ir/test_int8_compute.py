from __future__ import annotations

import numpy as np
import pytest

from merlin.llvmlower import toolchain


def _i8_matmul_src(m, k, n):
    return f"builtin.module {{ func.func @forward(%a: tensor<{m}x{k}xi8>, %b: tensor<{k}x{n}xi8>) -> tensor<{m}x{n}xi32> {{ %e = tensor.empty() : tensor<{m}x{n}xi32> %c0 = arith.constant 0 : i32 %f = linalg.fill ins(%c0 : i32) outs(%e : tensor<{m}x{n}xi32>) -> tensor<{m}x{n}xi32> %y = linalg.matmul ins(%a, %b : tensor<{m}x{k}xi8>, tensor<{k}x{n}xi8>) outs(%f : tensor<{m}x{n}xi32>) -> tensor<{m}x{n}xi32> func.return %y : tensor<{m}x{n}xi32> }} }}"


def _run_i8_matmul(src, A, B, M, N, tmp_path):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.lower import lower_model

    res = lower_model(src, tmp_path / "i8", targets=("host",))
    model = HostModel.load(str(res.host_so))
    Y = np.zeros((M, N), np.int32)
    model([(A.ctypes.data, (M, A.shape[1])), (B.ctypes.data, (B.shape[0], N)), (Y.ctypes.data, (M, N))])
    return Y


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv / clang-23 missing")
def test_i8_matmul_is_exact_integer(tmp_path):
    """i8xi8->i32 contraction is bit-exact integer arithmetic (no float)."""
    M, K, N = (8, 64, 16)
    rng = np.random.default_rng(0)
    A = rng.integers(-8, 8, (M, K), np.int8)
    B = rng.integers(-8, 8, (K, N), np.int8)
    A[0, 0], B[0, 0] = (-128, 127)
    Y = _run_i8_matmul(_i8_matmul_src(M, K, N), A, B, M, N, tmp_path)
    assert np.array_equal(Y, A.astype(np.int32) @ B.astype(np.int32))


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv / clang-23 missing")
def test_w8a8_dynamic_quant_matches_f32(tmp_path):
    """W8A8: quantize activations per-row + weights per-col, run the i8 GEMM, dequantize.
    Result tracks the f32 matmul to int8 tolerance (cos > 0.999)."""
    M, K, N = (16, 256, 32)
    rng = np.random.default_rng(1)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ref = A @ B
    sa = np.abs(A).max(1, keepdims=True) / 127.0
    sw = np.abs(B).max(0, keepdims=True) / 127.0
    Aq = np.clip(np.round(A / sa), -127, 127).astype(np.int8)
    Bq = np.clip(np.round(B / sw), -127, 127).astype(np.int8)
    acc = _run_i8_matmul(_i8_matmul_src(M, K, N), Aq, Bq, M, N, tmp_path)
    out = acc.astype(np.float32) * sa * sw
    cos = float(out.ravel() @ ref.ravel() / (np.linalg.norm(out) * np.linalg.norm(ref) + 1e-12))
    assert cos > 0.999, cos


_DEQUANT_MM = 'builtin.module attributes {{prov.quantization = "int8_weight_only"}} {{ func.func @forward(%act: tensor<{m}x{k}xf32>, %w: tensor<{k}x{n}xi8>, %s: tensor<{n}xf32>, %zp: tensor<{n}xi32>) -> tensor<{m}x{n}xf32> {{ %wd = "quant_ext.dequantize_per_channel"(%w, %s, %zp) <{{axis = 1 : i64, input_dtype = "i8"}}> : (tensor<{k}x{n}xi8>, tensor<{n}xf32>, tensor<{n}xi32>) -> tensor<{k}x{n}xf32> %e = tensor.empty() : tensor<{m}x{n}xf32> %c0 = arith.constant 0.0 : f32 %f = linalg.fill ins(%c0 : f32) outs(%e : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32> %y = linalg.matmul ins(%act, %wd : tensor<{m}x{k}xf32>, tensor<{k}x{n}xf32>) outs(%f : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32> func.return %y : tensor<{m}x{n}xf32> }} }}'


def test_lower_matmul_int8_makes_integer_contraction(tmp_path):
    """``lower_matmul_int8`` turns dequant(weight)+f32-matmul into an i8×i8→i32 contraction:
    the weight stays i8, the activation is dynamically quantized, the matmul accumulates i32."""
    from xdsl.dialects.arith import Arith
    from xdsl.dialects.linalg import Linalg
    from xdsl.dialects.linalg.ops import MatmulOp
    from xdsl.dialects.tensor import Tensor
    from xdsl.parser import Parser
    from xdsl.utils.exceptions import ParseError, VerifyException

    from merlin.frontends.linalg_mlir import make_context, parse_mlir_file
    from merlin.llvmlower.passes_quant_int import lower_matmul_int8
    from merlin.targetgen.contract.linalg_iface import make_linalg_context
    from merlin.xdsl_dialects._common import make_context as core_context
    from merlin.xdsl_dialects._common import text

    source = _i8_matmul_src(2, 4, 3)
    for context in (make_context(), core_context(Arith, Linalg, Tensor), make_linalg_context()):
        parsed = Parser(context, source).parse_module()
        parsed.verify()
        matmul = next((op for op in parsed.walk() if isinstance(op, MatmulOp)))
    for cast in ("cast_unsigned", "unsupported_cast"):
        unsupported = source.replace("linalg.matmul ins", f"linalg.matmul {{cast = #linalg.type_fn<{cast}>}} ins")
        with pytest.raises(ParseError, match="default or signed cast"):
            Parser(make_context(), unsupported).parse_module()
    signed = source.replace("linalg.matmul ins", "linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins")
    Parser(make_context(), signed).parse_module().verify()
    explicit = Parser(make_context(), source).parse_module()
    matmul = next((op for op in explicit.walk() if isinstance(op, MatmulOp)))
    matmul.detach_region(0)
    matmul.add_region(MatmulOp.get_hidden_region(matmul.inputs, matmul.outputs))
    unchanged = Parser(make_context(), text(explicit, generic=True)).parse_module()
    with pytest.raises(VerifyException):
        unchanged.verify()
    src = tmp_path / "deqmm.mlir"
    src.write_text(_DEQUANT_MM.format(m=8, k=64, n=16), encoding="utf-8")
    module = parse_mlir_file(src)
    n = lower_matmul_int8(module)
    module.verify()
    assert n == 1
    assert all((op.name != "linalg.matmul" for op in module.walk()))

    def _ibits(t):
        return getattr(getattr(t.element_type, "width", None), "data", None)

    int_contract = [
        op
        for op in module.walk()
        if op.name == "linalg.generic"
        and len(op.inputs) == 2
        and all((_ibits(i.type) == 8 for i in op.inputs))
        and (_ibits(op.results[0].type) == 32)
    ]
    assert int_contract, "expected one i8×i8→i32 integer contraction"


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv / clang-23 missing")
@pytest.mark.parametrize("case", ["tiny"])
def test_dynamic_matmul_quantization_is_bit_exact_torchao(case, tmp_path):
    """The real contraction path uses TorchAO's scale floor and reciprocal spelling exactly.

    An identity int8 weight exposes the activation code after requantization.  The expected scale
    and code come from TorchAO itself, not from another transcription in this test.
    """
    from merlin.common.paths import merlin_dir
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.lower import lower_model
    from merlin.llvmlower.passes_quant_int import lower_matmul_int8
    from merlin.xdsl_dialects._common import text as to_text

    ref = np.load(merlin_dir() / "tests" / "data" / "torchao_affine" / "reference.npz")
    act = np.ascontiguousarray(ref[f"{case}::x"])
    want_scale = np.asarray(ref[f"{case}::scale"], dtype=np.float32).reshape(1, 1)
    want_q = np.asarray(ref[f"{case}::q"], dtype=np.int8)
    k = act.shape[1]
    src = tmp_path / f"torchao_{case}.mlir"
    src.write_text(_DEQUANT_MM.format(m=1, k=k, n=k), encoding="utf-8")
    module = parse_mlir_file(src)
    assert lower_matmul_int8(module) == 1
    result = lower_model(to_text(module), tmp_path / f"torchao_{case}_host", targets=("host",))
    model = HostModel.load(str(result.host_so))
    weight = np.eye(k, dtype=np.int8)
    weight_scale = np.ones((k,), dtype=np.float32)
    zero_point = np.zeros((k,), dtype=np.int32)
    out = np.full((1, k), np.nan, dtype=np.float32)
    model(
        [
            (act.ctypes.data, act.shape),
            (weight.ctypes.data, weight.shape),
            (weight_scale.ctypes.data, weight_scale.shape),
            (zero_point.ctypes.data, zero_point.shape),
            (out.ctypes.data, out.shape),
        ]
    )
    expected = want_q.astype(np.float32) * want_scale
    assert np.array_equal(out.view(np.uint32), expected.view(np.uint32)), (
        case,
        int(np.count_nonzero(out.view(np.uint32) != expected.view(np.uint32))),
    )


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv / clang-23 missing")
def test_zero_activation_stays_finite_after_quant_round_fusion(tmp_path):
    """Fresh-process host reproducer for the TinyLlama all-NaN failure.

    A zero activation must quantize to integer zero and requantize to finite float zero even after
    the round/clamp/convert chain is inlined for vectorization.
    """
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.lower import lower_model
    from merlin.llvmlower.passes_quant_int import lower_matmul_int8
    from merlin.llvmlower.quant_round import fuse_round_clamp_convert
    from merlin.xdsl_dialects._common import text as to_text

    m, k, n = (2, 8, 4)
    src = tmp_path / "zero_matmul.mlir"
    src.write_text(_DEQUANT_MM.format(m=m, k=k, n=n), encoding="utf-8")
    module = parse_mlir_file(src)
    assert lower_matmul_int8(module) == 1
    assert fuse_round_clamp_convert(module) == 1
    result = lower_model(to_text(module), tmp_path / "zero_matmul_host", targets=("host",))
    act = np.zeros((m, k), np.float32)
    weight = np.arange(k * n, dtype=np.int8).reshape(k, n) - np.int8(16)
    scale = np.full((n,), 0.03125, np.float32)
    zero_point = np.zeros((n,), np.int32)
    out = np.full((m, n), np.nan, np.float32)
    HostModel.load(str(result.host_so))(
        [
            (act.ctypes.data, act.shape),
            (weight.ctypes.data, weight.shape),
            (scale.ctypes.data, scale.shape),
            (zero_point.ctypes.data, zero_point.shape),
            (out.ctypes.data, out.shape),
        ]
    )
    assert np.array_equal(out, np.zeros_like(out)), out
