"""The numpy evaluator of linalg-on-tensors: the independent oracle an open model is graded against.

Each case states a small module and the numbers numpy computes for it by hand, so a disagreement is
the evaluator's and not a shared mistake. The cases are the constructs a captured transformer uses and
a naive evaluator gets wrong: an exact int8 contraction, a reduction whose body carries two
accumulators (an arg-min), a gathering map (a window's ``stride * out + tap``), a rank-0 tensor, a
bf16 result, and a call the module leaves to its caller.
"""

from __future__ import annotations

import numpy as np
import pytest

from merlin.common import mlir_query as mq
from merlin.runtime import linalg_numpy as LN

_ID2 = "affine_map<(d0, d1) -> (d0, d1)>"


def _evaluate(text: str, *args, **kwargs):
    return LN.evaluate(mq.parse(text), list(args), **kwargs)


def test_an_int8_contraction_is_exact_at_its_accumulator_width() -> None:
    text = """
module {
  func.func @forward(%a: tensor<3x5xi8>, %b: tensor<5x4xi8>) -> tensor<3x4xi32> {
    %c0 = arith.constant 0 : i32
    %s = tensor.splat %c0 : tensor<3x4xi32>
    %m = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a, %b : tensor<3x5xi8>, tensor<5x4xi8>) outs(%s : tensor<3x4xi32>) {
    ^bb0(%x: i8, %y: i8, %acc: i32):
      %xe = arith.extsi %x : i8 to i32
      %ye = arith.extsi %y : i8 to i32
      %p = arith.muli %xe, %ye : i32
      %r = arith.addi %acc, %p : i32
      linalg.yield %r : i32
    } -> tensor<3x4xi32>
    return %m : tensor<3x4xi32>
  }
}
"""
    rng = np.random.default_rng(0)
    a = rng.integers(-128, 128, size=(3, 5), dtype=np.int8)
    b = rng.integers(-128, 128, size=(5, 4), dtype=np.int8)
    (out,) = _evaluate(text, a, b)
    assert out.dtype == np.int32
    assert np.array_equal(out, a.astype(np.int64) @ b.astype(np.int64))


def test_a_two_accumulator_reduction_runs_in_the_irs_own_order() -> None:
    """An arg-min carries a value and an index; no single combiner states it, so it is stepped."""
    text = """
module {
  func.func @forward(%x: tensor<1x6xi64>) -> tensor<1xi64> {
    %big = arith.constant 9223372036854775807 : i64
    %zero = arith.constant 0 : i64
    %v0 = tensor.splat %big : tensor<1xi64>
    %i0 = tensor.splat %zero : tensor<1xi64>
    %v, %i = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%x : tensor<1x6xi64>) outs(%v0, %i0 : tensor<1xi64>, tensor<1xi64>) {
    ^bb0(%in: i64, %best: i64, %at: i64):
      %k = linalg.index 1 : index
      %kk = arith.index_cast %k : index to i64
      %lt = arith.cmpi slt, %in, %best : i64
      %nb = arith.select %lt, %in, %best : i64
      %na = arith.select %lt, %kk, %at : i64
      linalg.yield %nb, %na : i64, i64
    } -> (tensor<1xi64>, tensor<1xi64>)
    return %i : tensor<1xi64>
  }
}
"""
    x = np.array([[5, 3, 9, 3, 1, 7]], np.int64)
    (index,) = _evaluate(text, x)
    assert index.tolist() == [4]


def test_a_gathering_map_reads_the_window_it_names() -> None:
    """``(d0 * 2 + d1)`` over a 1-D source is a stride-2 window of width 2: an im2col in miniature."""
    text = """
module {
  func.func @forward(%x: tensor<8xf32>) -> tensor<4x2xf32> {
    %e = tensor.empty() : tensor<4x2xf32>
    %g = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0 * 2 + d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<8xf32>) outs(%e : tensor<4x2xf32>) {
    ^bb0(%in: f32, %o: f32):
      linalg.yield %in : f32
    } -> tensor<4x2xf32>
    return %g : tensor<4x2xf32>
  }
}
"""
    x = np.arange(8, dtype=np.float32)
    (out,) = _evaluate(text, x)
    assert out.tolist() == [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_a_rank0_splat_stays_rank0_through_a_reduction() -> None:
    text = """
module {
  func.func @forward(%x: tensor<5xi64>) -> tensor<i64> {
    %zero = arith.constant 0 : i64
    %s = tensor.splat %zero : tensor<i64>
    %r = linalg.reduce ins(%x : tensor<5xi64>) outs(%s : tensor<i64>) dimensions = [0]
      (%a: i64, %b: i64) {
        %c = arith.addi %a, %b : i64
        linalg.yield %c : i64
      }
    return %r : tensor<i64>
  }
}
"""
    (out,) = _evaluate(text, np.arange(5, dtype=np.int64))
    assert out.shape == () and int(out) == 10


def test_a_generic_whose_body_only_adds_is_a_sum_not_a_contraction() -> None:
    text = """
module {
  func.func @forward(%x: tensor<2x3x4xf32>) -> tensor<2x3xf32> {
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<2x3xf32>
    %f = linalg.fill ins(%zero : f32) outs(%e : tensor<2x3xf32>) -> tensor<2x3xf32>
    %r = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%x : tensor<2x3x4xf32>) outs(%f : tensor<2x3xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %s = arith.addf %v, %acc : f32
      linalg.yield %s : f32
    } -> tensor<2x3xf32>
    return %r : tensor<2x3xf32>
  }
}
"""
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    (out,) = _evaluate(text, x)
    assert np.array_equal(out, x.sum(axis=2))


def test_a_bf16_result_is_rounded_to_nearest_even_and_stored_as_its_bits() -> None:
    text = f"""
module {{
  func.func @forward(%x: tensor<1x3xf32>) -> tensor<1x3xbf16> {{
    %e = tensor.empty() : tensor<1x3xbf16>
    %t = linalg.generic {{indexing_maps = [{_ID2}, {_ID2}], iterator_types = ["parallel", "parallel"]}} ins(%x : tensor<1x3xf32>) outs(%e : tensor<1x3xbf16>) {{
    ^bb0(%in: f32, %o: bf16):
      %r = arith.truncf %in : f32 to bf16
      linalg.yield %r : bf16
    }} -> tensor<1x3xbf16>
    return %t : tensor<1x3xbf16>
  }}
}}
"""
    x = np.array([[1.0, 1.00390625, 1.01171875]], np.float32)  # 1 + 2^-8 ties to even; 1 + 3*2^-8 rounds up
    (bits,) = _evaluate(text, x)
    assert bits.dtype == np.uint16
    values = (bits.astype(np.uint32) << 16).view(np.float32)
    assert values.reshape(-1).tolist() == [1.0, 1.0, 1.015625]


def test_a_call_is_answered_by_the_caller_or_refused() -> None:
    text = """
module {
  func.func @forward(%x: tensor<2xf32>) -> tensor<2xf32> {
    %y = func.call @ext(%x) : (tensor<2xf32>) -> tensor<2xf32>
    return %y : tensor<2xf32>
  }
  func.func private @ext(tensor<2xf32>) -> tensor<2xf32>
}
"""
    x = np.array([1.0, 2.0], np.float32)
    (out,) = _evaluate(text, x, calls={"ext": lambda values: [values[0] * 2]})
    assert out.tolist() == [2.0, 4.0]
    with pytest.raises(LN.LinalgNumpyError, match="does not define it"):
        _evaluate(text, x)


def test_an_unknown_body_op_is_refused_by_name() -> None:
    text = f"""
module {{
  func.func @forward(%x: tensor<1x2xf32>) -> tensor<1x2xf32> {{
    %e = tensor.empty() : tensor<1x2xf32>
    %t = linalg.generic {{indexing_maps = [{_ID2}, {_ID2}], iterator_types = ["parallel", "parallel"]}} ins(%x : tensor<1x2xf32>) outs(%e : tensor<1x2xf32>) {{
    ^bb0(%in: f32, %o: f32):
      %r = math.atan %in : f32
      linalg.yield %r : f32
    }} -> tensor<1x2xf32>
    return %t : tensor<1x2xf32>
  }}
}}
"""
    with pytest.raises(LN.LinalgNumpyError, match="math.atan"):
        _evaluate(text, np.zeros((1, 2), np.float32))


def _int_binary_module(body: str, element: str = "i64") -> str:
    return f"""
module {{
  func.func @forward(%a: tensor<1x8x{element}>, %b: tensor<1x8x{element}>) -> tensor<1x8x{element}> {{
    %e = tensor.empty() : tensor<1x8x{element}>
    %t = linalg.generic {{indexing_maps = [{_ID2}, {_ID2}, {_ID2}], iterator_types = ["parallel", "parallel"]}} ins(%a, %b : tensor<1x8x{element}>, tensor<1x8x{element}>) outs(%e : tensor<1x8x{element}>) {{
    ^bb0(%x: {element}, %y: {element}, %o: {element}):
{body}
    }} -> tensor<1x8x{element}>
    return %t : tensor<1x8x{element}>
  }}
}}
"""


def test_an_integer_floor_division_rounds_toward_minus_infinity() -> None:
    """``floordivsi``: ``-7 // 2 == -4``, never the truncation ``divsi`` gives (-3)."""
    text = _int_binary_module("      %r = arith.floordivsi %x, %y : i64\n      linalg.yield %r : i64")
    a = np.array([[-7, 7, -7, 7, -1, 0, -(2**63) + 1, 2**62]], np.int64)
    b = np.array([[2, -2, -2, 2, 3, 5, 7, -3]], np.int64)
    (out,) = _evaluate(text, a, b)
    assert out.tolist() == [[x // y for x, y in zip(a[0].tolist(), b[0].tolist())]]
    with pytest.raises(LN.LinalgNumpyError, match="by zero"):
        _evaluate(text, a, np.zeros_like(b))


def test_a_signed_division_truncates_exactly_past_2_pow_53() -> None:
    """``divsi``/``remsi`` round toward zero in integers; a float quotient would drop the low bits."""
    a = np.array([[-7, 7, -7, 7, 2**62 + 1, -(2**62) - 3, 0, 9]], np.int64)
    b = np.array([[2, -2, -2, 2, 3, 5, 7, -4]], np.int64)
    (q,) = _evaluate(_int_binary_module("      %r = arith.divsi %x, %y : i64\n      linalg.yield %r : i64"), a, b)
    (r,) = _evaluate(_int_binary_module("      %r = arith.remsi %x, %y : i64\n      linalg.yield %r : i64"), a, b)
    want_q = [
        int(x / abs(x) if x else 0) * (abs(x) // abs(y)) * (1 if y > 0 else -1)
        for x, y in zip(a[0].tolist(), b[0].tolist())
    ]
    assert q.tolist() == [want_q]
    assert r.tolist() == [[x - y * w for x, y, w in zip(a[0].tolist(), b[0].tolist(), want_q)]]


def test_an_equality_compare_reads_predicate_zero() -> None:
    """``cmpi eq`` carries predicate 0, an attribute that is falsy; it must still be read as ``eq``."""
    body = "      %c = arith.cmpi eq, %x, %y : i64\n      %r = arith.extui %c : i1 to i64\n      linalg.yield %r : i64"
    a = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], np.int64)
    b = np.array([[1, 0, 3, 0, 5, 0, 7, 0]], np.int64)
    (out,) = _evaluate(_int_binary_module(body), a, b)
    assert out.tolist() == [[1, 0, 1, 0, 1, 0, 1, 0]]


def test_an_unsigned_division_reads_bit_patterns_and_refuses_zero() -> None:
    """``divui``/``remui`` divide the operands' unsigned values; a zero divisor is undefined."""
    text = _int_binary_module("      %r = arith.divui %x, %y : i64\n      linalg.yield %r : i64")
    rem = _int_binary_module("      %r = arith.remui %x, %y : i64\n      linalg.yield %r : i64")
    a = np.array([[7, 360, -1, 2**62 + 5, 0, -8, 13, 1]], np.int64)
    b = np.array([[2, 7, 2, 3, 9, 3, -1, 1]], np.int64)
    (q,) = _evaluate(text, a, b)
    (r,) = _evaluate(rem, a, b)
    ua = [x % 2**64 for x in a[0].tolist()]
    ub = [y % 2**64 for y in b[0].tolist()]
    signed = [v - 2**64 if v >= 2**63 else v for v in (x // y for x, y in zip(ua, ub))]
    assert q.tolist() == [signed]
    assert r.tolist() == [[x % y for x, y in zip(ua, ub)]]
    with pytest.raises(LN.LinalgNumpyError, match="by zero"):
        _evaluate(text, a, np.zeros_like(b))


def test_an_arithmetic_shift_keeps_the_sign_and_refuses_poison() -> None:
    """``shrsi`` is floor division by a power of two; a clamped amount (``minui`` against width-1, how a
    signed shift by an out-of-range amount is written) is exact, and an unclamped one is poison."""
    clamped = (
        "      %c = arith.constant 63 : i64\n"
        "      %s = arith.minui %y, %c : i64\n"
        "      %r = arith.shrsi %x, %s : i64\n"
        "      linalg.yield %r : i64"
    )
    a = np.array([[-9, -8, -1, 1, 2**62, -(2**63), 5, -5]], np.int64)
    b = np.array([[1, 3, 0, 1, 70, 64, -1, 63]], np.int64)
    (out,) = _evaluate(_int_binary_module(clamped), a, b)
    want = [x >> (63 if not 0 <= s <= 63 else s) for x, s in zip(a[0].tolist(), b[0].tolist())]
    assert out.tolist() == [want]
    bare = "      %r = arith.shrsi %x, %y : i64\n      linalg.yield %r : i64"
    with pytest.raises(LN.LinalgNumpyError, match="poison"):
        _evaluate(_int_binary_module(bare), a, b)


def test_an_unsigned_min_compares_bit_patterns() -> None:
    text = _int_binary_module("      %r = arith.minui %x, %y : i32\n      linalg.yield %r : i32", "i32")
    a = np.array([[-1, 5, -5, 0, 7, -(2**31), 2**31 - 1, 3]], np.int32)
    b = np.array([[31, -2, 4, -1, 7, 2**31 - 1, -(2**31), 3]], np.int32)
    (out,) = _evaluate(text, a, b)
    assert out.tolist() == [[31, 5, 4, 0, 7, 2**31 - 1, 2**31 - 1, 3]]
