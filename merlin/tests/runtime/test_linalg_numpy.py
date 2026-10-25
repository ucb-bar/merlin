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

_ID1 = "affine_map<(d0) -> (d0)>"
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


def _f32_fma(a, b, c) -> np.float32:
    """``a * b + c`` rounded once to f32, from the exact rational value (nearest, ties to even)."""
    from fractions import Fraction

    exact = Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c))
    guess = np.float32(float(exact))
    candidates = [guess, np.nextafter(guess, np.float32(np.inf)), np.nextafter(guess, np.float32(-np.inf))]
    return min(candidates, key=lambda f: (abs(Fraction(float(f)) - exact), int(np.array(f).view(np.uint32)) & 1))


def _fma_module(element: str = "f32") -> str:
    return f"""
module {{
  func.func @forward(%a: tensor<1x3x{element}>, %b: tensor<1x3x{element}>, %c: tensor<1x3x{element}>) -> tensor<1x3x{element}> {{
    %e = tensor.empty() : tensor<1x3x{element}>
    %t = linalg.generic {{indexing_maps = [{_ID2}, {_ID2}, {_ID2}, {_ID2}], iterator_types = ["parallel", "parallel"]}} ins(%a, %b, %c : tensor<1x3x{element}>, tensor<1x3x{element}>, tensor<1x3x{element}>) outs(%e : tensor<1x3x{element}>) {{
    ^bb0(%x: {element}, %y: {element}, %z: {element}, %o: {element}):
      %r = math.fma %x, %y, %z : {element}
      linalg.yield %r : {element}
    }} -> tensor<1x3x{element}>
    return %t : tensor<1x3x{element}>
  }}
}}
"""


def test_a_fused_multiply_add_rounds_once() -> None:
    """``(1 + 2**-12)**2`` is exactly halfway between two f32 values. A product rounded first ties to
    even and loses what ``c`` adds; a float64 sum loses a ``c`` of ``2**-70`` too, then ties. One
    rounding of the exact value goes the way ``c`` points."""
    near_one = np.float32(1 + 2.0**-12)
    a = np.full((1, 3), near_one, np.float32)
    c = np.array([[2.0**-70, -(2.0**-70), 0.0]], np.float32)
    (out,) = _evaluate(_fma_module(), a, a, c)
    low = np.float32(1 + 2.0**-11)
    assert out.tolist() == [[float(np.nextafter(low, np.float32(2))), float(low), float(low)]]
    rng = np.random.default_rng(3)
    for _ in range(64):
        x, y, z = (rng.standard_normal((1, 3)).astype(np.float32) * np.float32(10.0) ** rng.integers(-8, 8)
                   for _ in range(3))  # fmt: skip
        (got,) = _evaluate(_fma_module(), x, y, z)
        assert got.tolist() == [[float(_f32_fma(*v)) for v in zip(x[0], y[0], z[0], strict=True)]]
    with pytest.raises(LN.LinalgNumpyError, match="math.fma of f64"):
        _evaluate(_fma_module("f64"), *(np.ones((1, 3)),) * 3)


def test_a_loop_in_a_structured_body_carries_its_values_in_source_order() -> None:
    """A per-row running mean (the shape a CPU-faithful LayerNorm's Welford loop lowers to): an
    ``scf.for`` inside a parallel generic, reading the row element by element, its two ``iter_args``
    carried from one iteration to the next and updated through ``math.fma``."""
    text = f"""
module {{
  func.func @forward(%x: tensor<3x8xf32>) -> tensor<3x1xf32> {{
    %e = tensor.empty() : tensor<3x1xf32>
    %f = tensor.empty() : tensor<3x1xf32>
    %mean, %m2 = linalg.generic {{indexing_maps = [{_ID2}, {_ID2}], iterator_types = ["parallel", "parallel"]}} outs(%e, %f : tensor<3x1xf32>, tensor<3x1xf32>) {{
    ^bb0(%o: f32, %p: f32):
      %row = linalg.index 0 : index
      %zero = arith.constant 0.000000e+00 : f32
      %one = arith.constant 1.000000e+00 : f32
      %lo = arith.constant 0 : index
      %hi = arith.constant 8 : index
      %st = arith.constant 1 : index
      %m, %s = scf.for %k = %lo to %hi step %st iter_args(%mu = %zero, %acc = %zero) -> (f32, f32) {{
        %c1 = arith.constant 1 : index
        %k1 = arith.addi %k, %c1 : index
        %ki = arith.index_cast %k1 : index to i64
        %kf = arith.sitofp %ki : i64 to f32
        %inv = arith.divf %one, %kf : f32
        %v = tensor.extract %x[%row, %k] : tensor<3x8xf32>
        %d = arith.subf %v, %mu : f32
        %nmu = math.fma %d, %inv, %mu : f32
        %d2 = arith.subf %v, %nmu : f32
        %nacc = math.fma %d, %d2, %acc : f32
        scf.yield %nmu, %nacc : f32, f32
      }}
      linalg.yield %m, %s : f32, f32
    }} -> (tensor<3x1xf32>, tensor<3x1xf32>)
    return %m2 : tensor<3x1xf32>
  }}
}}
"""
    x = (np.random.default_rng(5).standard_normal((3, 8)) * 3 + 20).astype(np.float32)
    expected = []
    for row in x:
        mu = acc = np.float32(0)
        for k, v in enumerate(row):
            d = np.float32(v - mu)
            mu = _f32_fma(d, np.float32(1) / np.float32(k + 1), mu)
            acc = _f32_fma(d, np.float32(v - mu), acc)
        expected.append([float(acc)])
    (m2,) = _evaluate(text, x)
    assert m2.tolist() == expected


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


def test_an_unsigned_compare_orders_bit_patterns() -> None:
    """``ult``/``uge`` compare the unsigned values: a negative signed operand is a large unsigned one."""
    body = "      %c = arith.cmpi uge, %x, %y : i32\n      %r = arith.extui %c : i1 to i32\n      linalg.yield %r : i32"
    a = np.array([[-1, 5, -5, 0, 7, -(2**31), 2**31 - 1, 3]], np.int32)
    b = np.array([[31, -2, 4, -1, 7, 2**31 - 1, -(2**31), 4]], np.int32)
    (out,) = _evaluate(_int_binary_module(body, "i32"), a, b)
    assert out.tolist() == [[int(x % 2**32 >= y % 2**32) for x, y in zip(a[0].tolist(), b[0].tolist())]]


def test_logical_shifts_move_the_bit_pattern_and_refuse_poison() -> None:
    right = _int_binary_module("      %r = arith.shrui %x, %y : i32\n      linalg.yield %r : i32", "i32")
    left = _int_binary_module("      %r = arith.shli %x, %y : i32\n      linalg.yield %r : i32", "i32")
    a = np.array([[-1, -8, 5, 2**30, -(2**31), 7, 1, 3]], np.int32)
    b = np.array([[1, 3, 0, 1, 31, 2, 31, 30]], np.int32)

    def signed(v):
        return v - 2**32 if v >= 2**31 else v

    (r,) = _evaluate(right, a, b)
    (l,) = _evaluate(left, a, b)
    pairs = list(zip(a[0].tolist(), b[0].tolist()))
    assert r.tolist() == [[signed((x % 2**32) >> s) for x, s in pairs]]
    assert l.tolist() == [[signed(((x % 2**32) << s) % 2**32) for x, s in pairs]]
    with pytest.raises(LN.LinalgNumpyError, match="poison"):
        _evaluate(right, a, np.full_like(b, 32))


def test_a_bitcast_reads_the_same_bits_as_the_other_type() -> None:
    text = f"""
module {{
  func.func @forward(%x: tensor<1x4xf32>) -> tensor<1x4xf32> {{
    %e = tensor.empty() : tensor<1x4xf32>
    %t = linalg.generic {{indexing_maps = [{_ID2}, {_ID2}], iterator_types = ["parallel", "parallel"]}} ins(%x : tensor<1x4xf32>) outs(%e : tensor<1x4xf32>) {{
    ^bb0(%in: f32, %o: f32):
      %bits = arith.bitcast %in : f32 to i32
      %mask = arith.constant 2147483647 : i32
      %abs = arith.andi %bits, %mask : i32
      %r = arith.bitcast %abs : i32 to f32
      linalg.yield %r : f32
    }} -> tensor<1x4xf32>
    return %t : tensor<1x4xf32>
  }}
}}
"""
    x = np.array([[-1.5, 2.0, -0.0, -np.inf]], np.float32)
    (out,) = _evaluate(text, x)
    assert out.view(np.uint32).tolist() == (np.abs(x)).view(np.uint32).tolist()


def test_an_integer_wider_than_64_bits_is_exact_and_wraps_at_its_width() -> None:
    """An exact readout multiplies in ``i281``-style types: Python integers, wrapped at the declared width."""
    body = (
        "      %wx = arith.extsi %x : i64 to i130\n"
        "      %wy = arith.extsi %y : i64 to i130\n"
        "      %p = arith.muli %wx, %wy : i130\n"
        "      %big = arith.constant 680564733841876926926749214863536422912 : i130\n"  # 2**129: -2**129 wrapped
        "      %q = arith.addi %p, %big : i130\n"
        "      %c64 = arith.constant 64 : i130\n"
        "      %h = arith.shrui %q, %c64 : i130\n"
        "      %neg = arith.cmpi slt, %q, %p : i130\n"
        "      %t = arith.trunci %h : i130 to i64\n"
        "      %z = arith.constant 0 : i64\n"
        "      %r = arith.select %neg, %t, %z : i64\n"
        "      linalg.yield %r : i64"
    )
    a = np.array([[2**62, -(2**62), 3, -1, 2**63 - 1, -(2**63), 0, 12345]], np.int64)
    b = np.array([[2**62, 2**62, -5, -1, 2**63 - 1, -(2**63), 9, -67890]], np.int64)
    (out,) = _evaluate(_int_binary_module(body), a, b)

    def wrap(v, w):
        return (v + 2 ** (w - 1)) % 2**w - 2 ** (w - 1)

    want = []
    for x, y in zip(a[0].tolist(), b[0].tolist()):
        p = wrap(x * y, 130)
        q = wrap(p - 2**129, 130)
        want.append(wrap((q % 2**130) >> 64, 64) if q < p else 0)
    assert out.tolist() == [want]


def test_a_runtime_assertion_is_checked_not_skipped() -> None:
    """``cf.assert`` is evaluated: a true condition passes through, a false one stops the oracle."""
    text = """
func.func @forward(%x: tensor<2xi64>, %y: i64) -> tensor<2xi64> {
  %c = arith.constant 4 : i64
  %ok = arith.cmpi slt, %y, %c : i64
  cf.assert %ok, "index in bounds"
  return %x : tensor<2xi64>
}
"""
    x = np.array([1, 2], np.int64)
    (out,) = _evaluate(text, x, np.int64(3))
    assert out.tolist() == [1, 2]
    with pytest.raises(LN.LinalgNumpyError, match="index in bounds"):
        _evaluate(text, x, np.int64(9))


def test_a_tensor_dim_reads_the_extent_and_refuses_an_index_past_the_rank() -> None:
    """``tensor.dim`` answers the operand's extent along a constant or runtime index; an index outside the
    rank is undefined and stops the oracle."""
    text = """
func.func @forward(%x: tensor<2x3xi64>, %i: index) -> (index, index) {
  %c1 = arith.constant 1 : index
  %d = tensor.dim %x, %c1 : tensor<2x3xi64>
  %e = tensor.dim %x, %i : tensor<2x3xi64>
  return %d, %e : index, index
}
"""
    x = np.zeros((2, 3), np.int64)
    fixed, read = _evaluate(text, x, np.int64(0))
    assert (int(fixed), int(read)) == (3, 2)
    with pytest.raises(LN.LinalgNumpyError, match="outside rank 2"):
        _evaluate(text, x, np.int64(2))


# ------------------------------------------------------------------------------- the constant cache

_CONTRACTION = (
    "{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, "
    'affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]}'
)
_TRANSPOSED_WEIGHT = """
module {
  func.func @forward(%x: tensor<2x4xi8>, %w: tensor<3x4xi8>) -> tensor<2x3xi32> {
    %t0 = tensor.empty() : tensor<4x3xi8>
    %wt = linalg.transpose ins(%w : tensor<3x4xi8>) outs(%t0 : tensor<4x3xi8>) permutation = [1, 0]
    %c0 = arith.constant 0 : i32
    %s = tensor.splat %c0 : tensor<2x3xi32>
    %m = linalg.generic CONTRACTION ins(%x, %wt : tensor<2x4xi8>, tensor<4x3xi8>) outs(%s : tensor<2x3xi32>) {
    ^bb0(%a: i8, %b: i8, %acc: i32):
      %ae = arith.extsi %a : i8 to i32
      %be = arith.extsi %b : i8 to i32
      %p = arith.muli %ae, %be : i32
      %r = arith.addi %acc, %p : i32
      linalg.yield %r : i32
    } -> tensor<2x3xi32>
    return %m : tensor<2x3xi32>
  }
}
""".replace("CONTRACTION", _CONTRACTION)


def _count_ops(monkeypatch, name: str) -> list[int]:
    seen = [0]
    real = LN.evaluate_op

    def counted(op, env, **kwargs):
        seen[0] += op.name == name
        return real(op, env, **kwargs)

    monkeypatch.setattr(LN, "evaluate_op", counted)
    return seen


def test_work_on_constants_alone_is_done_once_and_gives_the_same_bytes(monkeypatch) -> None:
    """A weight's transpose reads only a constant: one evaluation serves every later one, and every
    result is the bytes an uncached evaluation gives. The weight's float64 copy is kept the same way."""
    module = mq.parse(_TRANSPOSED_WEIGHT)
    rng = np.random.default_rng(1)
    w = rng.integers(-128, 128, size=(3, 4), dtype=np.int8)
    xs = [rng.integers(-128, 128, size=(2, 4), dtype=np.int8) for _ in range(3)]
    plain = [LN.evaluate(module, [x, w.copy()])[0] for x in xs]

    transposes = _count_ops(monkeypatch, "linalg.transpose")
    cache = LN.ConstantCache()
    weight = cache.mark(w.copy())
    cached = [LN.evaluate(module, [x, weight], constants=cache)[0] for x in xs]
    assert transposes[0] == 1
    for a, b in zip(plain, cached, strict=True):
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes()
    assert np.array_equal(cached[0], xs[0].astype(np.int64) @ w.astype(np.int64).T)
    assert not weight.flags.writeable  # a reused value is never written
    with pytest.raises(ValueError):
        weight[0, 0] = 1


def test_an_op_whose_body_reads_an_outer_value_is_never_hoisted() -> None:
    """Every operand of the generic is a constant, but its body multiplies by ``%s``, an argument that
    changes between evaluations: hoisting it would answer the first ``%s`` forever."""
    text = f"""
module {{
  func.func @forward(%w: tensor<4xf32>, %s: f32) -> tensor<4xf32> {{
    %e = tensor.empty() : tensor<4xf32>
    %r = linalg.generic {{indexing_maps = [{_ID1}, {_ID1}], iterator_types = ["parallel"]}}
        ins(%w : tensor<4xf32>) outs(%e : tensor<4xf32>) {{
    ^bb0(%a: f32, %o: f32):
      %m = arith.mulf %a, %s : f32
      linalg.yield %m : f32
    }} -> tensor<4xf32>
    return %r : tensor<4xf32>
  }}
}}
"""
    module = mq.parse(text)
    cache = LN.ConstantCache()
    w = cache.mark(np.arange(4, dtype=np.float32))
    generic = next(op for op in module.walk() if op.name == "linalg.generic")
    assert not cache.self_contained(generic)
    two = LN.evaluate(module, [w, np.float32(2)], constants=cache)[0]
    three = LN.evaluate(module, [w, np.float32(3)], constants=cache)[0]
    assert two.tolist() == [0, 2, 4, 6] and three.tolist() == [0, 3, 6, 9]


def test_an_op_reading_a_value_that_changes_is_evaluated_every_time(monkeypatch) -> None:
    """Only constants are hoisted: the contraction reads the input, so it runs on every evaluation."""
    module = mq.parse(_TRANSPOSED_WEIGHT)
    generics = _count_ops(monkeypatch, "linalg.generic")
    cache = LN.ConstantCache()
    w = cache.mark(np.ones((3, 4), np.int8))
    for k in range(3):
        (out,) = LN.evaluate(module, [np.full((2, 4), k, np.int8), w], constants=cache)
        assert out.tolist() == [[4 * k] * 3] * 2
    assert generics[0] == 3
