"""A linalg-on-tensors module evaluated independently, without lowering the model.

    values = evaluate(module, arguments)                     # every result of @forward
    values = evaluate(module, arguments, observe=hook)       # hook(op, index, results) per op
    values = evaluate(module, arguments, constants=cache)    # constant work done once per cache

The whole-model program of an OPEN model (one whose host regions compute between its accelerator
groups) needs an oracle that is independent of every arm, host code included. The dispatch runtime
(:mod:`merlin.runtime.dispatch_runtime`) is not one: it compiles each kernel with this repo's own LLVM
lowering, so it grades the compiler with itself. This module reads each operation's semantics off
the IR -- its indexing maps, its iterator types, its body -- and computes it with numpy, so a wrong
lowering of any host region is a disagreement with this, not a shared mistake.

Semantics, stated because an oracle that rounds differently from the model is a different model:

* integers wrap at their declared width, like the IR; an int8 x int8 contraction is summed exactly
  (float64 products of int8 operands are exact, and so is their sum while it stays under 2**53);
* ``f32`` arithmetic is carried out in float32; a contraction sums in float32 through BLAS, whose
  order differs from a sequential loop -- the float results are an oracle up to reassociation, which
  is why a float comparison against them is BOUNDED, never exact;
* ``bf16`` is stored as its 16-bit pattern (the runtime's layout) and computed in float32, each bf16
  result rounded to nearest-even, which is what a bf16 operation over f32 math does.
* non-contraction floating reductions carry their initial accumulator through the source's
  lexicographic reduction indices, rounding each body operation at its declared type. Only an
  explicit ``reassoc`` flag on each floating combiner permits a NumPy reduction tree.
* ``math.fma`` rounds once: the product of two f32/f16/bf16 operands is exact in float64, the sum
  is rounded to odd there (an error-free transformation recovers what float64 dropped), and that is
  rounded to the declared type -- the same value a fused multiply-add returns;
* an ``scf.for`` inside a structured op's body runs as the IR states it, one iteration after another
  with its ``iter_args`` carried, over every point of the iteration space at once;
* an explicitly selected host-math context may define binary32 exp/sin/cos using versioned scalar
  source shared with the runtime. It does not compile model IR or import a device implementation;
  all other interpreter operations remain independently evaluated from the source IR.

Only the vocabulary a captured model uses is implemented, and anything else RAISES naming the
operation: an evaluator that skipped an op would compute a different model and still return.
Nothing here names a target or a model.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

__all__ = [
    "ConstantCache",
    "LinalgNumpyError",
    "NUMERIC_CONTRACT_VERSION",
    "bf16_round",
    "evaluate",
    "evaluate_op",
    "last_uses",
    "numeric_contract",
    "storage_dtype",
]

NUMERIC_CONTRACT_VERSION = "merlin.linalg_numpy.numeric.v2.strict_float_reductions"


def numeric_contract() -> dict[str, Any]:
    """Policy identity for newly derived references; source/dependency bytes also need binding."""
    contract = {
        "version": NUMERIC_CONTRACT_VERSION,
        "floating_reductions": "source_lexicographic_init_first_per_operation_rounding",
        "explicit_reassociation": "numpy_f32_f64_tree_only_with_combiner_reassoc",
        "floating_contractions": "blas_bounded_up_to_reassociation",
    }
    from merlin.runtime.host_math_reference import active_host_math

    selected = active_host_math()
    if selected is not None:
        contract["host_math"] = selected.identity()
    return contract


class LinalgNumpyError(RuntimeError):
    """An operation this evaluator cannot state in numpy, named."""


_DYN = -9223372036854775808  # xDSL's dynamic-extent sentinel

_STORAGE = {
    "f32": np.float32,
    "f64": np.float64,
    "f16": np.float16,
    "bf16": np.uint16,  # the runtime's layout: the raw 16-bit pattern
    "i64": np.int64,
    "i32": np.int32,
    "i16": np.int16,
    "i8": np.int8,
    "i1": np.bool_,
    "index": np.int64,
}
#: The dtype a body COMPUTES a value of this element type in (bf16 in float32, then rounded).
_COMPUTE = {**_STORAGE, "bf16": np.float32}


def _contiguous(x, dtype=None) -> np.ndarray:
    """``np.ascontiguousarray`` that keeps a rank-0 array rank-0 (numpy's promotes it to rank 1,
    which turns a scalar tensor into a one-element vector and breaks every later reshape)."""
    array = np.asarray(x, dtype=dtype)
    return array if array.ndim == 0 else np.ascontiguousarray(array)


def _elem(t) -> str:
    element = getattr(t, "element_type", t)
    return str(element)


def storage_dtype(element: str):
    """numpy storage for an MLIR element type (bf16 as its 16-bit pattern)."""
    if element not in _STORAGE:
        raise LinalgNumpyError(f"no numpy storage for element type {element!r}")
    return _STORAGE[element]


def _compute_dtype(element: str):
    if element not in _COMPUTE:
        if _wide_width(element) is not None:
            return np.object_
        raise LinalgNumpyError(f"no numpy arithmetic for element type {element!r}")
    return _COMPUTE[element]


def _wide_width(element: str) -> int | None:
    """The width of an integer type past numpy's 64 bits (an exact readout's ``i281``), else ``None``."""
    digits = element[1:] if element.startswith("i") else ""
    return int(digits) if digits.isdigit() and int(digits) > 64 else None


def _fit(value, element: str):
    """Integer values wrapped to ``element``'s width, in its compute form.

    A type wider than 64 bits computes in Python integers (an object array) and is wrapped here
    explicitly; a native type wraps through numpy's own conversion, as it always has.
    """
    array = np.asarray(value)
    width = _wide_width(element)
    if width is None and array.dtype != np.object_:
        return array.astype(_compute_dtype(element))
    width = width or (int(element[1:]) if element[1:].isdigit() else 64)
    half = 1 << (width - 1)
    wrapped = ((array.astype(np.object_) + half) & ((1 << width) - 1)) - half
    return np.asarray(wrapped, dtype=_compute_dtype(element))


def bf16_round(x) -> np.ndarray:
    """float32 values rounded to bf16 (nearest-even), returned as float32."""
    u = _contiguous(np.asarray(x, np.float32)).view(np.uint32)
    bias = ((u >> 16) & np.uint32(1)) + np.uint32(0x7FFF)
    return (((u + bias) >> 16) << 16).astype(np.uint32).view(np.float32)


def _bf16_bits(x) -> np.ndarray:
    u = _contiguous(np.asarray(x, np.float32)).view(np.uint32)
    bias = ((u >> 16) & np.uint32(1)) + np.uint32(0x7FFF)
    return ((u + bias) >> 16).astype(np.uint16)


def _bf16_values(bits) -> np.ndarray:
    return (_contiguous(bits, np.uint16).astype(np.uint32) << 16).view(np.float32)


def _shape(t) -> tuple[int, ...]:
    get = getattr(t, "get_shape", None)
    return tuple(int(d) for d in get()) if get is not None else ()


def _ints(attr) -> list[int]:
    return [int(v) for v in attr.get_values()]


def _load(value: np.ndarray, element: str) -> np.ndarray:
    """A stored tensor as the values a body computes with."""
    if element == "bf16":
        return _bf16_values(value)
    return value


def _store(value, element: str, shape: Sequence[int] | None = None) -> np.ndarray:
    """Computed values as the stored tensor of ``element`` (bf16 rounded to its pattern)."""
    if element == "bf16":
        out = _bf16_bits(value)
    else:
        out = np.asarray(value).astype(storage_dtype(element), copy=False)
    if shape is not None:
        shape = tuple(int(e) for e in shape)
        # A rank-0 tensor may arrive as one element of rank 1 (a splat's storage); same element.
        out = out.reshape(shape) if out.size == int(np.prod(shape, dtype=np.int64)) else np.broadcast_to(out, shape)
    return out


# --------------------------------------------------------------------------------------- scalars

#: arith.cmpi predicates by their enum value (eq ne slt sle sgt sge ult ule ugt uge).
#: The unsigned predicates (ult ule ugt uge), compared on the operands' bit patterns.
_UNSIGNED_CMPI = frozenset({6, 7, 8, 9})
_CMPI = {
    0: np.equal,
    1: np.not_equal,
    2: np.less,
    3: np.less_equal,
    4: np.greater,
    5: np.greater_equal,
    6: np.less,
    7: np.less_equal,
    8: np.greater,
    9: np.greater_equal,
}


def _cmpf(predicate: int, a, b):
    """arith.cmpf by enum value (false oeq ogt oge olt ole one ord ueq ugt uge ult ule une uno true)."""
    a, b = np.asarray(a), np.asarray(b)
    unordered = np.isnan(a) | np.isnan(b)
    ordered = ~unordered
    table = {
        0: lambda: np.zeros(np.broadcast(a, b).shape, np.bool_),
        1: lambda: ordered & (a == b),
        2: lambda: ordered & (a > b),
        3: lambda: ordered & (a >= b),
        4: lambda: ordered & (a < b),
        5: lambda: ordered & (a <= b),
        6: lambda: ordered & (a != b),
        7: lambda: ordered,
        8: lambda: unordered | (a == b),
        9: lambda: unordered | (a > b),
        10: lambda: unordered | (a >= b),
        11: lambda: unordered | (a < b),
        12: lambda: unordered | (a <= b),
        13: lambda: unordered | (a != b),
        14: lambda: unordered,
        15: lambda: np.ones(np.broadcast(a, b).shape, np.bool_),
    }
    if predicate not in table:
        raise LinalgNumpyError(f"arith.cmpf predicate {predicate} is not defined")
    return table[predicate]()


def _unsigned(value, element: str):
    bits = int(element[1:]) if element.startswith("i") and element[1:].isdigit() else 64
    if bits > 64:
        return np.asarray(np.asarray(value).astype(np.object_) & ((1 << bits) - 1), dtype=np.object_)
    kind = {1: np.uint8, 8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}[bits]
    return np.asarray(value).astype(kind)


_ERF = np.frompyfunc(math.erf, 1, 1)


def _erf(x):
    """erf in float64 per element (numpy has none), returned in the input's float type."""
    x = np.asarray(x)
    return _ERF(x.astype(np.float64)).astype(np.float64).astype(x.dtype)


def _predicate(op) -> int:
    # An explicit None test: predicate 0 (``eq``) is an IntegerAttr that is falsy, so ``a or b`` would
    # read it as absent.
    attr = op.properties.get("predicate")
    if attr is None:
        attr = op.attributes.get("predicate")
    if attr is None:
        raise LinalgNumpyError(f"{op.name} has no predicate")
    return int(attr.value.data)


def _constant(op):
    """An arith.constant as a numpy scalar or array, in compute form."""
    value = op.properties["value"]
    rtype = op.results[0].type
    element = _elem(rtype)
    if hasattr(rtype, "get_shape"):  # a dense tensor constant
        shape = _shape(rtype)
        values = list(value.get_values()) if hasattr(value, "get_values") else None
        if values is None:
            raise LinalgNumpyError(f"tensor constant {value} has no readable elements")
        if len(values) == 1 and int(np.prod(shape, dtype=np.int64)) != 1:
            array = np.full(shape, values[0], dtype=_compute_dtype(element))
        else:
            array = np.asarray(values, dtype=_compute_dtype(element)).reshape(shape)
        return _store(array, element)
    data = getattr(value, "value", None)
    raw = data.data if data is not None and hasattr(data, "data") else value
    if element in ("f32", "bf16", "f16", "f64"):
        return np.asarray(float(raw), dtype=_compute_dtype(element))[()]
    if element == "i1":
        return np.bool_(int(raw) != 0)
    if _wide_width(element) is not None:
        # Kept a rank-0 object array: a bare Python int that large is cast to int64 by numpy's
        # promotion rules (and overflows) as soon as it meets a native array.
        return _fit(np.asarray(int(raw), dtype=np.object_), element)
    return np.asarray(int(raw), dtype=_compute_dtype(element))[()]


def _cast_to(value, element: str):
    """Values converted to ``element``'s compute form, bf16 rounded."""
    if element == "bf16":
        return bf16_round(np.asarray(value, np.float32))
    return np.asarray(value).astype(_compute_dtype(element))


def _fma(a, b, c, element: str) -> np.ndarray:
    """``a * b + c`` with ONE rounding to ``element``, as a fused multiply-add computes it.

    The product of two operands of at most 24 significand bits is exact in float64. The float64 sum
    is not, so the part it drops is recovered (Knuth's two-sum) and the sum is rounded to ODD: kept
    when inexact only if its last bit is odd, otherwise moved one step toward the dropped part. A
    value rounded to odd with at least two more bits than the target rounds to the target exactly
    as the unrounded value would (Boldo and Melquiond), so the second rounding is not a double one.
    """
    if element not in ("f32", "f16", "bf16"):
        raise LinalgNumpyError(f"math.fma of {element} is not evaluated with a single rounding")
    a, b, c = (np.asarray(v, np.float64) for v in (a, b, c))
    with np.errstate(all="ignore"):
        product = a * b
        total = product + c
        virtual = total - c
        dropped = (product - virtual) + (c - (total - virtual))
        inexact = np.isfinite(total) & np.isfinite(dropped) & (dropped != 0)
        even = (_contiguous(total).view(np.uint64) & np.uint64(1)) == 0
        odd = np.where(inexact & even, np.nextafter(total, np.where(dropped > 0, np.inf, -np.inf)), total)
        if element != "bf16":
            return odd.astype(_compute_dtype(element))
        # bf16 is rounded from float32: round to odd once more at float32 so that step is exact too.
        narrow = odd.astype(np.float32)
        back = narrow.astype(np.float64)
        miss = np.isfinite(back) & (back != odd)
        narrow_even = (_contiguous(narrow).view(np.uint32) & np.uint32(1)) == 0
        narrow = np.where(
            miss & narrow_even,
            np.nextafter(narrow, np.where(odd > back, np.float32(np.inf), np.float32(-np.inf))),
            narrow,
        )
        return bf16_round(narrow)


def _body_op(op, env: dict[int, Any], *, index_of: Callable[[int], np.ndarray] | None = None):
    """One scalar op of a region, over numpy arrays (or scalars). Returns its result."""
    name = op.name
    args = [env[id(o)] for o in op.operands]
    out = op.results[0].type if op.results else None
    element = _elem(out) if out is not None else None

    def fl(v):
        return np.asarray(v)

    if name == "arith.constant":
        return _constant(op)
    if name == "linalg.index":
        if index_of is None:
            raise LinalgNumpyError("linalg.index outside a structured op")
        return index_of(int(op.properties["dim"].value.data))
    if name in ("arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.maximumf", "arith.minimumf"):
        a, b = fl(args[0]), fl(args[1])
        fn = {
            "arith.addf": np.add,
            "arith.subf": np.subtract,
            "arith.mulf": np.multiply,
            "arith.divf": np.divide,
            "arith.maximumf": np.maximum,
            "arith.minimumf": np.minimum,
        }[name]
        with np.errstate(all="ignore"):
            return _cast_to(fn(a, b), element)
    if name in ("arith.maxnumf", "arith.minnumf"):
        fn = np.fmax if name == "arith.maxnumf" else np.fmin
        return _cast_to(fn(fl(args[0]), fl(args[1])), element)
    if name == "arith.negf":
        return _cast_to(-fl(args[0]), element)
    if name in ("arith.addi", "arith.subi", "arith.muli", "arith.andi", "arith.ori", "arith.xori"):
        fn = {
            "arith.addi": np.add,
            "arith.subi": np.subtract,
            "arith.muli": np.multiply,
            "arith.andi": np.bitwise_and,
            "arith.ori": np.bitwise_or,
            "arith.xori": np.bitwise_xor,
        }[name]
        dtype = _compute_dtype(element)
        with np.errstate(all="ignore"):
            return _fit(fn(fl(args[0]).astype(dtype), fl(args[1]).astype(dtype)), element)
    if name in ("arith.maxsi", "arith.minsi"):
        fn = np.maximum if name == "arith.maxsi" else np.minimum
        return np.asarray(fn(fl(args[0]), fl(args[1]))).astype(_compute_dtype(element))
    if name in ("arith.divsi", "arith.remsi"):
        # Integer arithmetic throughout: a float quotient loses the low bits of any operand past 2**53.
        a, b = fl(args[0]).astype(np.int64), fl(args[1]).astype(np.int64)
        if np.any(b == 0):
            raise LinalgNumpyError(f"{name} by zero is undefined")
        floor = np.floor_divide(a, b)
        quotient = floor + (((a - b * floor) != 0) & ((a < 0) != (b < 0)))  # round toward zero
        result = quotient if name == "arith.divsi" else a - b * quotient
        return result.astype(_compute_dtype(element))
    if name in ("arith.divui", "arith.remui"):
        ua, ub = _unsigned(args[0], element), _unsigned(args[1], element)
        if np.any(ub == 0):
            raise LinalgNumpyError(f"{name} by zero is undefined")
        result = ua // ub if name == "arith.divui" else ua % ub
        return result.astype(_compute_dtype(element))
    if name == "arith.floordivsi":
        a, b = fl(args[0]).astype(np.int64), fl(args[1]).astype(np.int64)
        if np.any(b == 0):
            raise LinalgNumpyError("arith.floordivsi by zero is undefined")
        return np.floor_divide(a, b).astype(_compute_dtype(element))  # rounds toward -inf, as the op does
    if name in ("arith.minui", "arith.maxui"):
        a, b = fl(args[0]), fl(args[1])
        ua, ub = _unsigned(a, element), _unsigned(b, element)
        keep_a = ua <= ub if name == "arith.minui" else ua >= ub
        return np.where(keep_a, a, b).astype(_compute_dtype(element))
    if name in ("arith.shrui", "arith.shli"):
        # Logical shifts of the bit pattern, wrapping at the declared width; an amount outside
        # [0, width) is poison, refused as for arith.shrsi.
        width = int(element[1:]) if element[1:].isdigit() else 64
        s = fl(args[1]).astype(np.int64)
        if np.any((s < 0) | (s >= width)):
            raise LinalgNumpyError(f"{name} by an amount outside [0, {width}) is poison")
        bits = _unsigned(args[0], element)
        shift = s.astype(bits.dtype)
        moved = np.right_shift(bits, shift) if name == "arith.shrui" else np.left_shift(bits, shift)
        return _fit(moved, element)
    if name == "arith.shrsi":
        # Arithmetic shift; an amount outside [0, width) is poison in the IR, so it is refused rather
        # than given whatever value numpy's shift would produce.
        width = int(element[1:])
        a, s = fl(args[0]), fl(args[1]).astype(np.int64)
        if np.any((s < 0) | (s >= width)):
            raise LinalgNumpyError(f"arith.shrsi by an amount outside [0, {width}) is poison")
        return np.right_shift(a.astype(_compute_dtype(element)), s.astype(_compute_dtype(element)))
    if name == "arith.cmpi":
        predicate = _predicate(op)
        if predicate in _UNSIGNED_CMPI:
            # ult/ule/ugt/uge order the BIT PATTERNS: a negative signed value is a large unsigned one.
            source = _elem(op.operands[0].type)
            return _CMPI[predicate](_unsigned(args[0], source), _unsigned(args[1], source))
        return _CMPI[predicate](fl(args[0]), fl(args[1]))
    if name == "arith.cmpf":
        return _cmpf(_predicate(op), args[0], args[1])
    if name == "arith.select":
        return np.where(fl(args[0]).astype(np.bool_), args[1], args[2])
    if name in ("arith.extsi", "arith.trunci", "arith.index_cast"):
        dtype = _compute_dtype(element)
        if element != "i1" and (dtype == np.object_ or np.asarray(args[0]).dtype == np.object_):
            return _fit(args[0], element)  # to or from a type wider than 64 bits: exact, then wrapped
        with np.errstate(all="ignore"):
            return fl(args[0]).astype(np.int64).astype(dtype) if element != "i1" else fl(args[0]).astype(np.bool_)
    if name in ("arith.extui", "arith.index_castui"):
        source = _elem(op.operands[0].type)
        return _unsigned(args[0], source).astype(_compute_dtype(element))
    if name == "arith.sitofp":
        return _cast_to(fl(args[0]).astype(np.float64), element)
    if name == "arith.uitofp":
        return _cast_to(_unsigned(args[0], _elem(op.operands[0].type)).astype(np.float64), element)
    if name == "arith.fptosi":
        with np.errstate(all="ignore"):
            if element == "i1":
                return np.trunc(fl(args[0])) != 0
            return np.trunc(fl(args[0])).astype(np.int64).astype(_compute_dtype(element))
    if name in ("arith.extf", "arith.truncf"):
        return _cast_to(args[0], element)
    if name == "arith.bitcast":
        # The same bits read as another type of the same width (bf16 through its 16-bit pattern).
        source = _elem(op.operands[0].type)
        bits = _bf16_bits(args[0]) if source == "bf16" else fl(args[0]).astype(storage_dtype(source))
        target = np.uint16 if element == "bf16" else storage_dtype(element)
        if bits.dtype.itemsize != np.dtype(target).itemsize or np.bool_ in (bits.dtype, np.dtype(target)):
            raise LinalgNumpyError(f"arith.bitcast from {source} to {element} is not evaluated")
        cast = _contiguous(bits).view(target)
        return _bf16_values(cast) if element == "bf16" else cast
    if name == "math.fma":
        return _fma(args[0], args[1], args[2], element)
    if name.startswith("math."):
        x = fl(args[0])
        if name in {"math.exp", "math.sin", "math.cos", "math.tanh", "math.powf"}:
            from merlin.runtime.host_math_reference import active_host_math

            selected = active_host_math()
            # The portable recipe wraps binary32 symbols only. Other declared
            # types retain their existing native evaluator, just as the emitted
            # program retains the unwrapped binary64 libm symbols.
            if selected is not None and name in selected.operations and element == "f32":
                if name == "math.powf":
                    return selected.evaluate(
                        name, np.asarray(x, dtype=np.float32), np.asarray(fl(args[1]), dtype=np.float32)
                    )
                return selected.evaluate(name, np.asarray(x, dtype=np.float32))
        with np.errstate(all="ignore"):
            if name == "math.exp":
                y = np.exp(x)
            elif name == "math.erf":
                y = _erf(x)
            elif name == "math.tanh":
                y = np.tanh(x)
            elif name == "math.rsqrt":
                y = 1.0 / np.sqrt(x)
            elif name == "math.sqrt":
                y = np.sqrt(x)
            elif name == "math.roundeven":
                y = np.rint(x)
            elif name == "math.round":
                y = np.sign(x) * np.floor(np.abs(x) + 0.5)
            elif name == "math.floor":
                y = np.floor(x)
            elif name == "math.ceil":
                y = np.ceil(x)
            elif name == "math.absf":
                y = np.abs(x)
            elif name == "math.sin":
                y = np.sin(x)
            elif name == "math.cos":
                y = np.cos(x)
            elif name == "math.log":
                y = np.log(x)
            elif name == "math.powf":
                y = np.power(x, fl(args[1]))
            else:
                raise LinalgNumpyError(f"no numpy evaluation for {name}")
        return _cast_to(y, element)
    raise LinalgNumpyError(f"no numpy evaluation for body op {name}")


# ------------------------------------------------------------------------------ structured ops


def _bind_extents(value_shape: Sequence[int], amap, extents: list[int | None]) -> None:
    """Record the extent of every dim ``amap`` names as a bare result, from the operand's shape."""
    from xdsl.ir.affine import AffineDimExpr

    for axis, expr in enumerate(amap.results):
        if isinstance(expr, AffineDimExpr) and extents[int(expr.position)] is None:
            extents[int(expr.position)] = int(value_shape[axis])


def _affine(expr, index_of: Callable[[int], np.ndarray]):
    """An affine expression over the iteration space, as a broadcastable integer array."""
    from xdsl.ir.affine import AffineBinaryOpExpr, AffineBinaryOpKind, AffineConstantExpr, AffineDimExpr

    if isinstance(expr, AffineDimExpr):
        return index_of(int(expr.position))
    if isinstance(expr, AffineConstantExpr):
        return np.int64(int(expr.value))
    if isinstance(expr, AffineBinaryOpExpr):
        lhs, rhs = _affine(expr.lhs, index_of), _affine(expr.rhs, index_of)
        kind = expr.kind
        if kind == AffineBinaryOpKind.Add:
            return lhs + rhs
        if kind == AffineBinaryOpKind.Mul:
            return lhs * rhs
        if kind == AffineBinaryOpKind.FloorDiv:
            return np.floor_divide(lhs, rhs)
        if kind == AffineBinaryOpKind.CeilDiv:
            return -np.floor_divide(-lhs, rhs)
        if kind == AffineBinaryOpKind.Mod:
            return np.mod(lhs, rhs)
    raise LinalgNumpyError(f"affine expression {expr} is not evaluated")


def _map_view(value: np.ndarray, amap, extents: list[int | None]) -> np.ndarray:
    """``value`` laid over the iteration space through ``amap``: a view broadcastable to it.

    A map of bare dims and constants is a transpose-and-reshape (no copy); a map with a compound
    result (a window's ``stride * out + tap``) is a gather over index arrays built from the map."""
    from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr

    n = int(amap.num_dims)
    array = np.asarray(value)
    if any(not isinstance(e, (AffineDimExpr, AffineConstantExpr)) for e in amap.results):
        if any(e is None for e in extents):
            raise LinalgNumpyError(f"a gathering map {amap} reads dims no other operand bounds")

        def index_of(d: int) -> np.ndarray:
            shape = [1] * n
            shape[d] = int(extents[d])
            return np.arange(int(extents[d]), dtype=np.int64).reshape(shape)

        index = tuple(np.asarray(_affine(e, index_of)) for e in amap.results)
        return array[index] if index else array
    dims: list[int] = []
    take: list[tuple[int, int]] = []
    for axis, expr in enumerate(amap.results):
        if isinstance(expr, AffineDimExpr):
            d = int(expr.position)
            if d in dims:
                raise LinalgNumpyError("an indexing map that names one dim twice is not evaluated")
            dims.append(d)
            size = array.shape[axis] if array.ndim else 1
            if extents[d] is None:
                extents[d] = int(size)
        else:
            take.append((axis, int(expr.value)))
    for axis, index in sorted(take, reverse=True):
        array = np.take(array, index, axis=axis)
    order = sorted(range(len(dims)), key=lambda i: dims[i])
    array = np.transpose(array, order) if array.ndim > 1 else array
    shape = [1] * n
    for i, d in enumerate(sorted(dims)):
        shape[d] = array.shape[i]
    return array.reshape(shape)


def _output_from_space(value: np.ndarray, amap, shape: Sequence[int]) -> np.ndarray:
    """A value over the iteration space (reduced dims dropped) laid out by an output map."""
    from xdsl.ir.affine import AffineDimExpr

    dims = [int(e.position) for e in amap.results if isinstance(e, AffineDimExpr)]
    if len(dims) != len(amap.results):
        raise LinalgNumpyError("an output map with a constant result is not evaluated")
    array = np.asarray(value)
    keep = sorted(dims)
    # One axis per iteration dim; a dim the output does not name was reduced to extent one.
    dropped = tuple(d for d in range(array.ndim) if d not in keep)
    if dropped:
        if any(array.shape[d] != 1 for d in dropped):
            raise LinalgNumpyError("an output map drops a dim that was not reduced")
        array = np.squeeze(array, axis=dropped)
    order = [keep.index(d) for d in dims]
    array = np.transpose(array, order) if array.ndim > 1 else array
    return np.broadcast_to(array, tuple(shape))


_COMBINERS = {
    "arith.addf": np.add,
    "arith.addi": np.add,
    "arith.maximumf": np.maximum,
    "arith.minimumf": np.minimum,
    "arith.maxnumf": np.fmax,
    "arith.minnumf": np.fmin,
    "arith.maxsi": np.maximum,
    "arith.minsi": np.minimum,
    "arith.mulf": np.multiply,
    "arith.muli": np.multiply,
    "arith.andi": np.bitwise_and,
    "arith.ori": np.bitwise_or,
}


def _split_reduction(block, acc_args: Sequence[Any]):
    """``(combiner op, the operand that is not the accumulator)`` for each yielded value, or None."""
    yielded = list(block.last_op.operands)
    out = []
    for value, acc in zip(yielded, acc_args, strict=True):
        owner = getattr(value, "owner", None)
        if owner is None or owner.name not in _COMBINERS or len(owner.operands) != 2:
            return None
        a, b = owner.operands
        if a is acc:
            out.append((owner, b))
        elif b is acc:
            out.append((owner, a))
        else:
            return None
    return out


def _ordered_float_reduction(split) -> bool:
    """Scalar fast-math permission belongs to the actual combiner, never provenance/shape."""
    for combiner, _ in split:
        element = _elem(combiner.results[0].type)
        if element not in {"bf16", "f16", "f32", "f64"}:
            continue
        # NumPy promotes low-precision reduction intermediates. Reassociation does
        # not authorize changing each source operation's declared precision.
        if element in {"bf16", "f16"}:
            return True
        flags = getattr(getattr(combiner, "fastmath", None), "data", ())
        if not any(getattr(flag, "value", None) == "reassoc" for flag in flags):
            return True
    return False


def _depends_on(value, targets: set[int], block) -> bool:
    seen: set[int] = set()
    stack = [value]
    while stack:
        v = stack.pop()
        if id(v) in targets:
            return True
        owner = getattr(v, "owner", None)
        if owner is None or getattr(owner, "parent", None) is not block or id(owner) in seen:
            continue
        seen.add(id(owner))
        stack.extend(owner.operands)
    return False


def _contraction(op, block, element_in: list[str]):
    """``(a arg index, b arg index, a chain, b chain)`` when the body is cast(a) * cast(b) + acc."""
    acc = block.args[-1]
    split = _split_reduction(block, [acc])
    if split is None or split[0][0].name not in ("arith.addf", "arith.addi"):
        return None
    _combiner, product = split[0]
    # A plain sum (``acc + in``) adds a block argument, whose owner is the block, not an operation.
    owner = getattr(product, "owner", None)
    if getattr(owner, "name", None) not in ("arith.mulf", "arith.muli"):
        return None

    def chain(value):
        ops = []
        while not any(value is arg for arg in block.args):
            o = getattr(value, "owner", None)
            if o is None or o.name not in ("arith.extsi", "arith.extui", "arith.sitofp", "arith.extf", "arith.truncf"):
                return None
            ops.append(o)
            value = o.operands[0]
        index = next(i for i, arg in enumerate(block.args) if arg is value)
        return index, list(reversed(ops))

    left, right = chain(owner.operands[0]), chain(owner.operands[1])
    if left is None or right is None or left[0] == right[0] or max(left[0], right[0]) >= len(block.args) - 1:
        return None
    return left, right, owner


def _einsum_letters(n: int) -> str:
    return "abcdefghijklmnopqrstuvwxyz"[:n]


def _generic(op, env: dict[int, Any], constants: ConstantCache | None = None) -> list[np.ndarray]:
    from xdsl.ir.affine import AffineDimExpr

    maps = [m.data for m in op.indexing_maps.data]
    iterators = [str(i.data) for i in op.iterator_types.data]
    n = len(iterators)
    inputs, outputs = list(op.inputs), list(op.outputs)
    operands = inputs + outputs
    extents: list[int | None] = [None] * n
    elements = [_elem(v.type) for v in operands]
    raw = [env[id(v)] for v in operands]
    for r, m in zip(raw, maps, strict=True):
        _bind_extents(np.shape(r), m, extents)
    views = [_map_view(_load(r, e), m, extents) for r, e, m in zip(raw, elements, maps, strict=True)]
    block = op.body.blocks[0]
    reduced = [d for d, kind in enumerate(iterators) if kind == "reduction"]
    if any(e is None for e in extents):
        raise LinalgNumpyError(f"iteration extents {extents} are not all bound by an operand")
    shapes = [_shape(v.type) for v in outputs]

    if reduced:
        contraction = _contraction(op, block, elements)
        if contraction is not None and len(outputs) == 1:
            (li, lchain), (ri, rchain), product = contraction
            benv: dict[int, Any] = {}

            def through(index, chain_ops):
                value = _load(raw[index], elements[index])
                benv[id(block.args[index])] = value
                for c in chain_ops:
                    benv[id(c.results[0])] = _body_op(c, benv)
                return benv[id(chain_ops[-1].results[0])] if chain_ops else value

            a, b = through(li, lchain), through(ri, rchain)
            letters = _einsum_letters(n)

            def subscript(m):
                return "".join(letters[int(e.position)] for e in m.results if isinstance(e, AffineDimExpr))

            if any(not isinstance(e, AffineDimExpr) for m in maps for e in m.results):
                raise LinalgNumpyError("a contraction whose maps take constants is not evaluated")
            spec = f"{subscript(maps[li])},{subscript(maps[ri])}->{subscript(maps[-1])}"
            out_element = elements[-1]
            if np.issubdtype(np.asarray(a).dtype, np.integer):
                # EXACT: products of int8-range operands and their sums stay far below 2**53, so float64
                # BLAS adds them without rounding. Wider operands take the exact integer path.
                def absmax(x, index):
                    return _derived(
                        constants, op, ("absmax", index), raw[index], lambda: np.abs(np.asarray(x)).max(initial=0)
                    )

                def wide(x, index):
                    return _derived(constants, op, ("f64", index), raw[index], lambda: np.asarray(x, np.float64))

                small = max(absmax(a, li), absmax(b, ri)) <= 1 << 15
                depth = int(np.prod([extents[d] for d in reduced]))
                if small and depth < (1 << 20):
                    summed = np.einsum(spec, wide(a, li), wide(b, ri), optimize=True)
                    summed = np.rint(summed).astype(np.int64)
                else:
                    summed = np.einsum(spec, np.asarray(a, np.int64), np.asarray(b, np.int64))
                init = np.asarray(_load(raw[-1], out_element)).astype(np.int64)
                total = summed + init
            else:
                summed = np.einsum(spec, np.asarray(a, np.float32), np.asarray(b, np.float32), optimize=True)
                total = summed.astype(np.float32) + np.asarray(_load(raw[-1], out_element), np.float32)
            return [_store(_cast_to(total, out_element) if out_element == "bf16" else total, out_element, shapes[0])]

    space = tuple(int(e) for e in extents)

    def index_of(d: int) -> np.ndarray:
        shape = [1] * n
        shape[d] = space[d]
        return np.arange(space[d], dtype=np.int64).reshape(shape)

    benv = {id(arg): view for arg, view in zip(block.args, views, strict=True)}
    if not reduced:
        _body_region(block, benv, env, index_of)
        results = []
        for value, amap, shape, element in zip(
            block.last_op.operands, maps[len(inputs) :], shapes, elements[len(inputs) :], strict=True
        ):
            full = np.broadcast_to(np.asarray(benv[id(value)]), space)
            results.append(_contiguous(_store(_output_from_space(full, amap, shape), element)))
        return results

    # General reductions use the original source loop unless their arithmetic permits the folded
    # path. That path computes the input-only side, folds it, then combines the initial output once.
    accs = list(block.args[len(inputs) :])
    split = _split_reduction(block, accs)
    if split is None or _ordered_float_reduction(split):
        return _sequential_reduction(op, block, env, views, space, reduced, maps, shapes, elements, len(inputs))
    acc_ids = {id(a) for a in accs}
    for body in block.ops:
        if body.name == "linalg.yield":
            break
        if any(body is s[0] for s in split):
            continue
        if any(_depends_on(r, acc_ids, block) for r in body.results):
            raise LinalgNumpyError("a reduction body that reads its accumulator before combining is not evaluated")
        if body.name == "tensor.extract":
            source = env[id(body.operands[0])]
            idx = tuple(np.asarray(benv[id(o)]).astype(np.int64) for o in body.operands[1:])
            benv[id(body.results[0])] = _load(np.asarray(source)[idx], _elem(body.operands[0].type))
            continue
        benv[id(body.results[0])] = _body_op(body, _Chain(benv, env), index_of=index_of)
    results = []
    for (combiner, x), acc_view, amap, shape, element in zip(
        split, views[len(inputs) :], maps[len(inputs) :], shapes, elements[len(inputs) :], strict=True
    ):
        ufunc = _COMBINERS[combiner.name]
        full = np.broadcast_to(np.asarray(benv[id(x)]), space)
        folded = ufunc.reduce(full, axis=tuple(reduced), keepdims=True)
        with np.errstate(all="ignore"):
            total = _cast_to(ufunc(np.asarray(acc_view), folded), element)
        results.append(_contiguous(_store(_output_from_space(total, amap, shape), element)))
    return results


#: A source-ordered reduction runs one reduction index at a time; bound that interpreter work.
SEQUENTIAL_REDUCTION_LIMIT = 1 << 16


def _sequential_reduction(op, block, env, views, space, reduced, maps, shapes, elements, n_inputs):
    """A reduction evaluated as the IR states it: one reduction index after another, the body run over
    every parallel point at once with the accumulators carried between steps. Exact for any body (an
    arg-min's value/index pair included), and bounded, because it is a loop in Python."""
    steps = int(np.prod([space[d] for d in reduced], dtype=np.int64))
    if steps > SEQUENTIAL_REDUCTION_LIMIT:
        raise LinalgNumpyError(
            f"a source-ordered reduction runs {steps} steps, over the "
            f"{SEQUENTIAL_REDUCTION_LIMIT} this evaluator steps through"
        )
    n = len(space)
    carried = [
        np.array(np.broadcast_to(v, [1 if d in reduced else space[d] for d in range(n)])) for v in views[n_inputs:]
    ]
    for point in np.ndindex(*[space[d] for d in reduced]):
        at = dict(zip(reduced, point, strict=True))

        def index_of(d: int, at=at) -> np.ndarray:
            if d in at:
                return np.int64(at[d])
            shape = [1] * n
            shape[d] = space[d]
            return np.arange(space[d], dtype=np.int64).reshape(shape)

        benv: dict[int, Any] = {}
        for arg, view in zip(block.args[:n_inputs], views[:n_inputs], strict=True):
            index = tuple(slice(at[d], at[d] + 1) if d in at and view.shape[d] != 1 else slice(None) for d in range(n))
            benv[id(arg)] = view[index]
        for arg, value in zip(block.args[n_inputs:], carried, strict=True):
            benv[id(arg)] = value
        yielded = _body_region(block, benv, env, index_of)
        carried = [np.array(np.broadcast_to(v, c.shape)) for v, c in zip(yielded, carried, strict=True)]
    return [
        _contiguous(_store(_output_from_space(value, amap, shape), element))
        for value, amap, shape, element in zip(carried, maps[n_inputs:], shapes, elements[n_inputs:], strict=True)
    ]


def _body_region(block, benv: dict[int, Any], env, index_of: Callable[[int], np.ndarray]) -> list[Any]:
    """Run a structured op's scalar region over arrays (one value per point of the iteration space);
    every result lands in ``benv`` and what its terminator yields is returned.

    A body may read a tensor element by element (``tensor.extract`` at computed indices) and may hold
    a loop: an ``scf.for`` whose bounds are the same at every point runs one iteration after another,
    its ``iter_args`` carried as arrays, so a source-ordered scalar recurrence keeps its order.
    """
    for body in block.ops:
        if body.name in ("linalg.yield", "scf.yield"):
            return [benv[id(v)] if id(v) in benv else env[id(v)] for v in body.operands]
        if body.name == "tensor.extract":
            source = benv[id(body.operands[0])] if id(body.operands[0]) in benv else env[id(body.operands[0])]
            element = _elem(body.operands[0].type)
            idx = tuple(np.asarray(benv[id(o)] if id(o) in benv else env[id(o)]).astype(np.int64)
                        for o in body.operands[1:])  # fmt: skip
            benv[id(body.results[0])] = _load(np.asarray(source)[idx], element)
            continue
        if body.name == "cf.assert":
            evaluate_op(body, _Chain(benv, env))  # the program's own check, applied at every point
            continue
        if body.name == "scf.for":
            for result, value in zip(body.results, _body_for(body, benv, env, index_of), strict=True):
                benv[id(result)] = value
            continue
        benv[id(body.results[0])] = _body_op(body, _Chain(benv, env), index_of=index_of)
    return []


def _body_for(op, benv: dict[int, Any], env, index_of: Callable[[int], np.ndarray]) -> list[Any]:
    """An ``scf.for`` nested in a structured body: the iterations in order, the carried values arrays."""
    scope = _Chain(benv, env)
    bounds = [np.asarray(scope[id(o)]) for o in list(op.operands)[:3]]
    if any(bound.size != 1 for bound in bounds):
        raise LinalgNumpyError("an scf.for whose bounds differ between points of the iteration space is not evaluated")
    lower, upper, step = (int(bound.reshape(())) for bound in bounds)
    if step <= 0:
        raise LinalgNumpyError(f"an scf.for with step {step} does not terminate")
    trips = len(range(lower, upper, step))
    if trips > SEQUENTIAL_REDUCTION_LIMIT:
        raise LinalgNumpyError(
            f"an scf.for runs {trips} iterations, over the {SEQUENTIAL_REDUCTION_LIMIT} this evaluator steps through"
        )
    carried = [scope[id(o)] for o in list(op.operands)[3:]]
    block = op.regions[0].blocks[0]
    for i in range(lower, upper, step):
        benv[id(block.args[0])] = np.int64(i)
        for arg, value in zip(block.args[1:], carried, strict=True):
            benv[id(arg)] = value
        carried = _body_region(block, benv, env, index_of)
    return carried


class _Chain(dict):
    """A body environment that falls back to the enclosing function's (a body may read values
    defined outside it, e.g. a scalar constant or a tensor it extracts from)."""

    def __init__(self, inner: dict, outer: dict):
        super().__init__()
        self.inner, self.outer = inner, outer

    def __getitem__(self, key):
        if key in self.inner:
            return self.inner[key]
        return self.outer[key]

    def __contains__(self, key):
        return key in self.inner or key in self.outer


def _reduce(op, env) -> list[np.ndarray]:
    dims = tuple(_ints(op.properties["dimensions"]))
    count = len(op.results)
    inputs, inits = list(op.operands)[:count], list(op.operands)[count:]
    block = op.regions[0].blocks[0]
    split = _split_reduction(block, list(block.args[count:]))
    if split is None or any(x is not arg for (_c, x), arg in zip(split, block.args[:count], strict=True)):
        raise LinalgNumpyError("a linalg.reduce whose body is not combiner(input, acc) is not evaluated")
    if _ordered_float_reduction(split):
        return _ordered_reduce(op, env, dims)
    out = []
    for (combiner, _x), value, init, result in zip(split, inputs, inits, op.results, strict=True):
        element = _elem(result.type)
        data = _load(env[id(value)], _elem(value.type))
        folded = _COMBINERS[combiner.name].reduce(np.asarray(data), axis=dims)
        with np.errstate(all="ignore"):
            total = _COMBINERS[combiner.name](np.asarray(_load(env[id(init)], element)), folded)
        out.append(_contiguous(_store(_cast_to(total, element), element, _shape(result.type))))
    return out


def _ordered_reduce(op, env, dims) -> list[np.ndarray]:
    """Interpret the original scalar region at each reduction point, with the original init."""
    count = len(op.results)
    inputs, inits = list(op.operands)[:count], list(op.operands)[count:]
    block = op.regions[0].blocks[0]
    values = [_load(env[id(value)], _elem(value.type)) for value in inputs]
    carried = [np.array(_load(env[id(value)], _elem(value.type))) for value in inits]
    shape = values[0].shape
    steps = math.prod(shape[dim] for dim in dims)
    if steps > SEQUENTIAL_REDUCTION_LIMIT:
        raise LinalgNumpyError(
            f"a source-ordered reduction runs {steps} steps, over the "
            f"{SEQUENTIAL_REDUCTION_LIMIT} this evaluator steps through"
        )
    for point in np.ndindex(*(shape[dim] for dim in dims)):
        at = dict(zip(dims, point, strict=True))
        selection = tuple(at.get(dim, slice(None)) for dim in range(len(shape)))
        arguments = [value[selection] for value in values] + carried
        local = {id(arg): value for arg, value in zip(block.args, arguments, strict=True)}
        for body in block.ops:
            if body.name == "linalg.yield":
                carried = [np.array(local[id(value)]) for value in body.operands]
                break
            local[id(body.results[0])] = _body_op(body, _Chain(local, env))
    return [
        _contiguous(_store(value, _elem(result.type), _shape(result.type)))
        for value, result in zip(carried, op.results, strict=True)
    ]


def _scf_for(op, env) -> list[Any]:
    lower, upper, step = (int(env[id(o)]) for o in list(op.operands)[:3])
    carried = [env[id(o)] for o in list(op.operands)[3:]]
    block = op.regions[0].blocks[0]
    for i in range(lower, upper, step):
        local = dict(env)
        local[id(block.args[0])] = i
        for arg, value in zip(block.args[1:], carried, strict=True):
            local[id(arg)] = value
        carried = _region(block, local)
    return carried


def _region(block, env) -> list[Any]:
    """Run a region's ops over scalars/tensors; returns what its terminator yields."""
    for op in block.ops:
        if op.name in ("scf.yield", "linalg.yield", "tensor.yield"):
            return [env[id(o)] for o in op.operands]
        for result, value in zip(op.results, evaluate_op(op, env), strict=True):
            env[id(result)] = value
    return []


def evaluate_op(op, env: Mapping[int, Any], *, constants: ConstantCache | None = None) -> list[Any]:
    """Every result of one top-level (or region) operation, from the values in ``env`` by ``id``.

    ``constants`` lets a contraction reuse the float64 copy and magnitude of an operand that is a
    constant of that cache (the same bytes on every evaluation)."""
    from merlin.runtime import dispatch_runtime as DR

    name = op.name
    if name == "linalg.generic":
        return _generic(op, env, constants)
    if name == "linalg.reduce":
        return _reduce(op, env)
    if name == "linalg.transpose":
        permutation = _ints(op.properties["permutation"])
        return [_contiguous(np.transpose(env[id(op.operands[0])], permutation))]
    if name == "linalg.matmul":
        maps = op.properties.get("indexing_maps")
        a, b, c = (env[id(o)] for o in op.operands)
        element = _elem(op.results[0].type)
        if maps is not None:
            identity = [str(m.data) for m in maps.data] == [
                "(d0, d1, d2) -> (d0, d2)",
                "(d0, d1, d2) -> (d2, d1)",
                "(d0, d1, d2) -> (d0, d1)",
            ]
            if not identity:
                raise LinalgNumpyError(f"linalg.matmul with maps {[str(m.data) for m in maps.data]} is not evaluated")
        product = np.matmul(_load(a, _elem(op.operands[0].type)), _load(b, _elem(op.operands[1].type)))
        return [_store(_cast_to(product + _load(c, element), element), element)]
    if name == "linalg.fill":
        value, init = env[id(op.operands[0])], op.operands[1]
        element = _elem(init.type)
        return [_contiguous(_store(np.full(_shape(init.type), value), element))]
    if name == "scf.for":
        return _scf_for(op, dict(env))
    if name == "scf.if":
        taken = op.regions[0] if bool(env[id(op.operands[0])]) else op.regions[1]
        if not taken.blocks:
            return []
        return _region(taken.blocks[0], dict(env))
    if name == "tensor.splat":
        element = _elem(op.results[0].type)
        value = env[id(op.operands[0])]
        return [_contiguous(_store(np.full(_shape(op.results[0].type), value, _compute_dtype(element)), element))]
    if name == "arith.constant":
        return [_constant(op)]
    if name == "tensor.empty":
        rtype = op.results[0].type
        dynamic = iter(op.operands)
        shape = [int(env[id(next(dynamic))]) if d <= _DYN // 2 else d for d in _shape(rtype)]
        return [np.zeros(shape, storage_dtype(_elem(rtype)))]
    if name == "tensor.extract":
        source = env[id(op.operands[0])]
        index = tuple(int(env[id(o)]) for o in op.operands[1:])
        element = _elem(op.operands[0].type)
        value = np.asarray(source)[index]
        return [_load(np.asarray(value), element)[()]]
    if name == "tensor.insert":
        scalar = env[id(op.operands[0])]
        element = _elem(op.operands[1].type)
        destination = np.array(env[id(op.operands[1])], copy=True)
        index = tuple(int(env[id(o)]) for o in op.operands[2:])
        destination[index] = _store(np.asarray(scalar), element)
        return [destination]
    if name.startswith("arith.") or name.startswith("math."):
        return [_body_op(op, env)]
    if name == "cf.assert":
        # The program's own runtime check: evaluated, never skipped, so a violated bound stops the oracle
        # exactly where the compiled program would trap.
        if not np.all(np.asarray(env[id(op.operands[0])]).astype(np.bool_)):
            message = op.properties.get("msg") or op.attributes.get("msg")
            raise LinalgNumpyError(f"cf.assert failed: {getattr(message, 'data', message)}")
        return []
    if name == "tensor.dim":
        source = np.asarray(env[id(op.operands[0])])
        axis = int(env[id(op.operands[1])])
        if not 0 <= axis < source.ndim:
            raise LinalgNumpyError(f"tensor.dim index {axis} is outside rank {source.ndim}")
        return [np.int64(source.shape[axis])]
    if name in (
        "tensor.collapse_shape",
        "tensor.expand_shape",
        "tensor.concat",
        "tensor.pad",
        "tensor.extract_slice",
        "tensor.insert_slice",
        "tensor.from_elements",
        "tensor.cast",
        "tensor.reshape",
    ):
        if name in ("tensor.cast",):
            return [env[id(op.operands[0])]]
        return [DR._eval_view(op, env)]
    raise LinalgNumpyError(f"no numpy evaluation for {name}")


# ------------------------------------------------------------------------------------- constants


class ConstantCache:
    """Values that are the same on every evaluation, and the work done on them, kept across evaluations.

    A caller marks the arguments that are constant (a model's weights, buffers and lifted tensors, loaded
    once). :func:`evaluate` then computes an op that reads only constants once, and reuses its results on
    every later evaluation; those results are constants in turn. A contraction keeps the float64 copy and
    the magnitude of a constant operand. A constant is identified by the object itself: the cache holds a
    reference to every value and op it keys by ``id``, so no other object can take that ``id``, and it
    marks every constant array read-only, so a value that is reused is never written.

    An op is hoisted only when every value it reads is a constant: its operands, and nothing that its
    regions read from outside the op (a body reading an enclosing value is not covered by its operands,
    and that value may change between evaluations). ``func.call`` is never hoisted.
    """

    #: Ops with no operands whose results depend only on the op itself.
    NULLARY = frozenset({"tensor.empty", "arith.constant"})

    def __init__(self) -> None:
        self._values: dict[int, Any] = {}
        self._results: dict[tuple[int, tuple[int, ...]], tuple[Any, list[Any]]] = {}
        self._contained: dict[int, tuple[Any, bool]] = {}
        self._derived: dict[tuple[Any, ...], tuple[Any, Any, Any]] = {}
        self._numeric_contract: dict[str, Any] | None = None

    def bind_numeric_contract(self) -> None:
        """Refuse reuse of hoisted arithmetic under a different selected numerical implementation."""
        current = numeric_contract()
        if self._numeric_contract is None:
            self._numeric_contract = current
        elif self._numeric_contract != current:
            raise LinalgNumpyError("constant cache numerical policy differs from this evaluation")

    def __len__(self) -> int:
        return len(self._values)

    def mark(self, value: Any) -> Any:
        """``value`` is a constant from now on (an array is made read-only); returns it."""
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
        self._values[id(value)] = value
        return value

    def is_constant(self, value: Any) -> bool:
        return id(value) in self._values

    def self_contained(self, op) -> bool:
        """No value read inside ``op``'s regions is defined outside ``op``."""
        key = id(op)
        if key not in self._contained:
            inner = [o for o in op.walk() if o is not op]
            inside: set[int] = set()
            for o in [op, *inner]:
                for region in o.regions:
                    for block in region.blocks:
                        inside.update(id(arg) for arg in block.args)
            for o in inner:
                inside.update(id(result) for result in o.results)
            contained = all(id(operand) in inside for o in inner for operand in o.operands)
            self._contained[key] = (op, contained)
        return self._contained[key][1]

    def hoistable(self, op, env: Mapping[int, Any]) -> bool:
        """``op`` reads only constants (and is not a call), so its results are the same every time."""
        if op.name == "func.call":
            return False
        if op.operands:
            if not all(id(o) in env and self.is_constant(env[id(o)]) for o in op.operands):
                return False
        elif op.name not in self.NULLARY:
            return False
        return self.self_contained(op)

    def results(self, op, env: Mapping[int, Any]) -> list[Any]:
        """The results of a :meth:`hoistable` op: evaluated the first time, reused after."""
        key = (id(op), tuple(id(env[id(o)]) for o in op.operands))
        if key not in self._results:
            self._results[key] = (op, [self.mark(v) for v in evaluate_op(op, env, constants=self)])
        return self._results[key][1]

    def derived(self, op, what: Any, source: Any, compute: Callable[[], Any]) -> Any:
        """``compute()`` -- a function of ``source`` alone -- once per (op, what) while ``source`` is constant."""
        if not self.is_constant(source):
            return compute()
        key = (id(op), what, id(source))
        if key not in self._derived:
            self._derived[key] = (op, source, self.mark(compute()))
        return self._derived[key][2]


def _derived(constants: ConstantCache | None, op, what: Any, source: Any, compute: Callable[[], Any]) -> Any:
    return compute() if constants is None else constants.derived(op, what, source, compute)


def last_uses(block) -> dict[int, int]:
    """``{id(value): index of the last top-level op that reads it}`` (a region read counts at its op)."""
    last: dict[int, int] = {}
    for index, op in enumerate(block.ops):
        for inner in op.walk():
            for operand in inner.operands:
                last[id(operand)] = index
    return last


def evaluate(
    module,
    arguments: Sequence[np.ndarray],
    *,
    function: str | None = None,
    observe: Callable[[Any, int, list[Any], list[Any]], None] | None = None,
    keep: Callable[[Any], bool] | None = None,
    calls: Mapping[str, Callable[[list[Any]], list[Any]]] | None = None,
    constants: ConstantCache | None = None,
) -> list[np.ndarray]:
    """Every result of the module's function (``function`` or the one with a body) on ``arguments``.

    ``arguments`` are in the function's order and in STORAGE form (bf16 as its 16-bit pattern), as
    :func:`merlin.runtime.dispatch_runtime.resolve_forward_args` binds them. ``observe(op, index,
    results, operands)`` is called after each top-level op, with its results and the values of its
    operands; a value is released after its last reader unless ``keep(value)`` says otherwise, so a
    model larger than memory in intermediates still evaluates. ``calls`` answers a ``func.call`` by
    callee name (a function of the argument values returning the results); a call to a function the
    module DEFINES is evaluated in place, and any other call is refused.

    ``constants`` (a :class:`ConstantCache` kept across evaluations) evaluates an op that reads only
    constants once and reuses its results: the arguments the caller marked constant, and what such ops
    compute from them. The results are the same bytes either way.
    """
    if constants is not None:
        constants.bind_numeric_contract()
    defined = {op.sym_name.data: op for op in module.walk() if op.name == "func.func" and op.body.blocks}

    def call(op, env) -> list[Any]:
        callee = op.callee.string_value() if hasattr(op.callee, "string_value") else str(op.callee)
        values = [env[id(o)] for o in op.operands]
        if calls is not None and callee in calls:
            return list(calls[callee](values))
        if callee in defined:
            return evaluate(module, values, function=callee, calls=calls, constants=constants)
        raise LinalgNumpyError(f"func.call @{callee}: the module does not define it and no answer was given")

    functions = [op for op in module.walk() if op.name == "func.func" and op.body.blocks]
    if function is not None:
        functions = [f for f in functions if f.sym_name.data == function]
    if not functions:
        raise LinalgNumpyError(f"no function {function or ''} with a body")
    block = functions[0].body.blocks[0]
    if len(arguments) != len(block.args):
        raise LinalgNumpyError(f"the function takes {len(block.args)} arguments, {len(arguments)} were given")
    env: dict[int, Any] = {id(arg): value for arg, value in zip(block.args, arguments, strict=True)}
    values_of: dict[int, Any] = {id(arg): arg for arg in block.args}
    for op in block.ops:
        for result in op.results:
            values_of[id(result)] = result
    released: dict[int, list[int]] = {}
    for key, index in last_uses(block).items():
        released.setdefault(index, []).append(key)
    results: list[np.ndarray] = []
    for index, op in enumerate(block.ops):
        if op.name == "func.return":
            results = [np.asarray(env[id(o)]) for o in op.operands]
            break
        try:
            if op.name == "func.call":
                values = call(op, env)
            elif constants is not None and constants.hoistable(op, env):
                values = constants.results(op, env)
            else:
                values = evaluate_op(op, env, constants=constants)
        except LinalgNumpyError as error:
            region = op.attributes.get("prov.region_id")
            raise LinalgNumpyError(f"op {index} ({op.name}, {region}): {error}") from error
        for result, value in zip(op.results, values, strict=True):
            env[id(result)] = value
        if observe is not None:
            observe(op, index, values, [env.get(id(o)) for o in op.operands])
        for key in released.get(index, ()):
            value = values_of.get(key)
            if value is not None and (keep is None or not keep(value)):
                env.pop(key, None)
    return results
