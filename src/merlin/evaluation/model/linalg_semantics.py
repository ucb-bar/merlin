"""Scalar semantics of the independent linalg evaluator.

The numeric contract, storage dtypes, bf16 rounding, comparison predicates, constants, casts and
the region-body operations :mod:`merlin.evaluation.model.linalg_evaluator` composes into structured ops.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

__all__ = [
    "LinalgNumpyError",
    "NUMERIC_CONTRACT_VERSION",
    "bf16_round",
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
    from merlin.runtime.host_math import active_host_math

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
        raise LinalgNumpyError(f"no numpy arithmetic for element type {element!r}")
    return _COMPUTE[element]


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
        from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, IntegerType

        if (
            type(value) is DenseIntOrFPElementsAttr
            and value.type == rtype
            and isinstance(value.get_element_type(), IntegerType)
            and element in ("i8", "i16", "i32", "i64")
            and len(value.data.data) % value.get_element_type().compile_time_size == 0
        ):
            # xDSL's scalar unpack format also specifies signedness and byte order.
            # Copy into native compute storage: constants remain writable until cached.
            values = np.frombuffer(value.data.data, dtype=np.dtype(value.get_element_type().format)).astype(
                _compute_dtype(element), copy=True
            )
        else:
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
    return np.asarray(int(raw), dtype=_compute_dtype(element))[()]


def _cast_to(value, element: str):
    """Values converted to ``element``'s compute form, bf16 rounded."""
    if element == "bf16":
        return bf16_round(np.asarray(value, np.float32))
    return np.asarray(value).astype(_compute_dtype(element))


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
            return fn(fl(args[0]).astype(dtype), fl(args[1]).astype(dtype)).astype(dtype)
    if name in ("arith.maxsi", "arith.minsi"):
        fn = np.maximum if name == "arith.maxsi" else np.minimum
        return fn(fl(args[0]), fl(args[1])).astype(_compute_dtype(element))
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
    if name == "arith.shrsi":
        # Arithmetic shift; an amount outside [0, width) is poison in the IR, so it is refused rather
        # than given whatever value numpy's shift would produce.
        width = int(element[1:])
        a, s = fl(args[0]), fl(args[1]).astype(np.int64)
        if np.any((s < 0) | (s >= width)):
            raise LinalgNumpyError(f"arith.shrsi by an amount outside [0, {width}) is poison")
        return np.right_shift(a.astype(_compute_dtype(element)), s.astype(_compute_dtype(element)))
    if name == "arith.cmpi":
        return _CMPI[_predicate(op)](fl(args[0]), fl(args[1]))
    if name == "arith.cmpf":
        return _cmpf(_predicate(op), args[0], args[1])
    if name == "arith.select":
        return np.where(fl(args[0]).astype(np.bool_), args[1], args[2])
    if name in ("arith.extsi", "arith.trunci", "arith.index_cast"):
        dtype = _compute_dtype(element)
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
        raise LinalgNumpyError("arith.bitcast is not evaluated")
    if name.startswith("math."):
        x = fl(args[0])
        if name in {"math.exp", "math.sin", "math.cos", "math.tanh", "math.powf"}:
            from merlin.runtime.host_math import active_host_math

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
