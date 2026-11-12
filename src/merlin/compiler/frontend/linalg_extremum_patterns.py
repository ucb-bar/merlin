"""Closed source-only signed-i64 minimum/first-index reductions.

The selected index width is a caller premise. Recognition does not grant host
placement, prove a compiler transformation, or qualify a multi-output ABI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from merlin.compiler.frontend.linalg_boolean_patterns import _checked_arith
from merlin.compiler.frontend.linalg_patterns import (
    InvalidLinalgPattern,
    _checked_static_indexed_linalg_shell,
    _screen_static_linalg_source,
)


@dataclass(frozen=True)
class StaticI64ArgminPattern:
    operation: str
    axis: int
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    ordered_types: tuple[str, ...]
    index_bits: int


@dataclass(frozen=True)
class StaticI64ArgminSource:
    raw_sha256: str
    normalized_sha256: str
    ordinals: tuple[tuple[int, StaticI64ArgminPattern], ...]


def _scalar(op, cls, operands, result_type, properties=frozenset()):
    from xdsl.dialects.linalg.ops import IndexOp

    if cls is IndexOp:
        # IndexOp reads its enclosing loop coordinate and is not marked Pure.
        # The checked shell and exact axis below supply that lexical context.
        if (
            type(op) is not IndexOp
            or op.regions
            or op.successors
            or set(op.properties) != {"dim"}
            or any(not name.startswith("prov.") for name in op.attributes)
        ):
            raise InvalidLinalgPattern("minimum coordinate is not a closed registered linalg.index")
    else:
        _checked_arith(op, cls, set(properties))
    if tuple(op.operands) != tuple(operands) or len(op.results) != 1 or str(op.results[0].type) != result_type:
        raise InvalidLinalgPattern("minimum body has inconsistent scalar SSA or type")
    return op.results[0]


def _comparison(op, lhs, rhs, predicate):
    from xdsl.dialects.arith import CmpiOp
    from xdsl.dialects.builtin import IntegerAttr

    result = _scalar(op, CmpiOp, (lhs, rhs), "i1", {"predicate"})
    attribute = op.properties.get("predicate")
    if not isinstance(attribute, IntegerAttr) or str(attribute.type) != "i64" or attribute.value.data != predicate:
        raise InvalidLinalgPattern("minimum body comparison has the wrong signedness or predicate")
    return result


def _seed(value, expected):
    from xdsl.dialects.arith import ConstantOp
    from xdsl.dialects.builtin import IntegerAttr
    from xdsl.dialects.tensor import SplatOp

    splat = value.owner
    _checked_arith(splat, SplatOp, set())
    if splat.dynamicSizes or tuple(splat.results) != (value,) or len(splat.operands) != 1:
        raise InvalidLinalgPattern("minimum seed is not a static scalar splat")
    constant = splat.input.owner
    _scalar(constant, ConstantOp, (), "i64", {"value"})
    attribute = constant.properties.get("value")
    if (
        not isinstance(attribute, IntegerAttr)
        or str(attribute.type) != "i64"
        or attribute.value.data != expected
        or splat.input is not constant.results[0]
    ):
        raise InvalidLinalgPattern("minimum value/index seed is not the required sentinel")


def recognize_static_i64_argmin(op, *, index_bits: int) -> StaticI64ArgminPattern:
    """Recognize lexicographic min(signed value, unsigned original index).

    Nonempty reduced axes and a maximum-i64 index sentinel make the first
    extremum invariant under reduction traversal order. This checks the source
    combiner and seeds only; binding ``index_bits`` to the emitted build and
    verifying frontend/lowering/numerical equivalence remain caller obligations.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr
    from xdsl.dialects.linalg.attrs import IteratorType
    from xdsl.dialects.linalg.ops import IndexOp, YieldOp
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    if type(index_bits) is not int or not 2 <= index_bits <= 128:
        raise InvalidLinalgPattern("minimum source requires an explicit valid index-width premise")
    shell = _checked_static_indexed_linalg_shell(op)
    rank = len(shell.shape)
    if len(op.inputs) != 1 or len(op.outputs) != 2 or shell.ordered_types != ("i64",) * 5:
        raise InvalidLinalgPattern("minimum source requires one i64 input and two initialized i64 outputs")
    axes = tuple(i for i, iterator in enumerate(shell.iterator_types) if iterator == IteratorType.REDUCTION)
    if len(axes) != 1:
        raise InvalidLinalgPattern("minimum source requires exactly one reduction axis")
    axis = axes[0]
    maximum = (1 << 63) - 1
    if shell.shape[axis] == 0 or any(extent > min(maximum, (1 << (index_bits - 1)) - 1) for extent in shell.shape):
        raise InvalidLinalgPattern("minimum axis is empty or an extent exceeds the index/i64 domain")
    output_shape = tuple(extent for i, extent in enumerate(shell.shape) if i != axis)
    output_map = AffineMap(rank, 0, tuple(AffineDimExpr(i) for i in range(rank) if i != axis))
    if (
        shell.input_shapes != (shell.shape,)
        or shell.output_shapes != (output_shape, output_shape)
        or shell.indexing_maps != (AffineMap.identity(rank), output_map, output_map)
    ):
        raise InvalidLinalgPattern("minimum source maps/shapes do not project exactly the reduced axis")
    for output in op.outputs:
        _seed(output, maximum)
    if tuple(inner.name for inner in shell.body) != (
        "linalg.index",
        "arith.index_cast",
        "arith.cmpi",
        "arith.cmpi",
        "arith.cmpi",
        "arith.andi",
        "arith.ori",
        "arith.select",
        "arith.select",
        "linalg.yield",
    ):
        raise InvalidLinalgPattern("minimum body is not the closed value/index combiner")
    index, cast, better, equal, earlier, tie, replace, value_select, index_select, yld = shell.body
    coordinate = _scalar(index, IndexOp, (), "index", {"dim"})
    dimension = index.properties.get("dim")
    if not isinstance(dimension, IntegerAttr) or str(dimension.type) != "i64" or dimension.value.data != axis:
        raise InvalidLinalgPattern("minimum index does not refer to its reduction axis")
    position = _scalar(cast, arith.IndexCastOp, (coordinate,), "i64")
    source, best_value, best_index = shell.args
    preferred = _comparison(better, source, best_value, 2)  # signed less-than
    same = _comparison(equal, source, best_value, 0)
    first = _comparison(earlier, position, best_index, 6)  # unsigned less-than
    first_tie = _scalar(tie, arith.AndIOp, (same, first), "i1")
    choose = _scalar(replace, arith.OrIOp, (preferred, first_tie), "i1")
    value = _scalar(value_select, arith.SelectOp, (choose, source, best_value), "i64")
    selected_index = _scalar(index_select, arith.SelectOp, (choose, position, best_index), "i64")
    if (
        type(yld) is not YieldOp
        or tuple(yld.operands) != (value, selected_index)
        or yld.results
        or yld.regions
        or yld.successors
        or yld.properties
        or any(not name.startswith("prov.") for name in yld.attributes)
    ):
        raise InvalidLinalgPattern("minimum body must yield the coupled value/index pair")
    return StaticI64ArgminPattern(
        "i64_min_first_index", axis, shell.shape, output_shape, shell.ordered_types, index_bits
    )


def screen_static_i64_argmin_source(path: Path, ordinals: tuple[int, ...], *, index_bits: int) -> StaticI64ArgminSource:
    raw, normalized, evidence = _screen_static_linalg_source(
        path, ordinals, lambda op: recognize_static_i64_argmin(op, index_bits=index_bits)
    )
    return StaticI64ArgminSource(raw, normalized, evidence)
