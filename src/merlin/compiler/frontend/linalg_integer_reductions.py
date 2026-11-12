"""Closed source-only witnesses for static i64 sum and prefix sum.

These witnesses describe prepared Linalg source. They do not admit host work,
establish frontend equivalence, or certify a linked lowering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from merlin.compiler.frontend.linalg_patterns import (
    InvalidLinalgPattern,
    _checked_static_indexed_linalg_shell,
    _screen_static_linalg_source,
    recognize_static_pointwise,
)


@dataclass(frozen=True)
class StaticIntegerReductionPattern:
    """Prepared reduction shape/body under an unbound index-width premise.

    ``input_type`` is i1 only when the i1-to-i64 predecessor is proved to be
    exact zero-extension; otherwise it describes the i64 reduction operand,
    without claiming anything about earlier frontend transformations.
    """

    operation: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    axis: tuple[int, ...]
    input_type: str
    index_bits_premise: int


@dataclass(frozen=True)
class StaticIntegerReductionSource:
    raw_sha256: str
    normalized_sha256: str
    ordinals: tuple[tuple[int, StaticIntegerReductionPattern], ...]


def _static_tensor(value, element: str, *, allow_scalar: bool = False) -> tuple[int, ...]:
    from xdsl.dialects.builtin import NoneAttr, TensorType

    typ = value.type
    if not isinstance(typ, TensorType) or str(typ.element_type) != element or not isinstance(typ.encoding, NoneAttr):
        raise InvalidLinalgPattern("integer reduction requires a default-encoded typed tensor")
    shape = tuple(typ.get_shape())
    if (not shape and not allow_scalar) or any(type(dim) is not int or dim <= 0 for dim in shape):
        raise InvalidLinalgPattern("integer reduction requires positive static tensor extents")
    return shape


def _checked_index_bounds(index_bits: int, *shapes: tuple[int, ...]) -> None:
    """Record a caller-supplied width premise, not a selected-build observation."""
    if type(index_bits) is not int or not 2 <= index_bits <= 128:
        raise InvalidLinalgPattern("integer reduction requires an explicit finite signed index-width premise")
    maximum = (1 << (index_bits - 1)) - 1
    if any(extent > maximum for shape in shapes for extent in shape):
        raise InvalidLinalgPattern("integer reduction loop extent exceeds the signed index-width premise")


def _checked_zero_init(value) -> object:
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import IntegerAttr

    splat = value.owner
    if (
        type(splat) is not tensor.SplatOp
        or value is not splat.result
        or tuple(splat.operands) != (splat.input,)
        or len(splat.results) != 1
        or splat.dynamicSizes
        or splat.properties
        or splat.regions
        or splat.successors
        or any(not name.startswith("prov.") for name in splat.attributes)
    ):
        raise InvalidLinalgPattern("integer reduction init is not a static tensor.splat")
    constant = splat.input.owner
    if (
        type(constant) is not arith.ConstantOp
        or splat.input is not constant.result
        or constant.operands
        or len(constant.results) != 1
        or constant.regions
        or constant.successors
        or set(constant.properties) != {"value"}
        or any(not name.startswith("prov.") for name in constant.attributes)
    ):
        raise InvalidLinalgPattern("integer reduction init is not a closed constant")
    literal = constant.properties["value"]
    if (
        not isinstance(literal, IntegerAttr)
        or str(literal.type) != "i64"
        or type(literal.value.data) is not int
        or literal.value.data != 0
        or str(constant.result.type) != "i64"
    ):
        raise InvalidLinalgPattern("integer reduction init is not exactly zero i64")
    try:
        constant.verify()
        splat.verify()
    except Exception as exc:  # noqa: BLE001 - owner verification is a fail-closed source boundary
        raise InvalidLinalgPattern(f"integer reduction zero init failed xDSL verification: {exc}") from exc
    return constant.result


def _checked_body_op(op, klass, properties: set[str]) -> None:
    from xdsl.traits import Pure

    if (
        type(op) is not klass
        or not type(op).has_trait(Pure)
        or op.regions
        or op.successors
        or any(not name.startswith("prov.") for name in op.attributes)
        or not set(op.properties) <= properties
    ):
        raise InvalidLinalgPattern("integer reduction body has an unknown operation or effect")


def _checked_add_and_yield(add, yld, lhs, rhs) -> None:
    from xdsl.dialects import arith
    from xdsl.dialects.linalg.ops import YieldOp

    _checked_body_op(add, arith.AddiOp, {"overflowFlags"})
    flags = add.properties.get("overflowFlags")
    if flags is not None and (not isinstance(flags, arith.IntegerOverflowAttr) or flags.data):
        raise InvalidLinalgPattern("integer reduction addition has overflow flags")
    if (
        tuple(add.operands) != (lhs, rhs)
        or len(add.results) != 1
        or str(add.results[0].type) != "i64"
        or type(yld) is not YieldOp
        or tuple(yld.operands) != (add.results[0],)
        or yld.regions
        or yld.successors
        or yld.results
        or yld.properties
        or any(not name.startswith("prov.") for name in yld.attributes)
    ):
        raise InvalidLinalgPattern("integer reduction does not yield its exact accumulator update")


def _checked_boolean_cast(value) -> None:
    from xdsl.dialects.linalg.ops import GenericOp

    cast = value.owner
    if type(cast) is not GenericOp or value is not cast.results[0]:
        raise InvalidLinalgPattern("Boolean sum input is not an exact extui generic")
    pattern = recognize_static_pointwise(cast)
    if pattern.operation != "arith.extui" or pattern.ordered_types != ("i1", "i64", "i64"):
        raise InvalidLinalgPattern("Boolean sum does not zero-extend i1 to i64")


def _recognize_sum(op, index_bits: int) -> StaticIntegerReductionPattern:
    from xdsl.dialects.builtin import DenseArrayBase, TensorType
    from xdsl.dialects.linalg.ops import GenericOp, ReduceOp

    if (
        type(op) is not ReduceOp
        or set(op.properties) != {"dimensions"}
        or any(not name.startswith("prov.") for name in op.attributes)
        or op.successors
        or len(op.results) != 1
        or len(op.regions) != 1
        or len(op.regions[0].blocks) != 1
    ):
        raise InvalidLinalgPattern("sum is not a registered one-result linalg.reduce")
    input_shape = _static_tensor(op.input, "i64")
    output_shape = _static_tensor(op.init, "i64", allow_scalar=True)
    if _static_tensor(op.results[0], "i64", allow_scalar=True) != output_shape or op.results[0].type != op.init.type:
        raise InvalidLinalgPattern("sum output differs from its initialized tensor")
    _checked_index_bounds(index_bits, input_shape, output_shape)
    dimensions = op.properties["dimensions"]
    if not isinstance(dimensions, DenseArrayBase) or str(dimensions.elt_type) != "i64":
        raise InvalidLinalgPattern("sum reduction dimensions are not typed i64")
    axis = tuple(dimensions.iter_values())
    if (
        not axis
        or any(type(dim) is not int or dim < 0 or dim >= len(input_shape) for dim in axis)
        or tuple(sorted(set(axis))) != axis
        or tuple(size for dim, size in enumerate(input_shape) if dim not in axis) != output_shape
    ):
        raise InvalidLinalgPattern("sum has invalid reduction axes or static output geometry")
    _checked_zero_init(op.init)
    block = op.regions[0].block
    args, body = tuple(block.args), tuple(block.ops)
    if tuple(str(arg.type) for arg in args) != ("i64", "i64") or len(body) != 2:
        raise InvalidLinalgPattern("sum body is not a two-argument i64 addition")
    _checked_add_and_yield(body[0], body[1], args[0], args[1])
    source_kind = "i64"
    owner = op.input.owner
    if (
        type(owner) is GenericOp
        and len(owner.inputs) == 1
        and isinstance(owner.inputs[0].type, TensorType)
        and str(owner.inputs[0].type.element_type) == "i1"
        and len(owner.regions) == 1
        and len(owner.regions[0].blocks) == 1
        and tuple(inner.name for inner in owner.regions[0].block.ops) == ("arith.extui", "linalg.yield")
    ):
        _checked_boolean_cast(op.input)
        source_kind = "i1"
    try:
        op.verify()
    except Exception as exc:  # noqa: BLE001 - xDSL is the fail-closed verification boundary
        raise InvalidLinalgPattern(f"sum failed xDSL verification: {exc}") from exc
    return StaticIntegerReductionPattern("sum", input_shape, output_shape, axis, source_kind, index_bits)


def _checked_index(op, dimension: int) -> None:
    from xdsl.dialects.builtin import IntegerAttr
    from xdsl.dialects.linalg.ops import IndexOp

    value = op.properties.get("dim")
    if (
        type(op) is not IndexOp
        or set(op.properties) != {"dim"}
        or not isinstance(value, IntegerAttr)
        or str(value.type) != "i64"
        or type(value.value.data) is not int
        or value.value.data != dimension
        or op.operands
        or len(op.results) != 1
        or str(op.results[0].type) != "index"
        or op.regions
        or op.successors
        or any(not name.startswith("prov.") for name in op.attributes)
    ):
        raise InvalidLinalgPattern("prefix sum index is not the selected parallel/reduction dimension")


def _recognize_cumsum(op, index_bits: int) -> StaticIntegerReductionPattern:
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr
    from xdsl.dialects.linalg.attrs import IteratorType
    from xdsl.ir.affine import AffineExpr, AffineMap

    shell = _checked_static_indexed_linalg_shell(op)
    if len(op.inputs) != 1 or len(op.outputs) != 1 or len(op.results) != 1:
        raise InvalidLinalgPattern("prefix sum requires one input and one initialized result")
    output_shape = _static_tensor(op.outputs[0], "i64")
    if _static_tensor(op.results[0], "i64") != output_shape or op.results[0].type != op.outputs[0].type:
        raise InvalidLinalgPattern("prefix sum result differs from initialized output")
    input_type = str(op.inputs[0].type.element_type)
    if input_type not in {"i1", "i64"}:
        raise InvalidLinalgPattern("prefix sum only proves Boolean or signed i64 input")
    input_shape = _static_tensor(op.inputs[0], input_type)
    if input_shape != output_shape:
        raise InvalidLinalgPattern("prefix sum input/output static shapes differ")
    rank = len(output_shape)
    if len(shell.shape) != rank + 1 or tuple(shell.shape[:rank]) != output_shape:
        raise InvalidLinalgPattern("prefix sum parallel loop bounds differ from output shape")
    _checked_index_bounds(index_bits, input_shape, output_shape, tuple(shell.shape))
    if tuple(shell.iterator_types) != (IteratorType.PARALLEL,) * rank + (IteratorType.REDUCTION,):
        raise InvalidLinalgPattern("prefix sum requires exactly one final reduction iterator")
    dims = tuple(AffineExpr.dimension(dim) for dim in range(rank))
    destination = AffineMap(rank + 1, 0, dims)
    maps = tuple(shell.indexing_maps)
    if len(maps) != 2 or maps[1] != destination:
        raise InvalidLinalgPattern("prefix sum destination map does not follow parallel indices")
    axes = tuple(
        axis
        for axis in range(rank)
        if maps[0] == AffineMap(rank + 1, 0, (*dims[:axis], AffineExpr.dimension(rank), *dims[axis + 1 :]))
    )
    if len(axes) != 1 or shell.shape[-1] != input_shape[axes[0]]:
        raise InvalidLinalgPattern("prefix sum input map does not substitute its bounded reduction axis")
    axis = axes[0]
    zero = _checked_zero_init(op.outputs[0])
    args, body = tuple(shell.args), tuple(shell.body)
    expected_names = (
        ("linalg.index", "linalg.index", "arith.cmpi", "arith.extui", "arith.select", "arith.addi", "linalg.yield")
        if input_type == "i1"
        else ("linalg.index", "linalg.index", "arith.cmpi", "arith.select", "arith.addi", "linalg.yield")
    )
    if tuple(inner.name for inner in body) != expected_names or tuple(str(arg.type) for arg in args) != (
        input_type,
        "i64",
    ):
        raise InvalidLinalgPattern("prefix sum body is not a closed masked i64 reduction")
    i, j, compare = body[:3]
    _checked_index(i, axis)
    _checked_index(j, rank)
    _checked_body_op(compare, arith.CmpiOp, {"predicate"})
    predicate = compare.properties.get("predicate")
    if (
        not isinstance(predicate, IntegerAttr)
        or str(predicate.type) != "i64"
        or predicate.value.data != 7  # unsigned less-or-equal: j <= i
        or tuple(compare.operands) != (j.results[0], i.results[0])
        or len(compare.results) != 1
        or str(compare.results[0].type) != "i1"
    ):
        raise InvalidLinalgPattern("prefix sum mask is not j <= i")
    if input_type == "i1":
        cast = body[3]
        _checked_body_op(cast, arith.ExtUIOp, set())
        if tuple(cast.operands) != (args[0],) or len(cast.results) != 1 or str(cast.results[0].type) != "i64":
            raise InvalidLinalgPattern("Boolean prefix sum does not zero-extend its input")
        selected, add, yld = body[4:]
        source = cast.results[0]
    else:
        selected, add, yld = body[3:]
        source = args[0]
    _checked_body_op(selected, arith.SelectOp, set())
    if (
        tuple(selected.operands) != (compare.results[0], source, zero)
        or len(selected.results) != 1
        or str(selected.results[0].type) != "i64"
    ):
        raise InvalidLinalgPattern("prefix sum does not select input or exact zero")
    _checked_add_and_yield(add, yld, args[1], selected.results[0])
    try:
        op.verify()
    except Exception as exc:  # noqa: BLE001 - xDSL is the fail-closed verification boundary
        raise InvalidLinalgPattern(f"prefix sum failed xDSL verification: {exc}") from exc
    return StaticIntegerReductionPattern("cumsum", input_shape, output_shape, (axis,), input_type, index_bits)


def recognize_static_integer_reduction(op, *, index_bits: int) -> StaticIntegerReductionPattern:
    """Recognize a closed prepared integer sum or cumsum, without admission."""
    from xdsl.dialects.linalg.ops import GenericOp, ReduceOp

    if type(op) is ReduceOp:
        return _recognize_sum(op, index_bits)
    if type(op) is GenericOp:
        return _recognize_cumsum(op, index_bits)
    raise InvalidLinalgPattern("not a registered integer sum or cumsum")


def screen_static_integer_reduction_source(
    path: Path, ordinals: tuple[int, ...], *, index_bits: int
) -> StaticIntegerReductionSource:
    """Reparse the complete source and bind checked operation ordinals to both byte hashes."""
    raw, normalized, evidence = _screen_static_linalg_source(
        path, ordinals, lambda op: recognize_static_integer_reduction(op, index_bits=index_bits)
    )
    return StaticIntegerReductionSource(raw, normalized, evidence)
