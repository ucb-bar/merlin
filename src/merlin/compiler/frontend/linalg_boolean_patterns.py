"""Source-only proofs for three exact Boolean scalar-region forms.

These checks do not grant host support or certify lowering. A caller must still
bind every proved ordinal to its selected source, policy and linked artifact.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import copysign
from pathlib import Path

from merlin.compiler.frontend.linalg_patterns import (
    InvalidLinalgPattern,
    _checked_static_linalg_shell,
    _screen_static_linalg_source,
    validate_singleton_projection_map,
)

STATIC_BOOLEAN_SOURCE_BODY_SCHEMA = "merlin.static_boolean_body.v1"
DYNAMIC_BOOLEAN_CAST_SOURCE_BODY_SCHEMA = "merlin.dynamic_boolean_cast_body.v1"
_BOOLEAN_ORDERED_TYPES = {
    "i1_not": ("i1", "i1", "i1"),
    "f32_nonzero_to_i1": ("f32", "i1", "i1"),
    "i1_mul_singleton_projected": ("i1", "i1", "i1", "i1"),
}


def validate_static_boolean_source_body(declaration: object) -> dict[str, str]:
    """Validate a closed declaration; this does not grant host admission."""
    if (
        not isinstance(declaration, dict)
        or set(declaration) != {"schema", "operation"}
        or declaration.get("schema") != STATIC_BOOLEAN_SOURCE_BODY_SCHEMA
        or not isinstance(declaration.get("operation"), str)
        or declaration["operation"] not in _BOOLEAN_ORDERED_TYPES
    ):
        raise InvalidLinalgPattern("source_body requires a closed static Boolean v1 declaration")
    return dict(declaration)


def static_boolean_ordered_types(declaration: object) -> tuple[str, ...]:
    """Return exact input/init/result types of one validated Boolean form."""
    body = validate_static_boolean_source_body(declaration)
    return _BOOLEAN_ORDERED_TYPES[body["operation"]]


@dataclass(frozen=True)
class StaticBooleanPattern:
    """An exact typed source form, without a placement or numerical verdict."""

    operation: str
    shape: tuple[int, ...]
    ordered_types: tuple[str, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    input_maps: tuple[str, ...]


@dataclass(frozen=True)
class StaticBooleanSource:
    raw_sha256: str
    normalized_sha256: str
    ordinals: tuple[tuple[int, StaticBooleanPattern], ...]


def _checked_arith(op, expected_class, properties: set[str]) -> None:
    from xdsl.traits import Pure

    if type(op) is not expected_class or not type(op).has_trait(Pure):
        raise InvalidLinalgPattern("Boolean source body has an unregistered or impure arithmetic operation")
    if op.regions or op.successors or any(not name.startswith("prov.") for name in op.attributes):
        raise InvalidLinalgPattern("Boolean source arithmetic has an effect, region or unknown attribute")
    if not set(op.properties) <= properties:
        raise InvalidLinalgPattern("Boolean source arithmetic has an unknown property")


def _checked_yield(op, scalar_result) -> None:
    from xdsl.dialects.linalg.ops import YieldOp

    if (
        type(op) is not YieldOp
        or op.regions
        or op.successors
        or op.results
        or op.properties
        or any(not name.startswith("prov.") for name in op.attributes)
        or tuple(op.operands) != (scalar_result,)
    ):
        raise InvalidLinalgPattern("Boolean source body does not yield only the scalar result")


def _checked_true_i1(op) -> None:
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr

    _checked_arith(op, arith.ConstantOp, {"value"})
    value = op.properties.get("value")
    if (
        set(op.properties) != {"value"}
        or not isinstance(value, IntegerAttr)
        or str(value.type) != "i1"
        or value.value.data != -1
        or op.operands
        or len(op.results) != 1
        or str(op.results[0].type) != "i1"
    ):
        raise InvalidLinalgPattern("Boolean NOT requires exactly a typed true i1 constant")


def _checked_positive_zero_f32(op) -> None:
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import FloatAttr

    _checked_arith(op, arith.ConstantOp, {"value"})
    value = op.properties.get("value")
    if (
        set(op.properties) != {"value"}
        or not isinstance(value, FloatAttr)
        or str(value.type) != "f32"
        or value.value.data != 0.0
        or copysign(1.0, value.value.data) != 1.0
        or op.operands
        or len(op.results) != 1
        or str(op.results[0].type) != "f32"
    ):
        raise InvalidLinalgPattern("float-to-Boolean requires exactly typed positive zero f32")


def recognize_static_boolean_body(op) -> StaticBooleanPattern:
    """Recognize exact i1 NOT, f32 nonzero, or projected i1 multiplication.

    The three forms have separate closed SSA/type/flag checks. Provenance labels
    never select a form; the registered operation body does. Projected input
    maps are permitted only for i1 multiplication and only at singleton axes.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr
    from xdsl.ir.affine import AffineMap

    shell = _checked_static_linalg_shell(op, singleton_projection_inputs=True)
    args, body, types = shell.args, shell.body, shell.ordered_types
    names = tuple(inner.name for inner in body)

    if names == ("arith.constant", "arith.xori", "linalg.yield"):
        if len(op.inputs) != 1 or types != ("i1", "i1", "i1"):
            raise InvalidLinalgPattern("Boolean NOT has incorrect tensor types or arity")
        if any(mapping != AffineMap.identity(len(shell.shape)) for mapping in shell.input_maps):
            raise InvalidLinalgPattern("Boolean NOT requires an identity input map")
        const, scalar, yld = body
        _checked_true_i1(const)
        _checked_arith(scalar, arith.XOrIOp, set())
        if tuple(scalar.operands) != (args[0], const.results[0]) or len(scalar.results) != 1:
            raise InvalidLinalgPattern("Boolean NOT does not XOR its input with true")
        operation = "i1_not"
    elif names == ("arith.constant", "arith.cmpf", "linalg.yield"):
        if len(op.inputs) != 1 or types != ("f32", "i1", "i1"):
            raise InvalidLinalgPattern("float-to-Boolean has incorrect tensor types or arity")
        if any(mapping != AffineMap.identity(len(shell.shape)) for mapping in shell.input_maps):
            raise InvalidLinalgPattern("float-to-Boolean requires an identity input map")
        const, scalar, yld = body
        _checked_positive_zero_f32(const)
        _checked_arith(scalar, arith.CmpfOp, {"predicate", "fastmath"})
        predicate = scalar.properties.get("predicate")
        fastmath = scalar.properties.get("fastmath")
        if (
            not isinstance(predicate, IntegerAttr)
            or str(predicate.type) != "i64"
            or predicate.value.data != 13  # arith.cmpf UNE: unordered or not equal
            or (fastmath is not None and (not isinstance(fastmath, arith.FastMathFlagsAttr) or fastmath.data))
            or tuple(scalar.operands) != (args[0], const.results[0])
            or len(scalar.results) != 1
        ):
            raise InvalidLinalgPattern("float-to-Boolean requires unflagged UNE against zero")
        operation = "f32_nonzero_to_i1"
    elif names == ("arith.muli", "linalg.yield"):
        if len(op.inputs) != 2 or types != ("i1", "i1", "i1", "i1"):
            raise InvalidLinalgPattern("Boolean multiplication has incorrect tensor types or arity")
        scalar, yld = body
        _checked_arith(scalar, arith.MuliOp, {"overflowFlags"})
        flags = scalar.properties.get("overflowFlags")
        if (
            (flags is not None and (not isinstance(flags, arith.IntegerOverflowAttr) or flags.data))
            or tuple(scalar.operands) != args[:2]
            or len(scalar.results) != 1
        ):
            raise InvalidLinalgPattern("Boolean multiplication requires unflagged input-only MULI")
        operation = "i1_mul_singleton_projected"
    else:
        raise InvalidLinalgPattern("Boolean source body is not one of the three closed forms")

    if str(scalar.results[0].type) != "i1":
        raise InvalidLinalgPattern("Boolean source scalar result is not i1")
    _checked_yield(yld, scalar.results[0])
    try:
        op.verify()
    except Exception as exc:  # noqa: BLE001 - xDSL verification is a fail-closed source boundary
        raise InvalidLinalgPattern(f"Boolean source operation failed xDSL verification: {exc}") from exc
    return StaticBooleanPattern(
        operation,
        shell.shape,
        types,
        tuple(tuple(value.type.get_shape()) for value in op.inputs),
        tuple(str(mapping) for mapping in shell.input_maps),
    )


def validate_serialized_static_boolean_pattern(pattern: object) -> None:
    """Recheck a source-derived Boolean shape/map record before linked completion."""
    from xdsl.context import Context
    from xdsl.ir.affine import AffineMap
    from xdsl.parser import Parser

    if not isinstance(pattern, Mapping) or set(pattern) != {
        "operation",
        "shape",
        "ordered_types",
        "input_shapes",
        "input_maps",
    }:
        raise InvalidLinalgPattern("serialized Boolean source pattern is not closed")
    declaration = {"schema": STATIC_BOOLEAN_SOURCE_BODY_SCHEMA, "operation": pattern["operation"]}
    expected_types = static_boolean_ordered_types(declaration)
    shape, shapes, maps = pattern["shape"], pattern["input_shapes"], pattern["input_maps"]
    if (
        not isinstance(shape, (tuple, list))
        or not shape
        or any(type(dim) is not int or dim <= 0 for dim in shape)
        or not isinstance(shapes, (tuple, list))
        or len(shapes) != len(expected_types) - 2
        or not isinstance(maps, (tuple, list))
        or len(maps) != len(shapes)
        or not isinstance(pattern["ordered_types"], (tuple, list))
        or tuple(pattern["ordered_types"]) != expected_types
    ):
        raise InvalidLinalgPattern("serialized Boolean source has invalid static tensor types or shapes")
    for input_shape, text in zip(shapes, maps, strict=True):
        if (
            not isinstance(input_shape, (tuple, list))
            or any(type(dim) is not int or dim <= 0 for dim in input_shape)
            or not isinstance(text, str)
        ):
            raise InvalidLinalgPattern("serialized Boolean input shape or map is malformed")
        try:
            mapping = Parser(Context(), text).parse_affine_map()
        except Exception as exc:  # noqa: BLE001 - malformed untrusted serialized map must refuse
            raise InvalidLinalgPattern("serialized Boolean input map does not parse") from exc
        if str(mapping) != text:
            raise InvalidLinalgPattern("serialized Boolean input map is not canonical")
        if pattern["operation"] == "i1_mul_singleton_projected":
            validate_singleton_projection_map(mapping, tuple(shape), tuple(input_shape))
        elif mapping != AffineMap.identity(len(shape)) or tuple(input_shape) != tuple(shape):
            raise InvalidLinalgPattern("serialized Boolean unary input map is not identity")


def screen_static_boolean_source(path: Path, ordinals: tuple[int, ...]) -> StaticBooleanSource:
    """Bind every requested proved body to exact raw/normalized source bytes."""
    raw, normalized, evidence = _screen_static_linalg_source(path, ordinals, recognize_static_boolean_body)
    return StaticBooleanSource(raw, normalized, evidence)


@dataclass(frozen=True)
class DynamicBooleanCastPattern:
    """One exact dynamic rank-one unsigned extension, without a size bound."""

    operation: str
    shape: tuple[int, ...]
    ordered_types: tuple[str, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    input_maps: tuple[str, ...]
    shape_source: str


def validate_dynamic_boolean_cast_source_body(declaration: object) -> dict[str, str]:
    """Accept only the separate, closed dynamic Boolean extension contract."""
    if (
        not isinstance(declaration, dict)
        or set(declaration) != {"schema", "operation"}
        or declaration.get("schema") != DYNAMIC_BOOLEAN_CAST_SOURCE_BODY_SCHEMA
        or declaration.get("operation") != "i1_to_i64_extui"
    ):
        raise InvalidLinalgPattern("source_body requires the closed dynamic Boolean cast v1 declaration")
    return dict(declaration)


def dynamic_boolean_cast_ordered_types(declaration: object) -> tuple[str, ...]:
    validate_dynamic_boolean_cast_source_body(declaration)
    return ("i1", "i64", "i64")


def recognize_dynamic_boolean_cast_body(op) -> DynamicBooleanCastPattern:
    """Check value extension and runtime extent provenance, never grant placement.

    The bound on the runtime extent belongs to the source-linked caller, not to
    this shape-preserving scalar-body recognizer.
    """
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import (
        DYNAMIC_INDEX,
        AffineMapAttr,
        ArrayAttr,
        DenseArrayBase,
        IntegerAttr,
        NoneAttr,
        TensorType,
    )
    from xdsl.dialects.linalg.attrs import IteratorType, IteratorTypeAttr
    from xdsl.dialects.linalg.ops import GenericOp
    from xdsl.ir.affine import AffineMap

    if type(op) is not GenericOp or len(op.inputs) != 1 or len(op.outputs) != 1 or len(op.results) != 1:
        raise InvalidLinalgPattern("dynamic Boolean cast requires registered unary linalg.generic")
    if (
        op.successors
        or set(op.properties) != {"indexing_maps", "iterator_types", "operandSegmentSizes"}
        or any(not key.startswith("prov.") for key in op.attributes)
    ):
        raise InvalidLinalgPattern("dynamic Boolean cast has unknown effects or metadata")
    segments = op.properties["operandSegmentSizes"]
    if (
        not isinstance(segments, DenseArrayBase)
        or str(segments.elt_type) != "i32"
        or tuple(segments.iter_values()) != (1, 1)
    ):
        raise InvalidLinalgPattern("dynamic Boolean cast has incorrect operand segments")
    source, init, result = op.inputs[0], op.outputs[0], op.results[0]
    expected = ((source, "i1"), (init, "i64"), (result, "i64"))
    if (
        any(
            not isinstance(value.type, TensorType)
            or not isinstance(value.type.encoding, NoneAttr)
            or tuple(value.type.get_shape()) != (DYNAMIC_INDEX,)
            or str(value.type.element_type) != element
            for value, element in expected
        )
        or result.type != init.type
    ):
        raise InvalidLinalgPattern("dynamic Boolean cast requires same-rank dynamic i1/i64 tensors")
    maps = op.properties["indexing_maps"]
    iterators = op.properties["iterator_types"]
    identity = AffineMap.identity(1)
    if (
        not isinstance(maps, ArrayAttr)
        or len(maps.data) != 2
        or any(not isinstance(item, AffineMapAttr) or item.data != identity for item in maps.data)
        or not isinstance(iterators, ArrayAttr)
        or len(iterators.data) != 1
        or not isinstance(iterators.data[0], IteratorTypeAttr)
        or iterators.data[0].data != IteratorType.PARALLEL
    ):
        raise InvalidLinalgPattern("dynamic Boolean cast requires one identity-map parallel iterator")
    empty = init.owner
    if (
        type(empty) is not tensor.EmptyOp
        or empty.properties
        or empty.successors
        or empty.regions
        or any(not key.startswith("prov.") for key in empty.attributes)
        or tuple(empty.results) != (init,)
        or len(empty.operands) != 1
    ):
        raise InvalidLinalgPattern("dynamic Boolean cast requires a one-dimension output allocation")
    dim = empty.operands[0].owner
    if (
        type(dim) is not tensor.DimOp
        or dim.properties
        or dim.successors
        or dim.regions
        or any(not key.startswith("prov.") for key in dim.attributes)
        or len(dim.operands) != 2
        or dim.operands[0] is not source
        or len(dim.results) != 1
        or str(dim.results[0].type) != "index"
    ):
        raise InvalidLinalgPattern("dynamic Boolean cast output extent must come from its input")
    zero = dim.operands[1].owner
    if (
        type(zero) is not arith.ConstantOp
        or set(zero.properties) != {"value"}
        or zero.regions
        or zero.successors
        or any(not key.startswith("prov.") for key in zero.attributes)
        or not isinstance(value := zero.properties["value"], IntegerAttr)
        or str(value.type) != "index"
        or value.value.data != 0
        or tuple(zero.results) != (dim.operands[1],)
    ):
        raise InvalidLinalgPattern("dynamic Boolean cast dimension is not exactly input axis zero")
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        raise InvalidLinalgPattern("dynamic Boolean cast requires one scalar block")
    block = op.regions[0].block
    body = tuple(block.ops)
    if (
        tuple(str(arg.type) for arg in block.args) != ("i1", "i64")
        or len(body) != 2
        or type(body[0]) is not arith.ExtUIOp
        or tuple(body[0].operands) != (block.args[0],)
        or len(body[0].results) != 1
        or str(body[0].results[0].type) != "i64"
    ):
        raise InvalidLinalgPattern("dynamic Boolean cast body is not input-only i1 to i64 extui")
    _checked_arith(body[0], arith.ExtUIOp, set())
    _checked_yield(body[1], body[0].results[0])
    try:
        op.verify()
    except Exception as exc:  # noqa: BLE001 - malformed xDSL source fails closed
        raise InvalidLinalgPattern(f"dynamic Boolean cast failed xDSL verification: {exc}") from exc
    return DynamicBooleanCastPattern(
        "i1_to_i64_extui",
        (DYNAMIC_INDEX,),
        ("i1", "i64", "i64"),
        ((DYNAMIC_INDEX,),),
        (str(identity),),
        "input_dim_0",
    )


def validate_serialized_dynamic_boolean_cast_pattern(pattern: object) -> None:
    """Recheck the only legal serialized dynamic shape/map/value contract."""
    from xdsl.dialects.builtin import DYNAMIC_INDEX
    from xdsl.ir.affine import AffineMap

    fields = {"operation", "shape", "ordered_types", "input_shapes", "input_maps", "shape_source"}
    if (
        not isinstance(pattern, Mapping)
        or set(pattern) != fields
        or pattern["operation"] != "i1_to_i64_extui"
        or pattern["shape_source"] != "input_dim_0"
        or not isinstance(pattern["shape"], (tuple, list))
        or len(pattern["shape"]) != 1
        or type(pattern["shape"][0]) is not int
        or pattern["shape"][0] != DYNAMIC_INDEX
        or not isinstance(pattern["ordered_types"], (tuple, list))
        or tuple(pattern["ordered_types"]) != ("i1", "i64", "i64")
        or not isinstance(pattern["input_shapes"], (tuple, list))
        or len(pattern["input_shapes"]) != 1
        or not isinstance(pattern["input_shapes"][0], (tuple, list))
        or len(pattern["input_shapes"][0]) != 1
        or type(pattern["input_shapes"][0][0]) is not int
        or pattern["input_shapes"][0][0] != DYNAMIC_INDEX
        or not isinstance(pattern["input_maps"], (tuple, list))
        or tuple(pattern["input_maps"]) != (str(AffineMap.identity(1)),)
    ):
        raise InvalidLinalgPattern("serialized dynamic Boolean cast body is not closed")
