"""Closed prepared-source witness for one-axis f32 maximum reductions.

This describes the selected ``arith.maximumf`` source operation only. It does
not grant host placement or assert equivalence to a framework reduction:
signed-zero selection and reduction order may differ.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import isinf, prod
from pathlib import Path

from merlin.compiler.frontend.linalg_patterns import InvalidLinalgPattern, _screen_static_linalg_source

STATIC_F32_MAXIMUM_SOURCE_BODY_SCHEMA = "merlin.static_f32_maximum_source_body.v1"


def validate_static_f32_maximum_source_body(declaration: object) -> dict[str, str]:
    """Validate a closed prepared-source selector without granting placement."""
    if (
        not isinstance(declaration, dict)
        or set(declaration) != {"schema", "operation"}
        or declaration.get("schema") != STATIC_F32_MAXIMUM_SOURCE_BODY_SCHEMA
        or declaration.get("operation") != "arith.maximumf"
    ):
        raise InvalidLinalgPattern("source_body requires a closed f32 maximum v1 declaration")
    return dict(declaration)


def serialized_f32_maximum_pattern(pattern: StaticF32MaximumPattern) -> dict:
    """Persist only JSON-typed fields from the independently recognized source."""
    return json.loads(json.dumps(asdict(pattern), allow_nan=False))


@dataclass(frozen=True)
class StaticF32MaximumPattern:
    axis: int
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    ordered_types: tuple[str, ...]
    index_bits_premise: int


@dataclass(frozen=True)
class StaticF32MaximumSource:
    raw_sha256: str
    normalized_sha256: str
    ordinals: tuple[tuple[int, StaticF32MaximumPattern], ...]


def _closed(op, cls, properties: set[str]) -> None:
    from xdsl.traits import Pure

    if (
        type(op) is not cls
        or not type(op).has_trait(Pure)
        or op.regions
        or op.successors
        or set(op.properties) != properties
        or any(not name.startswith("prov.") for name in op.attributes)
    ):
        raise InvalidLinalgPattern("maximum source has an unproved owner or scalar effect")


def _shape(value) -> tuple[int, ...]:
    from xdsl.dialects.builtin import NoneAttr, TensorType

    typ = value.type
    if not isinstance(typ, TensorType) or str(typ.element_type) != "f32" or not isinstance(typ.encoding, NoneAttr):
        raise InvalidLinalgPattern("maximum source requires a default-encoded f32 tensor")
    shape = tuple(typ.get_shape())
    if any(type(extent) is not int or extent <= 0 for extent in shape):
        raise InvalidLinalgPattern("maximum source requires positive static extents")
    return shape


def _negative_infinity_init(value) -> None:
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import FloatAttr

    splat = value.owner
    _closed(splat, tensor.SplatOp, set())
    if tuple(splat.operands) != (splat.input,) or tuple(splat.results) != (value,) or splat.dynamicSizes:
        raise InvalidLinalgPattern("maximum source init is not a static splat")
    constant = splat.input.owner
    _closed(constant, arith.ConstantOp, {"value"})
    literal = constant.properties["value"]
    if (
        constant.operands
        or tuple(constant.results) != (splat.input,)
        or str(splat.input.type) != "f32"
        or not isinstance(literal, FloatAttr)
        or str(literal.type) != "f32"
        or not isinf(literal.value.data)
        or literal.value.data >= 0
    ):
        raise InvalidLinalgPattern("maximum source init is not exact f32 negative infinity")
    try:
        constant.verify()
        splat.verify()
    except Exception as exc:  # noqa: BLE001 - xDSL owner verification is fail-closed
        raise InvalidLinalgPattern(f"maximum source init failed verification: {exc}") from exc


def recognize_static_f32_maximum(op, *, index_bits: int) -> StaticF32MaximumPattern:
    """Prove the prepared one-axis reduction geometry and exact maximumf body.

    ``index_bits`` is an explicit caller premise, not a selected build proof.
    Element-count bounds do not prove byte-offset lowering or native ABI safety.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import DenseArrayBase
    from xdsl.dialects.linalg.ops import ReduceOp, YieldOp

    if type(index_bits) is not int or not 2 <= index_bits <= 128:
        raise InvalidLinalgPattern("maximum source requires an explicit valid index-width premise")
    if (
        type(op) is not ReduceOp
        or set(op.properties) != {"dimensions"}
        or any(not name.startswith("prov.") for name in op.attributes)
        or op.successors
        or len(op.operands) != 2
        or tuple(op.operands) != (op.input, op.init)
        or len(op.results) != 1
        or len(op.regions) != 1
        or len(op.regions[0].blocks) != 1
    ):
        raise InvalidLinalgPattern("maximum source is not a closed registered linalg.reduce")
    input_shape = _shape(op.input)
    output_shape = _shape(op.init)
    if not input_shape or op.results[0].type != op.init.type or _shape(op.results[0]) != output_shape:
        raise InvalidLinalgPattern("maximum source result differs from its initialized tensor")
    maximum_index = (1 << (index_bits - 1)) - 1
    if any(extent > maximum_index for extent in (*input_shape, *output_shape)) or any(
        prod(shape) > maximum_index for shape in (input_shape, output_shape)
    ):
        raise InvalidLinalgPattern("maximum source extent or element span exceeds the index-width premise")
    dimensions = op.properties["dimensions"]
    if not isinstance(dimensions, DenseArrayBase) or str(dimensions.elt_type) != "i64":
        raise InvalidLinalgPattern("maximum source axis is not typed i64")
    axes = tuple(dimensions.iter_values())
    if len(axes) != 1 or type(axes[0]) is not int or not 0 <= axes[0] < len(input_shape):
        raise InvalidLinalgPattern("maximum source requires exactly one valid reduction axis")
    axis = axes[0]
    if output_shape != tuple(extent for i, extent in enumerate(input_shape) if i != axis):
        raise InvalidLinalgPattern("maximum source output geometry does not project its axis")
    _negative_infinity_init(op.init)
    block = op.regions[0].block
    args, body = tuple(block.args), tuple(block.ops)
    if tuple(str(arg.type) for arg in args) != ("f32", "f32") or len(body) != 2:
        raise InvalidLinalgPattern("maximum source body is not a two-argument scalar reduction")
    maximum, yld = body
    _closed(maximum, arith.MaximumfOp, {"fastmath"} if "fastmath" in maximum.properties else set())
    flags = maximum.properties.get("fastmath")
    if flags is not None and (not isinstance(flags, arith.FastMathFlagsAttr) or flags.data):
        raise InvalidLinalgPattern("maximum source body carries unproved fast-math flags")
    if (
        tuple(maximum.operands) != args
        or len(maximum.results) != 1
        or str(maximum.results[0].type) != "f32"
        or type(yld) is not YieldOp
        or tuple(yld.operands) != (maximum.results[0],)
        or yld.results
        or yld.regions
        or yld.successors
        or yld.properties
        or any(not name.startswith("prov.") for name in yld.attributes)
    ):
        raise InvalidLinalgPattern("maximum source does not yield its exact maximumf result")
    try:
        op.verify()
    except Exception as exc:  # noqa: BLE001 - xDSL verification is a fail-closed source boundary
        raise InvalidLinalgPattern(f"maximum source reduction failed verification: {exc}") from exc
    return StaticF32MaximumPattern(axis, input_shape, output_shape, ("f32", "f32", "f32"), index_bits)


def screen_static_f32_maximum_source(
    path: Path, ordinals: tuple[int, ...], *, index_bits: int
) -> StaticF32MaximumSource:
    raw, normalized, evidence = _screen_static_linalg_source(
        path, ordinals, lambda op: recognize_static_f32_maximum(op, index_bits=index_bits)
    )
    return StaticF32MaximumSource(raw, normalized, evidence)
