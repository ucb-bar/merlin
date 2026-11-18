"""Explicit static accumulator readout semantics for canonical Interface IR.

``ordered_f32`` converts i32 to binary32, applies each binary32 scale in order,
rounds to nearest-even, clamps, then narrows to signed i8. ``integer_saturate``
clamps i32 directly and narrows to signed i8. Scale words are original binary32
bit patterns, never a reassociated product or backend-selected approximation.
Version 2 retains original scalar/channel scales and f32 bias tensors as actual
immutable SSA constants, with explicit channel mapping and arithmetic order.
Runtime qparams remain outside this static contract. A record describes source
arithmetic; it does not prove a target implementation or authorize a tolerance.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, IntegerAttr, StringAttr, i64

SCHEMA = "merlin.readout.v1"


def scale_bits(value):
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def readout_values(attribute):
    """Validate the closed schema and return kind, ordered scales and ReLU."""
    fields = {"schema", "kind", "scale_bits", "rounding", "minimum", "maximum"}
    if not isinstance(attribute, DictionaryAttr) or set(attribute.data) != fields:
        raise ValueError("Interface readout has an unknown numerical schema")
    values = attribute.data
    strings = {}
    for key in ("schema", "kind", "rounding"):
        if not isinstance(values[key], StringAttr):
            raise ValueError("Interface readout requires named numerical choices")
        strings[key] = values[key].data
    if strings["schema"] != SCHEMA:
        raise ValueError("unsupported Interface readout schema")
    if not isinstance(values["scale_bits"], ArrayAttr) or any(
        not isinstance(word, IntegerAttr) or not 0 <= word.value.data < 1 << 32 for word in values["scale_bits"]
    ):
        raise ValueError("Interface scales require explicit binary32 words")
    scales = tuple(struct.unpack("<f", struct.pack("<I", word.value.data))[0] for word in values["scale_bits"])
    if any(not math.isfinite(scale) or scale <= 0 for scale in scales):
        raise ValueError("static readout scales must be finite positive binary32")
    if any(not isinstance(values[key], IntegerAttr) for key in ("minimum", "maximum")):
        raise ValueError("Interface readout requires exact integer clamp bounds")
    minimum, maximum = (values[key].value.data for key in ("minimum", "maximum"))
    if maximum != 127 or minimum not in (-128, 0):
        raise ValueError("static Interface readout implements signed-i8 saturation only")
    kind = strings["kind"]
    if kind == "ordered_f32":
        if not scales or strings["rounding"] != "nearest_even":
            raise ValueError("ordered binary32 readout needs scales and nearest-even rounding")
    elif kind == "integer_saturate":
        if scales or minimum != -128 or strings["rounding"] != "none":
            raise ValueError("integer saturation has no scale or rounding stage")
    else:
        raise ValueError("unsupported static Interface readout arithmetic")
    return kind, scales, minimum == 0


def readout_contract(scales=(), *, relu=False, integer=False):
    """Represent the original ordered arithmetic using exact scalar parameters."""
    result = DictionaryAttr(
        {
            "schema": StringAttr(SCHEMA),
            "kind": StringAttr("integer_saturate" if integer else "ordered_f32"),
            "scale_bits": ArrayAttr([IntegerAttr(scale_bits(value), i64) for value in scales]),
            "rounding": StringAttr("none" if integer else "nearest_even"),
            "minimum": IntegerAttr(0 if relu else -128, i64),
            "maximum": IntegerAttr(127, i64),
        }
    )
    readout_values(result)
    return result


PARAMETER_SCHEMA = "merlin.readout.v2"


@dataclass(frozen=True)
class ReadoutStep:
    kind: str
    values: tuple[float, ...]
    parameter: int | None
    shape: tuple[int, ...]


@dataclass(frozen=True)
class ReadoutParameters:
    kind: str
    steps: tuple[ReadoutStep, ...]
    axis: int | None
    channels: int | None
    relu: bool


def constant_parameter(value):
    """Read an actual immutable f32 scalar/channel constant without replacing it.

    The caller retains the original SSA operation and exact attribute bytes.
    Floating values returned here are for mathematical admission only; they are
    never used to reconstruct the canonical constant.
    """
    from xdsl.dialects.arith import ConstantOp
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, Float32Type, FloatAttr, NoneAttr, TensorType

    owner = value.owner
    if (
        not isinstance(owner, ConstantOp)
        or tuple(owner.results) != (value,)
        or owner.operands
        or owner.regions
        or owner.successors
    ):
        raise ValueError("readout parameters require immutable arith.constant SSA values")
    if set(owner.properties) != {"value"} or any(not name.startswith("prov.") for name in owner.attributes):
        raise ValueError("readout parameter has unhandled attributes or properties")
    literal = owner.properties["value"]
    if isinstance(value.type, Float32Type) and isinstance(literal, FloatAttr) and literal.type == value.type:
        values, shape = (float(literal.value.data),), ()
    elif (
        isinstance(value.type, TensorType)
        and isinstance(value.type.element_type, Float32Type)
        and isinstance(value.type.encoding, NoneAttr)
        and len(value.type.get_shape()) == 1
        and value.type.get_shape()[0] > 0
        and isinstance(literal, DenseIntOrFPElementsAttr)
        and literal.type == value.type
    ):
        shape = tuple(value.type.get_shape())
        values = tuple(float(element) for element in literal.get_values())
        if len(values) == 1 and shape[0] != 1:
            values *= shape[0]
        if len(values) != shape[0]:
            raise ValueError("readout constant element count differs from its tensor shape")
    else:
        raise ValueError("readout parameter must be f32 or an ordinary one-dimensional f32 tensor")
    if any(not math.isfinite(element) for element in values):
        raise ValueError("readout parameters must be finite binary32 values")
    return values, shape


def readout_parameters(attribute, parameters=(), *, output_shape=None):
    """Normalize v1 scalars or v2 ordered static SSA parameters.

    V2 means f32(acc), then each listed f32 multiply/add in order, followed by
    nearest-even rounding, the explicit clamp and signed-i8 narrowing. A scale
    scalar broadcasts; a channel vector addresses exactly ``axis``. Bias is an
    original f32 addition, never a synthesized integer accumulator preload.
    """
    if not isinstance(attribute, DictionaryAttr):
        raise ValueError("Interface readout requires a typed numerical dictionary")
    schema = attribute.data.get("schema")
    if isinstance(schema, StringAttr) and schema.data == SCHEMA:
        if parameters:
            raise ValueError("readout.v1 has no SSA parameter operands")
        kind, scales, relu = readout_values(attribute)
        return ReadoutParameters(
            kind, tuple(ReadoutStep("scale", (value,), None, ()) for value in scales), None, None, relu
        )
    fields = {"schema", "kind", "steps", "axis", "channels", "rounding", "minimum", "maximum"}
    values = attribute.data
    if not isinstance(schema, StringAttr) or schema.data != PARAMETER_SCHEMA or set(values) != fields:
        raise ValueError("Interface readout has an unknown parameter schema")
    if (
        not isinstance(values["kind"], StringAttr)
        or values["kind"].data != "ordered_f32"
        or not isinstance(values["rounding"], StringAttr)
        or values["rounding"].data != "nearest_even"
    ):
        raise ValueError("parameterized readout requires ordered binary32 arithmetic and nearest-even rounding")
    if any(not isinstance(values[name], IntegerAttr) for name in ("axis", "channels", "minimum", "maximum")):
        raise ValueError("parameterized readout requires explicit channel axis and clamp")
    axis, channels, minimum, maximum = (values[name].value.data for name in ("axis", "channels", "minimum", "maximum"))
    if axis < 0 or channels <= 0 or minimum not in (-128, 0) or maximum != 127:
        raise ValueError("parameterized readout has an invalid axis, channel count or signed-i8 clamp")
    if output_shape is not None and (axis != len(output_shape) - 1 or output_shape[axis] != channels):
        raise ValueError("readout channel axis/count differs from its actual output tensor")
    if not isinstance(values["steps"], ArrayAttr) or not len(values["steps"]):
        raise ValueError("parameterized readout requires an ordered nonempty scalar computation")
    decoded = tuple(constant_parameter(value) for value in parameters)
    steps, used, biases = [], set(), 0
    for step in values["steps"]:
        if not isinstance(step, DictionaryAttr) or set(step.data) != {"kind", "parameter"}:
            raise ValueError("readout step requires exact kind and SSA parameter index")
        kind, index = step.data["kind"], step.data["parameter"]
        if (
            not isinstance(kind, StringAttr)
            or kind.data not in ("scale", "bias")
            or not isinstance(index, IntegerAttr)
            or not 0 <= index.value.data < len(decoded)
        ):
            raise ValueError("readout step does not reference a declared scalar/channel constant")
        number = index.value.data
        elements, shape = decoded[number]
        if shape not in ((), (channels,)):
            raise ValueError("readout parameter is not scalar or the exact output-channel vector")
        if kind.data == "scale" and any(value <= 0 for value in elements):
            raise ValueError("readout scales must be positive finite binary32")
        biases += kind.data == "bias"
        if biases > 1 or kind.data == "bias" and shape != (channels,):
            raise ValueError("static readout permits one explicit f32 channel-bias tensor")
        steps.append(ReadoutStep(kind.data, elements, number, shape))
        used.add(number)
    if used != set(range(len(parameters))):
        raise ValueError("readout has an unused or hidden SSA parameter")
    return ReadoutParameters("ordered_f32", tuple(steps), axis, channels, minimum == 0)


def parameter_readout_contract(steps, parameters, *, output_shape, relu=False):
    """Describe actual source SSA arithmetic, preserving parameter identity/order."""
    result = DictionaryAttr(
        {
            "schema": StringAttr(PARAMETER_SCHEMA),
            "kind": StringAttr("ordered_f32"),
            "steps": ArrayAttr(
                [
                    DictionaryAttr({"kind": StringAttr(kind), "parameter": IntegerAttr(index, i64)})
                    for kind, index in steps
                ]
            ),
            "axis": IntegerAttr(len(output_shape) - 1, i64),
            "channels": IntegerAttr(output_shape[-1], i64),
            "rounding": StringAttr("nearest_even"),
            "minimum": IntegerAttr(0 if relu else -128, i64),
            "maximum": IntegerAttr(127, i64),
        }
    )
    readout_parameters(result, parameters, output_shape=output_shape)
    return result
