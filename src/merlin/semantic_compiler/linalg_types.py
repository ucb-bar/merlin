"""Recognize typed Linalg bodies and affine index maps without selecting hardware."""

from __future__ import annotations

from struct import pack, unpack

from .model import IndexMap, TensorType, pointwise_identity_maps


class LinalgBridgeError(ValueError):
    pass


def _type(attribute) -> TensorType:
    from xdsl.dialects.builtin import NoneAttr
    from xdsl.dialects.builtin import TensorType as XdslTensorType

    if not isinstance(attribute, XdslTensorType):
        raise LinalgBridgeError("only ranked tensor values are admitted")
    if not isinstance(attribute.encoding, NoneAttr):
        raise LinalgBridgeError("source tensor encoding needs an explicit logical-layout translation")
    shape = tuple(int(dimension) for dimension in attribute.get_shape())
    dtype = str(attribute.element_type)
    if dtype == "i1":
        policy = "boolean"
    elif dtype == "i8":
        policy = "signed-i8"
    elif dtype == "i32":
        policy = "i32-wrap-k-ascending"
    elif dtype == "i64":
        policy = "signed-i64"
    elif dtype == "bf16":
        policy = "raw-bit-pattern"
    elif dtype in {"f32", "f64"}:
        # Source arithmetic needs a separate declared precision policy.
        # Its boundary values and bit-preserving moves still have exact types.
        policy = "raw-bit-pattern"
    elif dtype == "f8E4M3FN":
        dtype = "fp8_e4m3"
        policy = "raw-bit-pattern"
    else:
        raise LinalgBridgeError(f"no admitted numerical policy for {dtype}")
    try:
        return TensorType(shape, dtype, policy)
    except ValueError as exc:
        raise LinalgBridgeError(f"unadmitted tensor type {attribute}: {exc}") from exc


def _scalar_integer(op) -> int:
    attr = op.properties.get("value", op.attributes.get("value"))
    value = getattr(getattr(attr, "value", None), "data", None)
    if type(value) is not int or str(op.results[0].type) != "i32":
        raise LinalgBridgeError("fill scalar must be an explicit i32 integer constant")
    if not -(1 << 31) <= value < (1 << 31):
        raise LinalgBridgeError("fill constant is outside signed i32")
    return value


def _scalar_i64(op) -> int:
    attr = op.properties.get("value", op.attributes.get("value"))
    value = getattr(getattr(attr, "value", None), "data", None)
    if type(value) is not int or str(op.results[0].type) != "i64":
        raise LinalgBridgeError("fill scalar must be an explicit i64 integer constant")
    if not -(1 << 63) <= value < (1 << 63):
        raise LinalgBridgeError("fill constant is outside signed i64")
    return value


def _i64_to_f32_bits(value: int) -> int:
    """Round a signed i64 exactly to binary32, ties to even."""
    if not -(1 << 63) <= value < (1 << 63):
        raise LinalgBridgeError("signed integer cast is outside i64")
    magnitude = abs(value)
    if magnitude == 0:
        return 0
    exponent = magnitude.bit_length() - 1
    shift = max(0, exponent - 23)
    significand = magnitude >> shift
    if shift:
        remainder = magnitude - (significand << shift)
        halfway = 1 << (shift - 1)
        if remainder > halfway or remainder == halfway and significand & 1:
            significand += 1
    else:
        significand <<= 23 - exponent
    if significand == 1 << 24:
        significand >>= 1
        exponent += 1
    return (int(value < 0) << 31) | ((exponent + 127) << 23) | (significand & ((1 << 23) - 1))


def _scalar_f32_bits(op) -> int:
    from math import isnan

    from xdsl.dialects.builtin import FloatAttr

    attr = op.properties.get("value", op.attributes.get("value"))
    if str(op.results[0].type) != "f32" or not isinstance(attr, FloatAttr):
        raise LinalgBridgeError("fill scalar must be an explicit f32 constant")
    value = float(attr.value.data)
    if isnan(value):
        raise LinalgBridgeError("f32 NaN scalar payload cannot be preserved by this bridge")
    try:
        return unpack("<I", pack("<f", value))[0]
    except OverflowError as exc:
        raise LinalgBridgeError("f32 fill scalar is outside its storage range") from exc
