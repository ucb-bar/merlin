"""Preserve exact typed bytes of compiler-owned tensor literals."""

from math import prod

from .linalg_recognition import LinalgBridgeError, _type
from .reference import TensorValue


def read_tensor_bytes(type_, data):
    encodings = {
        "i1": ("i1-raw", 1, False),
        "i8": ("i8", 1, True),
        "i32": ("i32-le", 4, True),
        "i64": ("i64-le", 8, True),
        "bf16": ("bf16-le-raw", 2, False),
        "f32": ("f32-le-raw", 4, False),
        "f64": ("f64-le-raw", 8, False),
        "fp8_e4m3": ("fp8_e4m3-raw", 1, False),
    }
    if type_.dtype not in encodings:
        raise LinalgBridgeError(f"tensor constant has no byte encoding for {type_.dtype}")
    encoding, width, signed = encodings[type_.dtype]
    if not isinstance(data, bytes) or len(data) != prod(type_.shape) * width:
        raise LinalgBridgeError("constant byte length differs from its parsed tensor type and shape")
    elements = tuple(int.from_bytes(data[i : i + width], "little", signed=signed) for i in range(0, len(data), width))
    return encoding, TensorValue(type_, elements)


def read_dense_literal(op):
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr

    literal = op.properties.get("value")
    if (
        op.name != "arith.constant"
        or len(op.results) != 1
        or not isinstance(literal, DenseIntOrFPElementsAttr)
        or literal.type != op.results[0].type
        or set(op.properties) != {"value"}
        or any(not name.startswith("prov.") for name in op.attributes)
    ):
        raise LinalgBridgeError("tensor literal needs a closed, typed dense arith.constant")
    op.verify()
    type_ = _type(literal.type)
    # i1 has one logical bit; our byte interchange uses canonical 0/1 rather
    # than xDSL's signed storage spelling for a true bit.
    data = bytes(int(value) & 1 for value in literal.get_values()) if type_.dtype == "i1" else literal.data.data
    encoding, value = read_tensor_bytes(type_, data)
    return type_, encoding, data, value
