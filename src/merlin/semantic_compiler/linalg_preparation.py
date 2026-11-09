"""Translate the preparation source family, preserving numerical boundaries."""

from __future__ import annotations

from xdsl.dialects.builtin import IntegerAttr

from merlin.frontends.linalg_ordered_fp8 import is_positive_bf16_zero

from .linalg_recognition import (
    LinalgBridgeError,
    _i64_to_f32_bits,
    _scalar_f32_bits,
    _scalar_i64,
    _scalar_integer,
    _type,
)
from .linalg_state import TranslationState
from .model import ConstantBinding, SemanticNode


def translate(state: TranslationState, op, ordinal):
    nodes = state.nodes
    env = state.env
    scalars = state.scalars
    i64_scalars = state.i64_scalars
    f32_scalars = state.f32_scalars
    f64_scalars = state.f64_scalars
    boolean_scalars = state.boolean_scalars
    bf16_zeros = state.bf16_zeros
    constants = state.constants
    constant_bindings = state.constant_bindings
    append_splat = state.append_splat
    scan_prelude = state.scan_prelude
    ordered_scans = state.ordered_scans
    indexed_prelude = state.indexed_prelude
    indexed_updates = state.indexed_updates
    if op in scan_prelude:
        return True

    if op in ordered_scans:
        scan = ordered_scans[op]
        scan_input = env.get(scan.source)
        if scan_input is None:
            raise LinalgBridgeError("ordered floating scan reads an undefined or uninitialized input")
        node_id = f"op{ordinal}r0"
        nodes.append(
            SemanticNode(
                node_id,
                "ordered_f32_scan_f64",
                (scan_input,),
                _type(op.results[0].type),
                attrs=(
                    ("stride", scan.stride),
                    ("extent", scan.extent),
                    ("numeric_semantics", "ordered-f32-prefix-f64-rne-v1"),
                ),
            )
        )
        env[op.results[0]] = node_id
        return True

    if op in indexed_prelude:
        return True

    if op in indexed_updates:
        indexed = indexed_updates[op]
        values = (indexed.source, indexed.destination, indexed.positions)
        if any(value not in env or env[value] is None for value in values):
            raise LinalgBridgeError("indexed update reads an undefined or uninitialized source")
        node_id = f"op{ordinal}r0"
        nodes.append(
            SemanticNode(
                node_id,
                "indexed_insert_slice",
                tuple(env[value] for value in values),
                _type(op.results[0].type),
                (("axis", indexed.axis), ("index_domain", "nonnegative-below-destination-axis")),
            )
        )
        env[op.results[0]] = node_id
        return True

    if op.name == "arith.constant":
        if len(op.results) != 1:
            raise LinalgBridgeError("scalar constant has an unsupported result count")
        from xdsl.dialects.builtin import TensorType as XdslTensorType

        if isinstance(op.results[0].type, XdslTensorType):
            from .linalg_constants import read_dense_literal

            type_, encoding, data, value = read_dense_literal(op)
            node_id = f"op{ordinal}r0"
            nodes.append(SemanticNode(node_id, "constant", (), type_, effect="constant"))
            constants[node_id] = value
            constant_bindings.append(ConstantBinding(node_id, encoding, data.hex()))
            env[op.results[0]] = node_id
            return True
        if is_positive_bf16_zero(op):
            bf16_zeros.add(op.results[0])
        elif str(op.results[0].type) == "f32":
            f32_scalars[op.results[0]] = _scalar_f32_bits(op)
        elif str(op.results[0].type) == "f64":
            from math import isnan
            from struct import pack

            from xdsl.dialects.builtin import FloatAttr

            literal = op.properties.get("value")
            if (
                not isinstance(literal, FloatAttr)
                or str(literal.type) != "f64"
                or set(op.properties) != {"value"}
                or any(not key.startswith("prov.") for key in op.attributes)
                or isnan(literal.value.data)
            ):
                raise LinalgBridgeError("f64 scalar needs an exact non-NaN literal")
            f64_scalars[op.results[0]] = int.from_bytes(pack("<d", literal.value.data), "little")
        elif str(op.results[0].type) == "i64":
            i64_scalars[op.results[0]] = _scalar_i64(op)
        elif str(op.results[0].type) == "i1":
            literal = op.properties.get("value")
            if (
                not isinstance(literal, IntegerAttr)
                or str(literal.type) != "i1"
                or literal.value.data not in (-1, 0, 1)
                or set(op.properties) != {"value"}
                or any(not key.startswith("prov.") for key in op.attributes)
            ):
                raise LinalgBridgeError("Boolean scalar needs an exact i1 literal")
            boolean_scalars[op.results[0]] = literal.value.data & 1
        else:
            scalars[op.results[0]] = _scalar_integer(op)
        return True

    if op.name == "arith.sitofp":
        if (
            len(op.operands) != 1
            or len(op.results) != 1
            or op.properties
            or any(not key.startswith("prov.") for key in op.attributes)
            or str(op.operands[0].type) != "i64"
            or str(op.results[0].type) != "f32"
            or op.operands[0] not in i64_scalars
        ):
            raise LinalgBridgeError("scalar signed-i64-to-f32 cast needs a declared constant")
        try:
            op.verify()
        except Exception as exc:  # noqa: BLE001 - xDSL owns source verification
            raise LinalgBridgeError(f"scalar signed-i64-to-f32 cast failed verification: {exc}") from exc
        f32_scalars[op.results[0]] = _i64_to_f32_bits(i64_scalars[op.operands[0]])
        return True

    if op.name == "tensor.empty":
        if len(op.results) != 1:
            raise LinalgBridgeError("tensor.empty has an unsupported result count")
        _type(op.results[0].type)
        env[op.results[0]] = None
        return True

    if op.name == "tensor.splat":
        if (
            len(op.operands) != 1
            or len(op.results) != 1
            or op.properties
            or any(not key.startswith("prov.") for key in op.attributes)
        ):
            raise LinalgBridgeError("tensor.splat has an unsupported signature or attributes")
        scalar = op.operands[0]
        type_ = _type(op.results[0].type)
        node_id = f"op{ordinal}r0"
        if type_.dtype == "f32" and scalar in f32_scalars:
            bits = f32_scalars[scalar]
            value, encoding, data = bits, "f32-le-raw", bits.to_bytes(4, "little")
        elif type_.dtype == "f64" and scalar in f64_scalars:
            bits = f64_scalars[scalar]
            value, encoding, data = bits, "f64-le-raw", bits.to_bytes(8, "little")
        elif type_.dtype == "i1" and scalar in boolean_scalars:
            value = boolean_scalars[scalar]
            encoding, data = "i1-raw", bytes((value,))
        elif type_.dtype == "i32" and scalar in scalars:
            value = scalars[scalar]
            encoding, data = "i32-le", value.to_bytes(4, "little", signed=True)
        elif type_.dtype == "i64" and scalar in i64_scalars:
            value = i64_scalars[scalar]
            encoding, data = "i64-le", value.to_bytes(8, "little", signed=True)
        elif type_.dtype == "bf16" and scalar in bf16_zeros:
            value, encoding, data = 0, "bf16-le-raw", b"\x00\x00"
        else:
            raise LinalgBridgeError("tensor.splat requires an admitted constant with matching element type")
        env[op.results[0]] = append_splat(node_id, type_, value, encoding, data)
        return True

    return False
