"""Independent evaluator for admitted integer, strict f32 arithmetic and opaque-bit operations.

This is a verification tool. Native selection never calls it, and unknown
numerical policies or operations fail instead of acquiring guessed semantics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import product as index_product
from math import isnan, prod
from struct import pack, unpack

from .model import IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps

_I32_MIN = -(1 << 31)
_I32_MAX = (1 << 31) - 1
_BF16_RELU_ATTRS = {"input_domain": "finite-normal-or-zero-bf16", "numeric_semantics": "bf16-sign-clamp-v1"}


def _i32(value: int) -> int:
    if type(value) is not int or not _I32_MIN <= value <= _I32_MAX:
        raise ValueError("exact-i32 value is outside its admitted domain")
    return value


def _wrap_i32(value: int) -> int:
    return (value + (1 << 31)) % (1 << 32) - (1 << 31)


def _matmul_maps() -> tuple[IndexMap, ...]:
    return (
        IndexMap(3, ((1, 0, 0), (0, 0, 1)), (0, 0)),
        IndexMap(3, ((0, 0, 1), (0, 1, 0)), (0, 0)),
        IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
        IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
    )


def _f32_add_rne(left: int, right: int) -> int:
    """Add two binary32 bit patterns exactly, then round once to nearest even."""
    signs = (left >> 31, right >> 31)
    fields = ((left >> 23 & 255, left & 0x7FFFFF), (right >> 23 & 255, right & 0x7FFFFF))
    if any(exponent == 255 and fraction for exponent, fraction in fields):
        raise ValueError("f32 addition NaN payload is not qualified by this reference")
    if any(exponent == 255 for exponent, _ in fields):
        if all(exponent == 255 for exponent, _ in fields) and signs[0] != signs[1]:
            raise ValueError("f32 addition NaN result is not qualified by this reference")
        return left if fields[0][0] == 255 else right

    def component(bits: int) -> tuple[int, int]:
        exponent = bits >> 23 & 255
        fraction = bits & 0x7FFFFF
        return ((1 << 23) | fraction, exponent - 150) if exponent else (fraction, -149)

    left_mantissa, left_exponent = component(left)
    right_mantissa, right_exponent = component(right)
    exponent = min(left_exponent, right_exponent)
    exact = (-1 if signs[0] else 1) * (left_mantissa << (left_exponent - exponent)) + (-1 if signs[1] else 1) * (
        right_mantissa << (right_exponent - exponent)
    )
    if exact == 0:
        return 0x80000000 if left_mantissa == right_mantissa == 0 and signs == (1, 1) else 0
    sign = 0x80000000 if exact < 0 else 0
    magnitude = abs(exact)

    def round_shift(value: int, shift: int) -> int:
        if shift <= 0:
            return value << -shift
        quotient, remainder = divmod(value, 1 << shift)
        halfway = 1 << (shift - 1)
        return quotient + (remainder > halfway or remainder == halfway and bool(quotient & 1))

    top_exponent = magnitude.bit_length() - 1 + exponent
    if top_exponent >= -126:
        significand = round_shift(magnitude, top_exponent - 23 - exponent)
        if significand == 1 << 24:
            significand >>= 1
            top_exponent += 1
        if top_exponent > 127:
            return sign | 0x7F800000
        return sign | ((top_exponent + 127) << 23) | (significand - (1 << 23))
    fraction = round_shift(magnitude, -149 - exponent)
    return sign | (0x00800000 if fraction == 1 << 23 else fraction)


def _f32_div_rne(left: int, right: int) -> int:
    """Divide binary32 values as an exact rational, then round once to nearest even."""
    sign = (left ^ right) & 0x80000000
    left_exponent, right_exponent = (left >> 23) & 255, (right >> 23) & 255
    left_fraction, right_fraction = left & 0x7FFFFF, right & 0x7FFFFF
    if (left_exponent == 255 and left_fraction) or (right_exponent == 255 and right_fraction):
        raise ValueError("f32 division NaN payload is not qualified by this reference")
    left_zero = left_exponent == 0 and left_fraction == 0
    right_zero = right_exponent == 0 and right_fraction == 0
    if (left_zero and right_zero) or (left_exponent == right_exponent == 255):
        raise ValueError("f32 division NaN result is not qualified by this reference")
    if right_zero or left_exponent == 255:
        return sign | 0x7F800000
    if left_zero or right_exponent == 255:
        return sign

    def component(exponent: int, fraction: int) -> tuple[int, int]:
        return ((1 << 23) | fraction, exponent - 150) if exponent else (fraction, -149)

    left_mantissa, left_power = component(left_exponent, left_fraction)
    right_mantissa, right_power = component(right_exponent, right_fraction)
    power = left_power - right_power
    numerator = left_mantissa << max(power, 0)
    denominator = right_mantissa << max(-power, 0)

    def round_ratio(numer: int, denom: int) -> int:
        quotient, remainder = divmod(numer, denom)
        return quotient + (2 * remainder > denom or 2 * remainder == denom and bool(quotient & 1))

    exponent = numerator.bit_length() - denominator.bit_length()
    if numerator < denominator << exponent if exponent >= 0 else numerator << -exponent < denominator:
        exponent -= 1
    if exponent >= -126:
        shift = 23 - exponent
        significand = round_ratio(
            numerator << max(shift, 0),
            denominator << max(-shift, 0),
        )
        if significand == 1 << 24:
            significand >>= 1
            exponent += 1
        if exponent > 127:
            return sign | 0x7F800000
        return sign | ((exponent + 127) << 23) | (significand - (1 << 23))
    fraction = round_ratio(numerator << 149, denominator)
    return sign | (0x00800000 if fraction == 1 << 23 else fraction)


def _f32_reduce_maps(rank: int, axis: int) -> tuple[IndexMap, ...]:
    identity = IndexMap(
        rank, tuple(tuple(int(dim == loop) for loop in range(rank)) for dim in range(rank)), (0,) * rank
    )
    projection = IndexMap(
        rank,
        tuple(tuple(int(dim == loop) for loop in range(rank)) for dim in range(rank) if dim != axis),
        (0,) * (rank - 1),
    )
    return identity, projection, projection


def _checked_f32_pointwise_maps(node: SemanticNode, operands: tuple[TensorValue, ...]) -> tuple[IndexMap, ...]:
    """Recheck a static singleton projection without trusting the source recognizer."""
    rank = len(node.type.shape)
    if len(node.index_maps) != len(operands) + 1:
        raise ValueError("f32 pointwise needs an index map for every tensor")
    output_map = node.index_maps[-1]
    if output_map != pointwise_identity_maps(rank, 1)[0]:
        raise ValueError("f32 pointwise output map is not identity")
    for operand, mapping in zip(operands, node.index_maps[:-1], strict=True):
        if mapping.loop_rank != rank or len(mapping.coefficients) != len(operand.type.shape):
            raise ValueError("f32 pointwise input map has the wrong rank")
        last_dimension = -1
        for extent, row, offset in zip(
            operand.type.shape,
            mapping.coefficients,
            mapping.offsets,
            strict=True,
        ):
            if not any(row) and offset == 0 and extent == 1:
                continue
            positions = [dimension for dimension, coefficient in enumerate(row) if coefficient == 1]
            if (
                offset != 0
                or len(positions) != 1
                or row != tuple(int(dimension == positions[0]) for dimension in range(rank))
                or positions[0] <= last_dimension
                or extent != node.type.shape[positions[0]]
            ):
                raise ValueError("f32 pointwise input map is not a singleton projection")
            last_dimension = positions[0]
    return node.index_maps[:-1]


def _flat_index(indices: tuple[int, ...], shape: tuple[int, ...]) -> int:
    flat = 0
    for index, extent in zip(indices, shape, strict=True):
        flat = flat * extent + index
    return flat


@dataclass(frozen=True)
class TensorValue:
    type: TensorType
    elements: tuple[int, ...]

    def __post_init__(self) -> None:
        admitted = {
            ("i1", "boolean"),
            ("i32", "exact-i32"),
            ("i32", "i32-wrap-k-ascending"),
            ("i64", "signed-i64"),
            ("i8", "signed-i8"),
            ("i8", "raw-byte-copy"),
        }
        raw_float = self.type.dtype in {"fp8_e4m3", "bf16", "f32", "f64"}
        if not raw_float and (self.type.dtype, self.type.numerical_policy) not in admitted:
            raise ValueError("reference evaluator has no admitted numerical policy")
        if len(self.elements) != prod(self.type.shape):
            raise ValueError("reference tensor size differs from its type")
        for element in self.elements:
            if self.type.dtype == "i1":
                if type(element) is not int or element not in (0, 1):
                    raise ValueError("boolean tensor elements must be zero or one")
            elif raw_float or self.type.numerical_policy == "raw-byte-copy":
                maximum = {"i8": 255, "fp8_e4m3": 255, "bf16": 65535, "f32": (1 << 32) - 1, "f64": (1 << 64) - 1}[
                    self.type.dtype
                ]
                if type(element) is not int or not 0 <= element <= maximum:
                    raise ValueError("raw-bit value is outside its admitted domain")
            elif self.type.dtype == "i8":
                if type(element) is not int or not -128 <= element <= 127:
                    raise ValueError("signed-i8 value is outside its admitted domain")
            elif self.type.dtype == "i64":
                if type(element) is not int or not -(1 << 63) <= element < (1 << 63):
                    raise ValueError("signed-i64 value is outside its admitted domain")
            else:
                _i32(element)


ReferencePrimitive = Callable[[SemanticNode, tuple[TensorValue, ...]], TensorValue]


def evaluate_graph(
    request: KernelRequest,
    inputs: dict[str, TensorValue],
    *,
    constants: dict[str, TensorValue] | None = None,
    primitives: Mapping[tuple[str, str], ReferencePrimitive] | None = None,
    source_primitives: Mapping[str, ReferencePrimitive] | None = None,
) -> tuple[TensorValue, ...]:
    """Evaluate ordered outputs; selected numerical primitives need a verifier-owned callback.

    Admitted callbacks are keyed by operation and result numerical policy.
    Diagnostic callbacks for unresolved source operations are separate and
    keyed by the full semantic node signature (type, attributes, and maps). Neither
    callback set is serialized or used by native selection.
    """
    constants = constants or {}
    expected_inputs = {node.id for node in request.nodes if node.effect == "input"}
    expected_constants = {node.id for node in request.nodes if node.effect == "constant"}
    if set(inputs) != expected_inputs or set(constants) != expected_constants:
        raise ValueError("reference inputs or declared constants differ from the graph boundary")
    bindings = {binding.node_id: binding for binding in request.constants}
    values: dict[str, TensorValue] = {}
    for node in request.nodes:
        if node.effect in {"input", "constant"}:
            value = inputs[node.id] if node.effect == "input" else constants[node.id]
            if value.type != node.type:
                raise ValueError(f"reference boundary type differs for {node.id}")
            if node.effect == "constant":
                binding = bindings[node.id]
                width = {
                    "i1-raw": 1,
                    "i8": 1,
                    "i32-le": 4,
                    "i64-le": 8,
                    "fp8_e4m3-raw": 1,
                    "bf16-le-raw": 2,
                    "f32-le-raw": 4,
                    "f64-le-raw": 8,
                }[binding.encoding]
                signed = value.type.numerical_policy not in {"raw-byte-copy", "raw-bit-pattern"}
                expected = b"".join(element.to_bytes(width, "little", signed=signed) for element in value.elements)
                if expected.hex() != binding.data_hex:
                    raise ValueError(f"reference constant differs from declared compiler bytes for {node.id}")
            values[node.id] = value
            continue
        if node.effect != "pure":
            raise ValueError(f"reference evaluator has no semantics for {node.id}")
        operands = tuple(values[child] for child in node.inputs)
        from .scalar_reference import OPERATIONS, evaluate_scalar_region

        if node.op in OPERATIONS:
            values[node.id] = evaluate_scalar_region(node, operands)
            continue
        if node.op == "mask":
            from .masks import evaluate_mask

            values[node.id] = evaluate_mask(node, operands)
            continue
        if node.op == "f32_nonzero":
            if (
                len(operands) != 1
                or operands[0].type.dtype != "f32"
                or operands[0].type.numerical_policy != "raw-bit-pattern"
                or node.type != TensorType(operands[0].type.shape, "i1", "boolean")
                or node.attrs != (("numeric_semantics", "f32-une-zero-v1"),)
                or node.index_maps != pointwise_identity_maps(len(node.type.shape), 1)
            ):
                raise ValueError("truth conversion has incompatible types, maps or semantics")
            values[node.id] = TensorValue(
                node.type, tuple(int((bits & 0x7FFFFFFF) != 0) for bits in operands[0].elements)
            )
            continue
        if node.op.startswith("source_"):
            primitive = (source_primitives or {}).get(node.semantic_key())
            if primitive is None:
                if node.op in {
                    "source_f32_matmul_unqualified",
                    "source_f32_reduce_add_unqualified",
                    "source_f32_reduce_max_unqualified",
                }:
                    raise ValueError(
                        "source f32 computation has unresolved reduction order and needs a qualified reference"
                    )
                raise ValueError("source computation has no qualified numerical reference")
            result = primitive(node, operands)
            if not isinstance(result, TensorValue) or result.type != node.type:
                raise ValueError(f"source reference primitive returned a wrong type for {node.id}")
            values[node.id] = result
            continue
        if node.op == "identity" and len(operands) == 1:
            if node.attrs or node.index_maps or operands[0].type != node.type:
                raise ValueError("reference identity has unsupported attributes, maps, or type")
            values[node.id] = TensorValue(node.type, operands[0].elements)
            continue
        if node.op == "reshape":
            if (
                len(operands) != 1
                or node.attrs
                or node.index_maps
                or (operands[0].type.dtype, operands[0].type.numerical_policy)
                != (node.type.dtype, node.type.numerical_policy)
                or prod(operands[0].type.shape) != prod(node.type.shape)
            ):
                raise ValueError("reference reshape has unsupported attributes, maps, shape, or policy")
            values[node.id] = TensorValue(node.type, operands[0].elements)
            continue
        if node.op == "concat":
            axis = dict(node.attrs).get("axis")
            rank = len(node.type.shape)
            if (
                len(operands) < 2
                or type(axis) is not int
                or not 0 <= axis < rank
                or node.attrs != (("axis", axis),)
                or node.index_maps
                or any(
                    len(operand.type.shape) != rank
                    or (operand.type.dtype, operand.type.numerical_policy)
                    != (node.type.dtype, node.type.numerical_policy)
                    or any(operand.type.shape[dim] != node.type.shape[dim] for dim in range(rank) if dim != axis)
                    for operand in operands
                )
                or sum(operand.type.shape[axis] for operand in operands) != node.type.shape[axis]
            ):
                raise ValueError("reference concat has an invalid axis, type, or shape")
            outer = prod(node.type.shape[:axis])
            inner = prod(node.type.shape[axis + 1 :])
            result = []
            for prefix in range(outer):
                for operand in operands:
                    width = operand.type.shape[axis] * inner
                    result.extend(operand.elements[prefix * width : (prefix + 1) * width])
            values[node.id] = TensorValue(node.type, tuple(result))
            continue
        if node.op == "transpose":
            source = operands[0] if len(operands) == 1 else None
            rank = len(node.type.shape)
            maps = node.index_maps
            source_map, output_map = maps if len(maps) == 2 else (None, None)
            identity = tuple(tuple(int(loop == axis) for loop in range(rank)) for axis in range(rank))
            if (
                node.attrs
                or len(operands) != 1
                or source_map is None
                or output_map is None
                or source_map.loop_rank != rank
                or output_map.loop_rank != rank
                or len(source_map.coefficients) != rank
                or source_map.offsets != (0,) * rank
                or output_map.offsets != (0,) * rank
                or output_map.coefficients != identity
                or any(sum(row) != 1 or any(value not in (0, 1) for value in row) for row in source_map.coefficients)
                or any(sum(row[axis] for row in source_map.coefficients) != 1 for axis in range(rank))
                or any(
                    source.type.shape[axis] != node.type.shape[row.index(1)]
                    for axis, row in enumerate(source_map.coefficients)
                )
                or (node.type.dtype, node.type.numerical_policy) != (source.type.dtype, source.type.numerical_policy)
            ):
                raise ValueError("reference transpose has unsupported maps, shape, or numerical policy")
            values[node.id] = TensorValue(
                node.type,
                tuple(
                    source.elements[_flat_index(source_map.apply(indices), source.type.shape)]
                    for indices in index_product(*(range(extent) for extent in node.type.shape))
                ),
            )
            continue
        if node.op == "indexed_insert_slice":
            if len(node.inputs) != 3 or node.index_maps:
                raise ValueError("indexed update needs three operands without static maps")
            update, destination, positions = (values[key] for key in node.inputs)
            axis = dict(node.attrs).get("axis")
            if (
                type(axis) is not int
                or not 0 <= axis < len(node.type.shape)
                or dict(node.attrs) != {"axis": axis, "index_domain": "nonnegative-below-destination-axis"}
                or destination.type != node.type
                or update.type.dtype != node.type.dtype
                or update.type.numerical_policy != node.type.numerical_policy
                or positions.type != TensorType((1,), "i64", "signed-i64")
                or len(update.type.shape) != len(node.type.shape)
                or update.type.shape[axis] != 1
                or any(update.type.shape[d] != node.type.shape[d] for d in range(len(node.type.shape)) if d != axis)
            ):
                raise ValueError("indexed update types or bounds differ")
            position = positions.elements[0]
            if not 0 <= position < node.type.shape[axis]:
                raise ValueError("indexed update position is outside its destination")
            output = list(destination.elements)
            for coordinate in index_product(*(range(n) for n in update.type.shape)):
                target = list(coordinate)
                target[axis] = position
                src = dst = 0
                for d, n in zip(coordinate, update.type.shape, strict=True):
                    src = src * n + d
                for d, n in zip(target, node.type.shape, strict=True):
                    dst = dst * n + d
                output[dst] = update.elements[src]
            values[node.id] = TensorValue(node.type, tuple(output))
            continue
        if node.op == "insert_slice":
            if len(operands) != 2 or node.attrs or len(node.index_maps) != 3:
                raise ValueError("insert_slice needs one source, one destination, and static maps")
            source, destination = operands
            rank = len(source.type.shape)
            source_map, destination_map, result_map = node.index_maps
            if (
                destination.type != node.type
                or len(destination.type.shape) != rank
                or (source.type.dtype, source.type.numerical_policy)
                != (destination.type.dtype, destination.type.numerical_policy)
                or source_map.loop_rank != rank
                or destination_map.loop_rank != rank
                or destination_map != result_map
                or source_map.coefficients
                != tuple(tuple(int(axis == dim) for axis in range(rank)) for dim in range(rank))
                or source_map.offsets != (0,) * rank
                or len(destination_map.coefficients) != rank
                or any(
                    row[dim] <= 0 or any(value != 0 for axis, value in enumerate(row) if axis != dim)
                    for dim, row in enumerate(destination_map.coefficients)
                )
                or any(offset < 0 for offset in destination_map.offsets)
            ):
                raise ValueError("insert_slice has an unsupported type or index map")
            result = list(destination.elements)
            for indices in index_product(*(range(extent) for extent in source.type.shape)):
                placed = destination_map.apply(indices)
                if any(not 0 <= index < extent for index, extent in zip(placed, node.type.shape)):
                    raise ValueError("insert_slice index exceeds destination")
                source_flat = 0
                destination_flat = 0
                for index, extent in zip(indices, source.type.shape):
                    source_flat = source_flat * extent + index
                for index, extent in zip(placed, node.type.shape):
                    destination_flat = destination_flat * extent + index
                result[destination_flat] = source.elements[source_flat]
            values[node.id] = TensorValue(node.type, tuple(result))
            continue
        if node.op == "gather":
            if len(operands) != 1 or node.attrs or len(node.index_maps) != 2:
                raise ValueError("gather needs one input and two static index maps")
            (source,) = operands
            input_map, output_map = node.index_maps
            rank = len(node.type.shape)
            if (
                (source.type.dtype, source.type.numerical_policy) != (node.type.dtype, node.type.numerical_policy)
                or input_map.loop_rank != rank
                or output_map.loop_rank != rank
                or output_map.coefficients
                != tuple(tuple(int(axis == dim) for axis in range(rank)) for dim in range(rank))
                or output_map.offsets != (0,) * rank
            ):
                raise ValueError("gather has an unsupported type or output map")
            result = []
            for indices in index_product(*(range(extent) for extent in node.type.shape)):
                read = input_map.apply(indices)
                if any(not 0 <= index < extent for index, extent in zip(read, source.type.shape)):
                    raise ValueError("gather reads outside its declared source")
                flat = 0
                for index, extent in zip(read, source.type.shape):
                    flat = flat * extent + index
                result.append(source.elements[flat])
            values[node.id] = TensorValue(node.type, tuple(result))
            continue
        if node.op == "signed_i64_to_f32":
            if (
                len(operands) != 1
                or operands[0].type != TensorType(node.type.shape, "i64", "signed-i64")
                or node.type != TensorType(node.type.shape, "f32", "raw-bit-pattern")
                or dict(node.attrs) != {"numeric_semantics": "signed-i64-to-f32-rne-v1"}
                or node.index_maps != pointwise_identity_maps(len(node.type.shape), 1)
            ):
                raise ValueError("signed integer cast has an unsupported type, map or rounding policy")
            result = []
            for integer in operands[0].elements:
                # Binary64 supplies a nearby binary32 candidate. Compare exact
                # integer distances to its neighbors to avoid double rounding
                # at i64 values just above a binary32 midpoint.
                approximate = unpack("<I", pack("<f", float(abs(integer))))[0]
                if integer == 0:
                    result.append(0)
                    continue
                bits = min(
                    range(approximate - 1, approximate + 2),
                    key=lambda candidate: (
                        abs(Fraction(unpack("<f", pack("<I", candidate))[0]) - abs(integer)),
                        candidate & 1,
                    ),
                )
                result.append(bits | (0x80000000 if integer < 0 else 0))
            values[node.id] = TensorValue(node.type, tuple(result))
            continue
        if node.op in {"f32_add", "f32_sub", "f32_mul", "f32_div"}:
            if (
                len(operands) != 2
                or node.type.dtype != "f32"
                or node.type.numerical_policy != "raw-bit-pattern"
                or any(
                    operand.type.dtype != "f32" or operand.type.numerical_policy != "raw-bit-pattern"
                    for operand in operands
                )
                or dict(node.attrs) != {"numeric_semantics": f"ieee754-f32-{node.op[4:]}-rne-v1"}
            ):
                raise ValueError("f32 pointwise arithmetic has an unsupported type or semantic contract")
            maps = _checked_f32_pointwise_maps(node, operands)
            result = []
            for output_indices in index_product(*(range(extent) for extent in node.type.shape)):
                left, right = tuple(
                    operand.elements[_flat_index(mapping.apply(output_indices), operand.type.shape)]
                    for operand, mapping in zip(operands, maps, strict=True)
                )
                if node.op != "f32_mul":
                    if node.op == "f32_div":
                        result.append(_f32_div_rne(left, right))
                        continue
                    result.append(_f32_add_rne(left, right ^ (0x80000000 if node.op == "f32_sub" else 0)))
                    continue
                if any((bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) for bits in (left, right)):
                    raise ValueError("f32 multiplication NaN payload is not qualified by this reference")
                # Two finite binary32 significands multiply to at most 48 bits,
                # so binary64 holds the exact product before one binary32 rounding.
                product = unpack("<f", pack("<I", left))[0] * unpack("<f", pack("<I", right))[0]
                if isnan(product):
                    raise ValueError("f32 multiplication NaN result is not qualified by this reference")
                try:
                    result.append(unpack("<I", pack("<f", product))[0])
                except OverflowError:
                    result.append((0x80000000 if product < 0 else 0) | 0x7F800000)
            values[node.id] = TensorValue(node.type, tuple(result))
            continue
        if node.op == "f32_relu_ordered_select":
            if (
                len(operands) != 1
                or operands[0].type != node.type
                or node.type.dtype != "f32"
                or node.type.numerical_policy != "raw-bit-pattern"
                or dict(node.attrs) != {"numeric_semantics": "f32-ordered-lt-positive-zero-select-v1"}
                or node.index_maps != pointwise_identity_maps(len(node.type.shape), 1)
            ):
                raise ValueError("f32 zero select has an unsupported numerical or indexing contract")
            result = []
            for bits in operands[0].elements:
                magnitude = bits & 0x7FFFFFFF
                nan = (magnitude & 0x7F800000) == 0x7F800000 and (magnitude & 0x007FFFFF) != 0
                if nan and not (magnitude & 0x00400000):
                    raise ValueError("signaling NaN compare is outside the qualified f32 reference domain")
                result.append(0 if bits & 0x80000000 and magnitude and not nan else bits)
            values[node.id] = TensorValue(node.type, tuple(result))
            continue
        if node.op in {"f32_reduce_add", "i64_reduce_add"}:
            integer = node.op == "i64_reduce_add"
            dtype, policy = ("i64", "signed-i64") if integer else ("f32", "raw-bit-pattern")
            if len(operands) != 2 or (node.type.dtype, node.type.numerical_policy) != (dtype, policy):
                raise ValueError("reduction has an unsupported type or operand count")
            source, init = operands
            axis = dict(node.attrs).get("axis")
            if (
                type(axis) is not int
                or len(source.type.shape) < 1
                or not 0 <= axis < len(source.type.shape)
                or source.type.dtype != dtype
                or source.type.numerical_policy != policy
                or init.type != node.type
                or node.type.shape != tuple(extent for dim, extent in enumerate(source.type.shape) if dim != axis)
                or dict(node.attrs)
                != (
                    {
                        "axis": axis,
                        "minimum_index_bits": max(2, max(source.type.shape).bit_length() + 1),
                        "numeric_semantics": "i64-add-modulo-2pow64-v1",
                    }
                    if integer
                    else {
                        "axis": axis,
                        "reduction_order": "axis-ascending",
                        "numeric_semantics": "ieee754-f32-add-rne-v1",
                    }
                )
                or node.index_maps != _f32_reduce_maps(len(source.type.shape), axis)
            ):
                raise ValueError("f32 reduction has an unsupported numerical or indexing contract")
            result = []
            for output_indices in index_product(*(range(extent) for extent in node.type.shape)):
                output_flat = 0
                for index, extent in zip(output_indices, node.type.shape):
                    output_flat = output_flat * extent + index
                accumulator = init.elements[output_flat]
                for reduction in range(source.type.shape[axis]):
                    source_indices = (*output_indices[:axis], reduction, *output_indices[axis:])
                    source_flat = 0
                    for index, extent in zip(source_indices, source.type.shape):
                        source_flat = source_flat * extent + index
                    accumulator = (
                        (source.elements[source_flat] + accumulator + (1 << 63)) % (1 << 64) - (1 << 63)
                        if integer
                        else _f32_add_rne(source.elements[source_flat], accumulator)
                    )
                result.append(accumulator)
            values[node.id] = TensorValue(node.type, tuple(result))
            continue
        if node.op == "embedding_lookup":
            if len(operands) != 2 or node.index_maps:
                raise ValueError("embedding lookup needs indices and a table without affine maps")
            indices, table = operands
            if (
                indices.type.dtype != "i64"
                or indices.type.numerical_policy != "signed-i64"
                or table.type.dtype not in {"f32", "f64", "bf16", "fp8_e4m3"}
                or table.type.numerical_policy != "raw-bit-pattern"
                or len(table.type.shape) != 2
                or node.type
                != TensorType((*indices.type.shape, table.type.shape[1]), table.type.dtype, "raw-bit-pattern")
                or dict(node.attrs)
                != {
                    "index_domain": "nonnegative-below-table-rows",
                    "minimum_index_bits": max(2, (table.type.shape[0] - 1).bit_length() + 1),
                }
            ):
                raise ValueError("embedding lookup has an unsupported type or index contract")
            rows, columns = table.type.shape
            result = []
            for index in indices.elements:
                if not 0 <= index < rows:
                    raise ValueError("embedding index is outside the table")
                result.extend(table.elements[index * columns : (index + 1) * columns])
            values[node.id] = TensorValue(node.type, tuple(result))
            continue
        if node.op == "relu" and node.type.dtype == "bf16":
            if (
                len(operands) != 1
                or operands[0].type.dtype != "bf16"
                or operands[0].type.shape != node.type.shape
                or node.type.numerical_policy != "raw-bit-pattern"
                or node.index_maps != pointwise_identity_maps(len(node.type.shape), 1)
                or dict(node.attrs) != _BF16_RELU_ATTRS
            ):
                raise ValueError("BF16 ReLU has an unsupported type, policy, or semantic contract")
            source = operands[0].elements
            if any((bits & 0x7F80) == 0x7F80 or ((bits & 0x7F80) == 0 and (bits & 0x007F) != 0) for bits in source):
                raise ValueError("BF16 ReLU input is outside finite-normal-or-zero domain")
            values[node.id] = TensorValue(node.type, tuple(0 if bits & 0x8000 else bits for bits in source))
            continue
        primitive = (primitives or {}).get((node.op, node.type.numerical_policy))
        if primitive is not None:
            result = primitive(node, operands)
            if not isinstance(result, TensorValue) or result.type != node.type:
                raise ValueError(f"reference primitive returned a wrong type for {node.id}")
            values[node.id] = result
            continue
        if node.type.dtype == "i8" and node.type.numerical_policy == "raw-byte-copy":
            raise ValueError("raw-byte reference admits only type-preserving identity")
        if node.type.numerical_policy == "raw-bit-pattern":
            raise ValueError("raw-bit reference admits only type-preserving identity or transpose")
        if node.op == "matmul_accumulate":
            if node.attrs or node.index_maps != _matmul_maps() or len(operands) != 3:
                raise ValueError("reference matmul has unsupported attributes or indexing maps")
            lhs, rhs, init = operands
            if node.type.dtype != "i32" or node.type.numerical_policy != "i32-wrap-k-ascending":
                raise ValueError("reference matmul has an unsupported numerical policy")
            expected_input_policy = "signed-i8" if lhs.type.dtype == "i8" else "i32-wrap-k-ascending"
            if (
                lhs.type.dtype not in {"i8", "i32"}
                or lhs.type.numerical_policy != expected_input_policy
                or rhs.type.dtype != lhs.type.dtype
                or rhs.type.numerical_policy != expected_input_policy
                or init.type != node.type
            ):
                raise ValueError("reference matmul operand type or policy mismatch")
            if len(lhs.type.shape) != 2 or len(rhs.type.shape) != 2:
                raise ValueError("reference matmul needs rank-two operands")
            m, k = lhs.type.shape
            right_k, n = rhs.type.shape
            if right_k != k or node.type.shape != (m, n):
                raise ValueError("reference matmul shape mismatch")
            result = []
            for row in range(m):
                for col in range(n):
                    acc = init.elements[row * n + col]
                    for reduction in range(k):
                        product = _wrap_i32(lhs.elements[row * k + reduction] * rhs.elements[reduction * n + col])
                        acc = _wrap_i32(acc + product)
                    result.append(acc)
            values[node.id] = TensorValue(node.type, tuple(result))
            continue
        if node.index_maps or node.attrs:
            raise ValueError(f"reference evaluator has no semantics for {node.id}")
        if node.type.dtype != "i32" or node.type.numerical_policy != "exact-i32":
            raise ValueError(f"reference evaluator has no numerical policy for {node.id}")
        if (
            node.op in {"add", "multiply"}
            and len(operands) == 2
            and all(operand.type == node.type for operand in operands)
        ):
            operation = int.__add__ if node.op == "add" else int.__mul__
            result = tuple(
                _i32(operation(left, right))
                for left, right in zip(
                    operands[0].elements,
                    operands[1].elements,
                )
            )
        elif node.op == "matmul" and len(operands) == 2:
            left, right = operands
            if len(left.type.shape) != 2 or len(right.type.shape) != 2:
                raise ValueError("exact-i32 matmul needs rank-two operands")
            m, k = left.type.shape
            right_k, n = right.type.shape
            if right_k != k or node.type.shape != (m, n):
                raise ValueError("exact-i32 matmul shape mismatch")
            if any(operand.type.dtype != "i32" or operand.type.numerical_policy != "exact-i32" for operand in operands):
                raise ValueError("exact-i32 matmul operand policy mismatch")
            result = tuple(
                _i32(
                    sum(
                        left.elements[row * k + reduction] * right.elements[reduction * n + col]
                        for reduction in range(k)
                    )
                )
                for row in range(m)
                for col in range(n)
            )
        else:
            raise ValueError(f"reference evaluator has no semantics for {node.op}")
        values[node.id] = TensorValue(node.type, result)
    return tuple(values[output] for output in request.outputs)
