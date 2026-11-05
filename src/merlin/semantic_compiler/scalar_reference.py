"""Independent exact checks for scalar-region casts, BF16 and binary64.

Finite arithmetic uses rational values and one IEEE rounding step. This
evaluator is not a compiler fallback. Unqualified math intrinsics stay outside
its operation set, and NaN-producing arithmetic has no payload guarantee here.
"""

from bisect import bisect_left, bisect_right
from fractions import Fraction
from itertools import product
from math import isfinite, isnan
from struct import pack, unpack

from .model import IndexMap, TensorType, pointwise_identity_maps

OPERATIONS = frozenset(
    {
        "select",
        "bucketize",
        "ordered_f32_scan_f64",
        "i64_cumsum",
        "i64_min_value",
        "i64_first_min_index",
        "f32_neg",
        "f64_neg",
        "f64_add",
        "f64_sub",
        "f64_mul",
        "f64_div",
        "bf16_add",
        "bf16_sub",
        "bf16_mul",
        "bf16_div",
        "i64_add",
        "i64_sub",
        "i64_mul",
        "signed_i64_to_f64",
        "f32_to_f64",
        "f64_to_f32",
        "bf16_to_f32",
        "f32_to_bf16",
        "f32_to_signed_i64",
        "f64_to_signed_i64",
    }
)


def _rounded_ratio(numerator, denominator):
    quotient, remainder = divmod(numerator, denominator)
    return quotient + (2 * remainder > denominator or 2 * remainder == denominator and bool(quotient & 1))


def _round(value, width, zero_sign=0):
    """Round an exact finite rational to the declared float, ties to even."""
    fraction_bits, bias = {16: (7, 127), 32: (23, 127), 64: (52, 1023)}[width]
    sign = int(value < 0) if value else zero_sign
    numerator, denominator = abs(value.numerator), value.denominator
    if numerator == 0:
        return sign << (width - 1)
    exponent = numerator.bit_length() - denominator.bit_length()
    if numerator < denominator << exponent if exponent >= 0 else numerator << -exponent < denominator:
        exponent -= 1
    if exponent >= 1 - bias:
        shift = fraction_bits - exponent
        significand = _rounded_ratio(numerator << max(shift, 0), denominator << max(-shift, 0))
        if significand == 1 << (fraction_bits + 1):
            significand >>= 1
            exponent += 1
        if exponent > bias:
            return (sign << (width - 1)) | (((1 << (width - 1 - fraction_bits)) - 1) << fraction_bits)
        return (sign << (width - 1)) | ((exponent + bias) << fraction_bits) | (significand - (1 << fraction_bits))
    fraction = _rounded_ratio(numerator << (bias - 1 + fraction_bits), denominator)
    return (sign << (width - 1)) | fraction


def _decode(bits, dtype):
    if dtype == "bf16":
        return unpack("<f", (bits << 16).to_bytes(4, "little"))[0]
    return unpack("<f" if dtype == "f32" else "<d", bits.to_bytes(4 if dtype == "f32" else 8, "little"))[0]


def _binary64(operation, left, right):
    return _binary_float(operation, left, right, 64)


def _binary_float(operation, left, right, width):
    dtype, fraction_bits = {16: ("bf16", 7), 64: ("f64", 52)}[width]
    sign = width - 1
    if operation == "sub":
        right ^= 1 << sign
        operation = "add"
    lhs, rhs = _decode(left, dtype), _decode(right, dtype)
    if isnan(lhs) or isnan(rhs):
        raise ValueError(f"{dtype} arithmetic NaN payload is unqualified")
    negative = (left ^ right) >> sign
    infinity = (negative << sign) | (((1 << (width - fraction_bits - 1)) - 1) << fraction_bits)
    if operation == "add":
        if not isfinite(lhs) or not isfinite(rhs):
            if not isfinite(lhs) and not isfinite(rhs) and (left ^ right) >> sign:
                raise ValueError(f"{dtype} arithmetic NaN result is unqualified")
            return left if not isfinite(lhs) else right
        return _round(Fraction(lhs) + Fraction(rhs), width, int(left >> sign == right >> sign == 1))
    if operation == "mul":
        if not isfinite(lhs) or not isfinite(rhs):
            if lhs == 0 or rhs == 0:
                raise ValueError(f"{dtype} arithmetic NaN result is unqualified")
            return infinity
        return _round(Fraction(lhs) * Fraction(rhs), width, negative)
    if operation == "div":
        if lhs == rhs == 0 or not isfinite(lhs) and not isfinite(rhs):
            raise ValueError(f"{dtype} arithmetic NaN result is unqualified")
        if rhs == 0 or not isfinite(lhs):
            return infinity
        if lhs == 0 or not isfinite(rhs):
            return negative << sign
        return _round(Fraction(lhs) / Fraction(rhs), width, negative)
    raise ValueError("unknown floating arithmetic")


def evaluate_scalar_region(node, operands):
    from .reference import TensorValue

    if node.op in {"i64_min_value", "i64_first_min_index"}:
        if len(operands) != 1:
            raise ValueError("minimum/first-index projection requires one source")
        source = operands[0]
        shape, attributes = source.type.shape, dict(node.attrs)
        axis = attributes.get("axis")
        if (
            not shape
            or type(axis) is not int
            or not 0 <= axis < len(shape)
            or source.type != TensorType(shape, "i64", "signed-i64")
            or node.type != TensorType(shape[:axis] + shape[axis + 1 :], "i64", "signed-i64")
            or node.index_maps
            or attributes
            != {
                "axis": axis,
                "minimum_index_bits": max(2, max(shape).bit_length() + 1),
                "numeric_semantics": "signed-i64-min-first-index-v1",
            }
        ):
            raise ValueError("minimum/first-index projection has an invalid type, axis or policy")
        result = []
        for point in product(*(range(n) for n in node.type.shape)):
            candidates = []
            for position in range(shape[axis]):
                indices = point[:axis] + (position,) + point[axis:]
                flat = 0
                for index, extent in zip(indices, shape):
                    flat = flat * extent + index
                candidates.append((source.elements[flat], position))
            pair = min(candidates)  # Tuple ordering checks signed values, then the first index.
            result.append(pair[0 if node.op == "i64_min_value" else 1])
        return TensorValue(node.type, tuple(result))

    if node.op == "i64_cumsum":
        attributes = dict(node.attrs)
        axis = attributes.get("axis")
        shape = node.type.shape
        if (
            len(operands) != 1
            or not shape
            or type(axis) is not int
            or not 0 <= axis < len(shape)
            or operands[0].type.shape != shape
            or operands[0].type.dtype not in {"i1", "i64"}
            or operands[0].type.numerical_policy != ("boolean" if operands[0].type.dtype == "i1" else "signed-i64")
            or node.type != TensorType(shape, "i64", "signed-i64")
            or node.index_maps
            or attributes
            != {
                "axis": axis,
                "minimum_index_bits": max(2, max(shape).bit_length() + 1),
                "numeric_semantics": "signed-i64-prefix-sum-wrap-v1",
            }
        ):
            raise ValueError("integer prefix sum has an unsupported type, axis or numerical policy")
        result = [0] * len(operands[0].elements)
        for rest in product(*(range(1 if dim == axis else n) for dim, n in enumerate(shape))):
            accumulated = 0
            for coordinate in range(shape[axis]):
                flat = 0
                for dim, n in enumerate(shape):
                    flat = flat * n + (coordinate if dim == axis else rest[dim])
                accumulated = (accumulated + operands[0].elements[flat] + (1 << 63)) % (1 << 64) - (1 << 63)
                result[flat] = accumulated
        return TensorValue(node.type, tuple(result))

    if node.op == "ordered_f32_scan_f64":
        attributes = dict(node.attrs)
        stride, extent = attributes.get("stride"), attributes.get("extent")
        if (
            len(operands) != 1
            or len(node.type.shape) != 1
            or node.type != TensorType(node.type.shape, "f32", "raw-bit-pattern")
            or operands[0].type != node.type
            or node.index_maps
            or set(attributes) != {"stride", "extent", "numeric_semantics"}
            or attributes["numeric_semantics"] != "ordered-f32-prefix-f64-rne-v1"
            or type(stride) is not int
            or type(extent) is not int
            or stride <= 0
            or extent <= 0
            or node.type.shape[0] >= 1 << 63
            or node.type.shape[0] % (stride * extent)
        ):
            raise ValueError("ordered scan has an unsupported type, lane geometry or numerical policy")
        accumulators, output = {}, []
        for position, bits in enumerate(operands[0].elements):
            lane = position % stride
            first = (position // stride) % extent == 0
            current = int.from_bytes(pack("<d", _decode(bits, "f32")), "little")
            summed = _binary64("add", 0 if first else accumulators[lane], current)
            accumulators[lane] = summed
            value = _decode(summed, "f64")
            output.append(
                _round(Fraction(value), 32, summed >> 63)
                if isfinite(value)
                else int.from_bytes(pack("<f", value), "little")
            )
        return TensorValue(node.type, tuple(output))

    shape = node.type.shape
    if node.op == "bucketize":
        if len(operands) != 2:
            raise ValueError("boundary count needs values and boundaries")
        source, boundaries = operands
        if len(boundaries.type.shape) != 1:
            raise ValueError("boundary count needs a rank-one boundary tensor")
        count = boundaries.type.shape[0]
        rank = len(shape)
        projection = IndexMap(
            rank + 1, tuple(tuple(int(i == j) for j in range(rank + 1)) for i in range(rank)), (0,) * rank
        )
        boundary_map = IndexMap(rank + 1, ((0,) * rank + (1,),), (0,))
        if (
            source.type != TensorType(shape, "f32", "raw-bit-pattern")
            or boundaries.type != TensorType((count,), "f32", "raw-bit-pattern")
            or node.type != TensorType(shape, "i64", "signed-i64")
            or count >= 1 << 63
            or node.index_maps != (projection, boundary_map, projection)
            or set(dict(node.attrs)) != {"numeric_semantics", "right", "boundary_count"}
            or dict(node.attrs)["numeric_semantics"] != "f32-unordered-boundary-count-i64-v1"
            or dict(node.attrs)["boundary_count"] != count
            or type(dict(node.attrs)["right"]) is not bool
        ):
            raise ValueError("boundary count has an unsupported type, map or semantic policy")
        limits = tuple(_decode(bits, "f32") for bits in boundaries.elements)
        if any(isnan(x) for x in limits) or any(a > b for a, b in zip(limits, limits[1:])):
            raise ValueError("boundary count requires nondecreasing non-NaN boundaries")
        search = bisect_right if dict(node.attrs)["right"] else bisect_left
        return TensorValue(
            node.type,
            tuple(count if isnan(value := _decode(bits, "f32")) else search(limits, value) for bits in source.elements),
        )
    if (
        not shape
        or any(value.type.shape != shape for value in operands)
        or (node.index_maps != pointwise_identity_maps(len(shape), len(operands)))
    ):
        raise ValueError("scalar operation needs equal shapes and identity maps")
    attrs, operation = dict(node.attrs), node.op

    def raw(dtype):
        return TensorType(shape, dtype, "raw-bit-pattern")

    integer = TensorType(shape, "i64", "signed-i64")
    types = tuple(value.type for value in operands)
    if operation == "select":
        if attrs or types != (TensorType(shape, "i1", "boolean"), node.type, node.type):
            raise ValueError("select needs an exact Boolean condition and matching values")
        result = tuple(a if flag else b for flag, a, b in zip(*(v.elements for v in operands), strict=True))
    elif operation in {"f32_neg", "f64_neg"}:
        dtype = operation[:3]
        if (
            types != (raw(dtype),)
            or node.type != raw(dtype)
            or attrs != {"numeric_semantics": f"{dtype}-sign-bit-flip-v1"}
        ):
            raise ValueError("floating negate has an unsupported type or policy")
        result = tuple(bits ^ (1 << (31 if dtype == "f32" else 63)) for bits in operands[0].elements)
    elif operation.startswith(("f64_", "bf16_")) and operation.split("_")[1] in {"add", "sub", "mul", "div"}:
        dtype, arithmetic = operation.split("_")
        if (
            types != (raw(dtype), raw(dtype))
            or node.type != raw(dtype)
            or (attrs != {"numeric_semantics": f"ieee754-{dtype}-{arithmetic}-rne-v1"})
        ):
            raise ValueError("floating arithmetic has an unsupported type or policy")
        result = tuple(
            _binary_float(arithmetic, a, b, 16 if dtype == "bf16" else 64)
            for a, b in zip(*(v.elements for v in operands), strict=True)
        )
    elif operation in {"i64_add", "i64_sub", "i64_mul"}:
        arithmetic = operation[4:]
        if (
            types != (integer, integer)
            or node.type != integer
            or attrs != {"numeric_semantics": f"signed-i64-{arithmetic}-wrap-v1"}
        ):
            raise ValueError("integer scalar arithmetic has an unsupported type or policy")
        result = []
        for a, b in zip(*(v.elements for v in operands), strict=True):
            value = a + b if arithmetic == "add" else a - b if arithmetic == "sub" else a * b
            result.append((value + (1 << 63)) % (1 << 64) - (1 << 63))
        result = tuple(result)
    else:
        conversions = {
            "signed_i64_to_f64": (integer, raw("f64"), "signed-i64-to-f64-rne-v1"),
            "f32_to_f64": (raw("f32"), raw("f64"), "f32-to-f64-exact-v1"),
            "f64_to_f32": (raw("f64"), raw("f32"), "f64-to-f32-rne-v1"),
            "bf16_to_f32": (raw("bf16"), raw("f32"), "bf16-to-f32-exact-v1"),
            "f32_to_bf16": (raw("f32"), raw("bf16"), "f32-to-bf16-rne-v1"),
            "f32_to_signed_i64": (raw("f32"), integer, "f32-to-signed-i64-toward-zero-in-range-v1"),
            "f64_to_signed_i64": (raw("f64"), integer, "f64-to-signed-i64-toward-zero-in-range-v1"),
        }
        if operation not in conversions:
            raise ValueError("scalar reference has no semantics for this operation")
        source, destination, policy = conversions[operation]
        if types != (source,) or node.type != destination or attrs != {"numeric_semantics": policy}:
            raise ValueError("scalar conversion has an unsupported type or numerical policy")
        result = []
        for bits in operands[0].elements:
            if operation == "signed_i64_to_f64":
                result.append(_round(Fraction(bits), 64))
                continue
            if operation == "bf16_to_f32":
                if bits & 0x7F80 == 0x7F80 and bits & 0x7F:
                    raise ValueError("scalar conversion NaN payload is unqualified")
                result.append(bits << 16)
                continue
            value = _decode(bits, source.dtype)
            if isnan(value):
                raise ValueError("scalar conversion NaN payload is unqualified")
            if destination.dtype == "i64":
                if not isfinite(value) or not -(1 << 63) <= int(value) < (1 << 63):
                    raise ValueError("floating-to-integer source is outside its defined domain")
                result.append(int(value))
            elif destination.dtype == "bf16":
                high, low = bits >> 16, bits & 65535
                result.append(high + (low > 32768 or low == 32768 and bool(high & 1)))
            elif not isfinite(value):
                result.append(int.from_bytes(pack("<f" if destination.dtype == "f32" else "<d", value), "little"))
            else:
                result.append(
                    _round(
                        Fraction(value),
                        32 if destination.dtype == "f32" else 64,
                        int(value == 0 and bits >> (31 if source.dtype == "f32" else 63)),
                    )
                )
        result = tuple(result)
    return TensorValue(node.type, result)
