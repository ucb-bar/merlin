"""Exact compile-time movement and ordered scans of declared tensor constants.

These operations read only compiler-owned bytes. They do not interpret runtime
regions, change numerical policies or erase physical storage obligations.
"""

from itertools import product
from math import prod

from .model import TensorType, pointwise_identity_maps

OPERATIONS = frozenset({"identity", "reshape", "gather", "ordered_f32_scan_f64"})


def fold_tensor(node, operands, payloads):
    if len(operands) != 1 or len(payloads) != 1:
        raise ValueError("constant tensor operation needs one operand")
    source, data = operands[0], payloads[0]
    if (source.type.dtype, source.type.numerical_policy) != (node.type.dtype, node.type.numerical_policy):
        raise ValueError("constant tensor movement cannot change dtype or numerical policy")
    if node.op in {"identity", "reshape"}:
        if (node.attrs or node.index_maps or prod(source.type.shape) != prod(node.type.shape)
                or node.op == "identity" and source.type != node.type):
            raise ValueError("constant identity or reshape has different attributes, maps or shape")
        return data
    if node.op == "ordered_f32_scan_f64":
        if source.type != node.type:
            raise ValueError("constant scan source type differs from its output")
        return _scan(node, data)
    if node.op != "gather" or node.attrs or len(node.index_maps) != 2 or not node.type.shape:
        raise ValueError("constant gather needs one checked affine read and an identity output map")
    read, write = node.index_maps
    rank = len(node.type.shape)
    if read.loop_rank != rank or write != pointwise_identity_maps(rank, 1)[1]:
        raise ValueError("constant gather has a different loop domain or output map")
    width = len(data) // prod(source.type.shape)
    if not width or width * prod(source.type.shape) != len(data):
        raise ValueError("constant gather has an invalid element width")
    result = bytearray()
    for coordinate in product(*(range(extent) for extent in node.type.shape)):
        indices = read.apply(coordinate)
        if len(indices) != len(source.type.shape) or any(
            not 0 <= index < extent for index, extent in zip(indices, source.type.shape, strict=True)
        ):
            raise ValueError("constant gather reads outside its source")
        flat = 0
        for index, extent in zip(indices, source.type.shape, strict=True):
            flat = flat * extent + index
        result.extend(data[flat * width:(flat + 1) * width])
    return bytes(result)


def _scan(node, data):
    # The independent semantic evaluator uses a separate implementation.
    from .constant_folding import _binary, _cast_bits

    attributes = dict(node.attrs)
    stride, extent = attributes.get("stride"), attributes.get("extent")
    if (len(node.type.shape) != 1 or node.type != TensorType(node.type.shape, "f32", "raw-bit-pattern")
            or node.index_maps or set(attributes) != {"stride", "extent", "numeric_semantics"}
            or attributes["numeric_semantics"] != "ordered-f32-prefix-f64-rne-v1"
            or type(stride) is not int or type(extent) is not int or stride <= 0 or extent <= 0
            or node.type.shape[0] >= 1 << 63 or node.type.shape[0] % (stride * extent)
            or len(data) != node.type.shape[0] * 4):
        raise ValueError("constant scan has a different numerical policy, layout or lane geometry")
    accumulators, output = {}, bytearray()
    for position in range(node.type.shape[0]):
        lane = position % stride
        first = (position // stride) % extent == 0
        bits = int.from_bytes(data[position * 4:(position + 1) * 4], "little")
        wide = _cast_bits(bits, "f32", "f64")
        summed = _binary(0 if first else accumulators[lane], wide, "f64", "add")
        accumulators[lane] = summed
        output.extend(_cast_bits(summed, "f64", "f32").to_bytes(4, "little"))
    return bytes(output)
