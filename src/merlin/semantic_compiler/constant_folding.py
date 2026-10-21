"""Fold declared integer and IEEE constants with replayable, bounded evidence.

Only the exact primitive policies below are eligible. Runtime values, target
arithmetic and unqualified math intrinsics are never evaluated by this pass.
"""

from dataclasses import replace
from hashlib import sha256
from math import prod

from .model import ConstantBinding, SemanticNode, TensorType, pointwise_identity_maps
from .predicate_values import PREDICATE_OPERATIONS, predicate_element, predicate_widths

_FORMATS = {"bf16": (16, 7, 127), "f32": (32, 23, 127), "f64": (64, 52, 1023)}
_CASTS = {
    "signed_i64_to_f32": ("i64", "f32", "signed-i64-to-f32-rne-v1"),
    "signed_i64_to_f64": ("i64", "f64", "signed-i64-to-f64-rne-v1"),
    "f32_to_f64": ("f32", "f64", "f32-to-f64-exact-v1"),
    "f64_to_f32": ("f64", "f32", "f64-to-f32-rne-v1"),
    "bf16_to_f32": ("bf16", "f32", "bf16-to-f32-exact-v1"),
    "f32_to_bf16": ("f32", "bf16", "f32-to-bf16-rne-v1"),
}
_ARITHMETIC = {f"{dtype}_{op}" for dtype in ("i64", "bf16", "f64") for op in ("add", "sub", "mul", "div")}
_ARITHMETIC.remove("i64_div")


def _components(bits, dtype):
    width, fraction, bias = _FORMATS[dtype]
    sign = bits >> (width - 1)
    exponent, significand = (bits >> fraction) & ((1 << (width - fraction - 1)) - 1), bits & ((1 << fraction) - 1)
    if exponent == (1 << (width - fraction - 1)) - 1:
        if significand:
            raise ValueError("NaN arithmetic payload is unqualified")
        return sign, None, 0
    return sign, significand | ((1 << fraction) if exponent else 0), (exponent or 1) - bias - fraction


def _encode(numerator, denominator, power, dtype, zero_sign=0):
    """Round an integer ratio times a binary power, without host float operations."""
    width, fraction, bias = _FORMATS[dtype]
    sign, magnitude = int(numerator < 0) if numerator else zero_sign, abs(numerator)
    if not magnitude:
        return sign << (width - 1)
    exponent = magnitude.bit_length() - denominator.bit_length()
    below = magnitude < denominator << exponent if exponent >= 0 else magnitude << -exponent < denominator
    exponent += power - int(below)
    quantum = max(exponent - fraction, 1 - bias - fraction)
    shift = power - quantum
    divisor = denominator << max(-shift, 0)
    significand, remainder = divmod(magnitude << max(shift, 0), divisor)
    significand += 2 * remainder > divisor or 2 * remainder == divisor and bool(significand & 1)
    if significand >= 1 << (fraction + 1):
        significand >>= 1
        quantum += 1
    if significand < 1 << fraction:
        return (sign << (width - 1)) | significand
    encoded_exponent = quantum + fraction + bias
    maximum = (1 << (width - fraction - 1)) - 1
    if encoded_exponent >= maximum:
        return (sign << (width - 1)) | (maximum << fraction)
    return (sign << (width - 1)) | (encoded_exponent << fraction) | (significand - (1 << fraction))


def _binary(left, right, dtype, operation):
    width, fraction, _ = _FORMATS[dtype]
    if operation == "sub":
        right ^= 1 << (width - 1)
        operation = "add"
    a_sign, a, a_power = _components(left, dtype)
    b_sign, b, b_power = _components(right, dtype)
    negative = a_sign ^ b_sign
    infinity = ((1 << (width - fraction - 1)) - 1) << fraction
    if operation == "add":
        if a is None or b is None:
            if a is None and b is None and a_sign != b_sign:
                raise ValueError("NaN result is unqualified")
            return left if a is None else right
        common = min(a_power, b_power)
        total = (-1 if a_sign else 1) * (a << (a_power - common)) + (-1 if b_sign else 1) * (b << (b_power - common))
        return _encode(total, 1, common, dtype, int(a == b == 0 and a_sign == b_sign == 1))
    if operation == "mul":
        if a is None or b is None:
            if a == 0 or b == 0:
                raise ValueError("NaN result is unqualified")
            return (negative << (width - 1)) | infinity
        return _encode((-1 if negative else 1) * a * b, 1, a_power + b_power, dtype, negative)
    if a is None and b is None or a == b == 0:
        raise ValueError("NaN result is unqualified")
    if a is None or b == 0:
        return (negative << (width - 1)) | infinity
    if b is None or a == 0:
        return negative << (width - 1)
    return _encode((-1 if negative else 1) * a, b, a_power - b_power, dtype, negative)


def _policy(node, operands):
    shape = node.type.shape
    if not shape or node.index_maps != pointwise_identity_maps(len(shape), len(operands)):
        raise ValueError("constant primitive requires equal-shape identity maps")
    if node.op in PREDICATE_OPERATIONS:
        predicate_widths(node, operands)
        return
    if node.op in _CASTS:
        source_dtype, dtype, policy = _CASTS[node.op]
        if len(operands) != 1:
            raise ValueError("constant conversion has a different arity")
    else:
        dtype, operation = node.op.split("_")
        source_dtype = dtype
        if len(operands) != 2:
            raise ValueError("constant arithmetic has a different arity")
        policy = f"signed-i64-{operation}-wrap-v1" if dtype == "i64" else f"ieee754-{dtype}-{operation}-rne-v1"
    expected = TensorType(shape, source_dtype, "signed-i64" if source_dtype == "i64" else "raw-bit-pattern")
    if (
        any(operand.type != expected for operand in operands)
        or node.type != TensorType(shape, dtype, "signed-i64" if dtype == "i64" else "raw-bit-pattern")
        or dict(node.attrs) != {"numeric_semantics": policy}
    ):
        raise ValueError("constant primitive types or numerical policy differ")


def _element(node, arguments):
    if node.op in _CASTS:
        source_dtype, dtype, _ = _CASTS[node.op]
        bits = arguments[0]
        if source_dtype == "i64":
            return _encode(bits if bits < 1 << 63 else bits - (1 << 64), 1, 0, dtype)
        sign, mantissa, power = _components(bits, source_dtype)
        if mantissa is None:
            width, fraction, _ = _FORMATS[dtype]
            return (sign << (width - 1)) | (((1 << (width - fraction - 1)) - 1) << fraction)
        return _encode((-1 if sign else 1) * mantissa, 1, power, dtype, sign)
    dtype, operation = node.op.split("_")
    if dtype == "i64":
        a, b = arguments
        return (a + b if operation == "add" else a - b if operation == "sub" else a * b) % (1 << 64)
    return _binary(*arguments, dtype, operation)


def fold_source_constants(source, *, element_budget=65536, eliminate_dead=False):
    """Fold byte-bound constants, optionally eliminating unused pure computation.

    Receipts store the original nodes, allowing source reconstruction without
    duplicating model weights. Budget exhaustion and unqualified NaN behavior
    defer an optimization; they never establish hardware unsupportability.
    Optional dead-code elimination removes only unused pure computations;
    every boundary input, constant, ordered output and state edge is retained.
    """
    if type(element_budget) is not int or element_budget < 0:
        raise ValueError("constant folding needs a nonnegative element budget")
    if type(eliminate_dead) is not bool:
        raise ValueError("dead computation elimination needs an explicit Boolean")
    if source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError("constant folding requires logical source values")
    nodes, bindings = {}, {c.node_id: c for c in source.constants}
    folds, deferred, work = [], [], 0
    for node in source.nodes:
        operands = tuple(nodes[key] for key in node.inputs)
        if (
            node.effect != "pure"
            or node.op not in _CASTS
            and node.op not in _ARITHMETIC
            and node.op not in PREDICATE_OPERATIONS
            or any(n.effect != "constant" for n in operands)
        ):
            nodes[node.id] = node
            continue
        cost = prod(node.type.shape) + sum(prod(n.type.shape) for n in operands)
        if work + cost > element_budget:
            deferred.append({"node": node.id, "reason": "element_budget"})
            nodes[node.id] = node
            continue
        try:
            _policy(node, operands)
            payloads = [bytes.fromhex(bindings[n.id].data_hex) for n in operands]
            predicate = node.op in PREDICATE_OPERATIONS
            if predicate:
                widths, width = predicate_widths(node, operands)
            else:
                widths = [8 if n.type.dtype == "i64" else _FORMATS[n.type.dtype][0] // 8 for n in operands]
                width = 8 if node.type.dtype == "i64" else _FORMATS[node.type.dtype][0] // 8
            result = bytearray()
            for index in range(prod(node.type.shape)):
                args = [
                    int.from_bytes(data[index * w : (index + 1) * w], "little")
                    for data, w in zip(payloads, widths, strict=True)
                ]
                value = predicate_element(node, args, widths) if predicate else _element(node, args)
                result.extend(value.to_bytes(width, "little"))
        except ValueError as error:
            deferred.append({"node": node.id, "reason": str(error)})
            nodes[node.id] = node
            work += cost
            continue
        work += cost
        encoding = {"i1": "i1-raw", "i8": "i8", "i32": "i32-le", "i64": "i64-le",
                    "fp8_e4m3": "fp8_e4m3-raw"}.get(node.type.dtype, node.type.dtype + "-le-raw")
        bindings[node.id] = ConstantBinding(node.id, encoding, result.hex())
        nodes[node.id] = replace(node, op="constant", inputs=(), attrs=(), index_maps=(), effect="constant")
        folds.append(
            {
                "original": node.record(),
                "operand_sha256": [sha256(data).hexdigest() for data in payloads],
                "result_sha256": sha256(result).hexdigest(),
            }
        )
    folded = replace(source, nodes=tuple(nodes.values()), constants=tuple(bindings.values())) if folds else source
    removed = []
    if eliminate_dead:
        live = set(folded.outputs)
        for node in reversed(folded.nodes):
            if node.effect != "pure" or node.id in live:
                live.add(node.id)
                live.update(node.inputs)
        removed = [{"position": i, "node": n.record()} for i, n in enumerate(folded.nodes) if n.id not in live]
        if removed:
            folded = replace(folded, nodes=tuple(n for n in folded.nodes if n.id in live))
    before_digest = source.digest()
    return folded, {
        "schema": "merlin.semantic_constant_folding.v2" if eliminate_dead else "merlin.semantic_constant_folding.v1",
        "element_budget": element_budget,
        "element_work": work,
        "before_digest": before_digest,
        "after_digest": folded.digest() if folds or removed else before_digest,
        "folds": folds,
        "deferred": deferred,
        **({"dead_computations": removed} if eliminate_dead else {}),
    }


def verify_constant_folding(source, record):
    """Reconstruct the original source and replay exact rule eligibility/bytes."""
    eliminate_dead = isinstance(record, dict) and record.get("schema") == "merlin.semantic_constant_folding.v2"
    fields = {"schema", "element_budget", "element_work", "before_digest", "after_digest", "folds", "deferred"}
    if eliminate_dead:
        fields.add("dead_computations")
    if (
        not isinstance(record, dict)
        or set(record) != fields
        or record["schema"] not in ("merlin.semantic_constant_folding.v1", "merlin.semantic_constant_folding.v2")
    ):
        raise ValueError("constant folding receipt has missing or unknown fields")
    optimized = source
    if eliminate_dead:
        if not isinstance(record["dead_computations"], list):
            raise ValueError("dead computation receipt needs an ordered list")
        removed = {}
        for row in record["dead_computations"]:
            if not isinstance(row, dict) or set(row) != {"position", "node"}:
                raise ValueError("dead computation receipt has missing or unknown fields")
            position = row["position"]
            node = SemanticNode.from_record(row["node"])
            if type(position) is not int or position in removed or node.effect != "pure":
                raise ValueError("dead computation receipt has an invalid position or effect")
            removed[position] = node
        total = len(source.nodes) + len(removed)
        if any(not 0 <= i < total for i in removed):
            raise ValueError("dead computation receipt position exceeds the original graph")
        retained = iter(source.nodes)
        source = replace(source, nodes=tuple(removed[i] if i in removed else next(retained) for i in range(total)))
    originals = tuple(SemanticNode.from_record(row["original"]) for row in record["folds"])
    by_id = {node.id: node for node in originals}
    if len(by_id) != len(originals) or any(source.node(key).effect != "constant" for key in by_id):
        raise ValueError("constant folding receipt has repeated or nonconstant results")
    original = replace(
        source,
        nodes=tuple(by_id.get(n.id, n) for n in source.nodes),
        constants=tuple(c for c in source.constants if c.node_id not in by_id),
    )
    if original.digest() != record["before_digest"]:
        raise ValueError("constant folding original source changed")
    replayed, expected = fold_source_constants(
        original, element_budget=record["element_budget"], eliminate_dead=eliminate_dead
    )
    if replayed.digest() != optimized.digest() or expected != record:
        raise ValueError("constant folding bytes or derivation differ from replay")
