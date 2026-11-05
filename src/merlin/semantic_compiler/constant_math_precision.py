"""Explicit source-math precision changes for declared constant positive powers.

This selects correctly rounded mathematical power under a pinned MPFR helper.
It does not assert bitwise equivalence to an unqualified framework intrinsic.
Ordinary exact constant folding remains a separate transformation.
"""

import json
from collections import Counter
from dataclasses import replace
from hashlib import sha256
from math import prod

from .constant_math_backend import MathHelper, validate_policy
from .model import ConstantBinding, SemanticNode, TensorType, pointwise_identity_maps


def _signature(node, operands):
    dtype = node.type.dtype
    if dtype not in {"f32", "f64"} or not node.type.shape:
        raise ValueError("constant power needs a positive-rank f32/f64 tensor")
    width = 32 if dtype == "f32" else 64
    attrs = dict(node.attrs)
    literal = None
    if attrs.get("operation") == "literal_base_pow_f32":
        bits = attrs.get("literal_bits")
        if (
            node.op != "source_f32_composite_unqualified"
            or dtype != "f32"
            or len(operands) != 1
            or set(attrs) != {"operation", "literal_bits", "numeric_semantics"}
            or attrs["numeric_semantics"] != "source-f32-composite-unqualified-v1"
            or not isinstance(bits, str)
            or len(bits) != 10
            or not bits.startswith("0x")
            or any(c not in "0123456789abcdef" for c in bits[2:])
        ):
            raise ValueError("constant literal power source signature differs")
        literal = int(bits, 16)
    elif (
        node.op != f"source_{dtype}_powf_unqualified"
        or len(operands) != 2
        or attrs != {"numeric_semantics": f"source-{dtype}-powf-unqualified-v1"}
    ):
        raise ValueError("constant power source signature differs")
    if (
        node.type != TensorType(node.type.shape, dtype, "raw-bit-pattern")
        or any(operand.type != node.type for operand in operands)
        or node.index_maps != pointwise_identity_maps(len(node.type.shape), len(operands))
    ):
        raise ValueError("constant power source types or logical maps differ")
    return width, literal


def prepare_constant_math(source, policy, *, helper=None, element_limit=None):
    """Replace only whole, checked constant results under an explicit policy."""
    policy = validate_policy(policy)
    helper = helper or MathHelper.selected(policy)
    if helper.identity != {key: policy[key] for key in ("helper_sha256", "dependencies")}:
        raise ValueError("constant math implementation differs from the selected policy")
    limit = policy["element_budget"] if element_limit is None else element_limit
    if type(limit) is not int or not 0 <= limit <= policy["element_budget"]:
        raise ValueError("constant math remaining work budget is invalid")
    if source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError("constant math precision needs logical source values")
    nodes, bindings = {}, {item.node_id: item for item in source.constants}
    folds, deferred, work = [], [], 0
    for node in source.nodes:
        nodes[node.id] = node
        if node.effect != "pure" or not (
            node.op in {"source_f32_powf_unqualified", "source_f64_powf_unqualified"}
            or node.op == "source_f32_composite_unqualified"
            and dict(node.attrs).get("operation") == "literal_base_pow_f32"
        ):
            continue
        operands = tuple(nodes[key] for key in node.inputs)
        if any(item.effect != "constant" for item in operands):
            deferred.append({"node": node.id, "status": "unsupported_semantics", "reason": "runtime power operand"})
            continue
        count = prod(node.type.shape)
        if work + count > limit:
            deferred.append({"node": node.id, "status": "resource_limit", "reason": "constant math element budget"})
            continue
        try:
            width, literal = _signature(node, operands)
            payloads = [bytes.fromhex(bindings[item.id].data_hex) for item in operands]
            if any(bindings[item.id].encoding != node.type.dtype + "-le-raw" for item in operands):
                raise ValueError("constant power input encoding differs")
            words = [
                tuple(int.from_bytes(data[i : i + width // 8], "little") for i in range(0, len(data), width // 8))
                for data in payloads
            ]
            pairs = zip(*words, strict=True) if literal is None else ((literal, x) for x in words[0])
            result, precisions = bytearray(), Counter()
            for base, exponent in pairs:
                bits, precision = helper.power(width, base, exponent, policy["max_precision_bits"])
                result.extend(bits.to_bytes(width // 8, "little"))
                precisions[str(precision)] += 1
        except (ValueError, ArithmeticError) as error:
            deferred.append(
                {
                    "node": node.id,
                    "status": "resource_limit" if isinstance(error, ArithmeticError) else "unsupported_semantics",
                    "reason": str(error),
                }
            )
            work += count
            continue
        work += count
        bindings[node.id] = ConstantBinding(node.id, node.type.dtype + "-le-raw", result.hex())
        nodes[node.id] = replace(node, op="constant", inputs=(), attrs=(), index_maps=(), effect="constant")
        folds.append(
            {
                "original": node.record(),
                "operand_sha256": [sha256(data).hexdigest() for data in payloads],
                "result_sha256": sha256(result).hexdigest(),
                "bound_precision_counts": dict(sorted(precisions.items())),
            }
        )
    identity = source.source_identity
    if folds:
        identity = sha256(json.dumps([identity, policy], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    prepared = (
        replace(source, nodes=tuple(nodes.values()), constants=tuple(bindings.values()), source_identity=identity)
        if folds
        else source
    )
    return prepared, {
        "schema": "merlin.constant_math_precision_receipt.v1",
        "kind": "precision_change",
        "policy": policy,
        "element_limit": limit,
        "element_work": work,
        "before_source_identity": source.source_identity,
        "before_digest": source.digest(),
        "after_digest": prepared.digest(),
        "folds": folds,
        "deferred": deferred,
        "evidence": "conditional_MPFR_directed_bounds; framework_equivalence_not_established",
    }


def verify_constant_math(source, record, *, helper=None):
    """Reconstruct source math and replay its selected library/bounds evidence."""
    fields = {
        "schema",
        "kind",
        "policy",
        "element_limit",
        "element_work",
        "before_source_identity",
        "before_digest",
        "after_digest",
        "folds",
        "deferred",
        "evidence",
    }
    if (
        not isinstance(record, dict)
        or set(record) != fields
        or record["schema"] != "merlin.constant_math_precision_receipt.v1"
        or record["kind"] != "precision_change"
        or source.digest() != record["after_digest"]
    ):
        raise ValueError("constant math precision receipt has invalid fields or result identity")
    originals = tuple(SemanticNode.from_record(row["original"]) for row in record["folds"])
    by_id = {node.id: node for node in originals}
    if len(by_id) != len(originals) or any(source.node(key).effect != "constant" for key in by_id):
        raise ValueError("constant math receipt has repeated or nonconstant results")
    original = replace(
        source,
        nodes=tuple(by_id.get(n.id, n) for n in source.nodes),
        constants=tuple(c for c in source.constants if c.node_id not in by_id),
        source_identity=record["before_source_identity"],
    )
    if original.digest() != record["before_digest"]:
        raise ValueError("constant math original source changed")
    replayed, expected = prepare_constant_math(
        original, record["policy"], helper=helper, element_limit=record["element_limit"]
    )
    if replayed.digest() != source.digest() or expected != record:
        raise ValueError("constant math bytes or bounds evidence differ from replay")
    return original
