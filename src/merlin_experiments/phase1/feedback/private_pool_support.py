"""Operator-private structural proof for value-only FP32 max-pool source.

This recognizes a narrowly typed frontend lowering, not compiled numerical
equivalence.  Host admission and a linked whole-program build remain separate.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from merlin.common import mlir_query as mq
from merlin.common.digest import is_sha256
from merlin.runtime.modelinputs import file_sha256

POOL_ATEN = "aten.max_pool2d_with_indices.default"
SCOPE = (
    "host-required value-only FP32 source structure when present; device pools retain standard CG/OA obligations; "
    "linked build only, no value equivalence or execution"
)
PENDING = "source_structural_pool_value_pending_build"
LINKED = "source_structural_pool_value_linked"


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"source max-pool proof: {message}")


def _shape(value: Any, rank: int) -> tuple[int, ...]:
    from xdsl.dialects.builtin import TensorType

    typ = value.type
    _need(isinstance(typ, TensorType) and typ.has_static_shape(), "dynamic or non-tensor operand")
    result = tuple(typ.get_shape())
    _need(len(result) == rank and all(type(x) is int and x > 0 for x in result), "invalid tensor rank or extent")
    _need(str(typ.element_type) == "f32", "non-FP32 tensor")
    return result


def _ints(value: Any) -> tuple[int, ...]:
    result = tuple(value.iter_values())
    _need(all(type(x) is int for x in result), "non-static integer property")
    return result


def _owner(value: Any, name: str) -> Any:
    op = value.owner
    _need(
        op is not None and mq.op_name(op) == name and len(op.results) == 1 and op.results[0] is value,
        f"expected {name} producer",
    )
    return op


def _neg_inf(value: Any) -> None:
    splat = _owner(value, "tensor.splat")
    _need(len(splat.operands) == 1, "invalid initializer splat")
    constant = _owner(splat.operands[0], "arith.constant")
    payload = constant.properties.get("value") or constant.attributes.get("value")
    number = getattr(getattr(payload, "value", None), "data", None)
    _need(type(number) is float and math.isinf(number) and number < 0, "initializer is not -inf")


def _pair(value: Any, default: tuple[int, int] | None = None) -> tuple[int, int]:
    if value is None or value == []:
        _need(default is not None, "missing kernel")
        return default
    if type(value) is int:
        return (value, value)
    _need(
        isinstance(value, list) and len(value) in (1, 2) and all(type(x) is int for x in value),
        "non-static pool argument",
    )
    return (value[0], value[0]) if len(value) == 1 else (value[0], value[1])


def _geometry(node: Mapping[str, Any], input_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> dict[str, Any]:
    args = node.get("args")
    _need(
        isinstance(args, list) and 2 <= len(args) <= 6 and node.get("kwargs") == {},
        "unrecognized frontend pool arguments",
    )
    kernel = _pair(args[1])
    stride = _pair(args[2] if len(args) > 2 else None, kernel)
    pad = _pair(args[3] if len(args) > 3 else None, (0, 0))
    dilation = _pair(args[4] if len(args) > 4 else None, (1, 1))
    _need(len(args) < 6 or args[5] is False, "ceil-mode pool")
    _need(min(*kernel, *stride, *dilation) > 0 and min(*pad) >= 0, "invalid pool geometry")
    _need(input_shape[:2] == output_shape[:2], "pool changes batch or channel")
    for axis in (0, 1):
        extent = input_shape[axis + 2] + 2 * pad[axis] - dilation[axis] * (kernel[axis] - 1) - 1
        _need(extent // stride[axis] + 1 == output_shape[axis + 2], "non-floor output geometry")
    return {"kernel": kernel, "stride": stride, "padding": pad, "dilation": dilation}


def _body(op: Any) -> None:
    _need(len(op.regions) == 1 and len(op.regions[0].blocks) == 1, "non-single reducer body")
    block = op.regions[0].blocks[0]
    ops = list(block.ops)
    _need(len(block.args) == 3 and all(str(arg.type) == "f32" for arg in block.args), "invalid reducer arguments")
    _need(
        [mq.op_name(item) for item in ops] == ["arith.cmpf", "arith.cmpf", "arith.ori", "arith.select", "linalg.yield"],
        "reducer is not exact ordered select",
    )
    _need(
        all(all(key.startswith("prov.") for key in item.attributes) for item in ops)
        and all(set(item.properties) == {"predicate", "fastmath"} for item in ops[:2])
        and all(not item.properties for item in ops[2:]),
        "reducer carries an unrecognized semantic property or attribute",
    )
    isnan, greater, union, winner, yielded = ops
    nan_predicate = isnan.properties.get("predicate") or isnan.attributes.get("predicate")
    greater_predicate = greater.properties.get("predicate") or greater.attributes.get("predicate")
    _need(getattr(getattr(nan_predicate, "value", None), "data", None) == 14, "missing unordered NaN comparison")
    _need(getattr(getattr(greater_predicate, "value", None), "data", None) == 2, "missing strict greater comparison")
    flags = [item.properties.get("fastmath") or item.attributes.get("fastmath") for item in (isnan, greater)]
    _need(all(item is not None and item.data == frozenset() for item in flags), "fast-math changes reducer semantics")
    _need(
        tuple(isnan.operands) == (block.args[0], block.args[0])
        and tuple(greater.operands) == (block.args[0], block.args[2])
        and tuple(union.operands) == (isnan.results[0], greater.results[0])
        and tuple(winner.operands) == (union.results[0], block.args[0], block.args[2])
        and tuple(yielded.operands) == (winner.results[0],),
        "reducer operand order changed",
    )


def verify_pool_generic(op: Any, node: Mapping[str, Any]) -> dict[str, Any]:
    """Check all typed storage, geometry, padding, maps, and reducer facts."""
    from xdsl.dialects.builtin import AffineMapAttr
    from xdsl.dialects.linalg.attrs import IteratorType
    from xdsl.ir.affine import AffineExpr, AffineMap

    _need(
        mq.op_name(op) == "linalg.generic" and len(op.operands) == 3 and len(op.results) == 1,
        "not a three-operand single-result generic",
    )
    _need(
        set(op.properties) == {"indexing_maps", "iterator_types", "operandSegmentSizes"}
        and all(key.startswith("prov.") for key in op.attributes),
        "pool generic carries an unrecognized semantic property or attribute",
    )
    source, window, init = op.operands
    padded_shape, window_shape, output_shape, result_shape = (
        _shape(source, 4),
        _shape(window, 2),
        _shape(init, 4),
        _shape(op.results[0], 4),
    )
    _need(output_shape == result_shape, "result differs from accumulator storage")
    _need(mq.op_name(window.owner) == "tensor.empty" and len(window.owner.operands) == 0, "window is not shape-only")
    _neg_inf(init)
    input_value = source
    if getattr(source.owner, "name", None) == "tensor.insert_slice":
        insertion = _owner(source, "tensor.insert_slice")
        _need(len(insertion.operands) == 2, "dynamic padding slice")
        input_value, base = insertion.operands
        _neg_inf(base)
        _need(_shape(base, 4) == padded_shape, "padding base/result storage differs")
        offsets = _ints(insertion.properties["static_offsets"])
        sizes = _ints(insertion.properties["static_sizes"])
        strides = _ints(insertion.properties["static_strides"])
        _need(strides == (1, 1, 1, 1) and sizes == _shape(input_value, 4), "padding slice changes values")
    else:
        offsets = (0, 0, 0, 0)
    input_shape = _shape(input_value, 4)
    geometry = _geometry(node, input_shape, output_shape)
    kh, kw = geometry["kernel"]
    sh, sw = geometry["stride"]
    ph, pw = geometry["padding"]
    dh, dw = geometry["dilation"]
    _need(
        window_shape == (kh, kw)
        and offsets == (0, 0, ph, pw)
        and padded_shape == (input_shape[0], input_shape[1], input_shape[2] + 2 * ph, input_shape[3] + 2 * pw),
        "padding/window do not match frontend geometry",
    )
    _need((ph == 0 and pw == 0) == (input_value is source), "padding chain missing or unexpected")
    maps = op.properties.get("indexing_maps")
    loops = op.properties.get("iterator_types")
    _need(
        maps is not None and loops is not None and len(maps.data) == 3 and len(loops.data) == 6,
        "pool maps or loops missing",
    )
    D = AffineExpr.dimension
    expected = [
        AffineMap(6, 0, (D(0), D(1), D(2) * sh + D(4) * dh, D(3) * sw + D(5) * dw)),
        AffineMap(6, 0, (D(4), D(5))),
        AffineMap(6, 0, (D(0), D(1), D(2), D(3))),
    ]
    _need(
        all(
            isinstance(item, AffineMapAttr) and item.data == want
            for item, want in zip(maps.data, expected, strict=True)
        ),
        "pool affine access order changed",
    )
    _need(
        [item.data for item in loops.data] == [IteratorType.PARALLEL] * 4 + [IteratorType.REDUCTION] * 2,
        "pool iterator order changed",
    )
    _body(op)
    return geometry


def prove_pool_source(
    capture: Path, module: Any, inventory: Mapping[str, Any], *, raw_sha256: str, normalized_sha256: str
) -> dict[str, Any]:
    """Reconcile all tagged pool ordinals; trace-prove only the host-required subset."""
    parsed = tuple(mq.walk(module))
    normalization = inventory.get("capture_normalization")
    _need(
        inventory.get("n_operations") == len(parsed)
        and inventory.get("capture_sha256") == raw_sha256
        and isinstance(normalization, Mapping)
        and normalization.get("output_sha256") == normalized_sha256,
        "inventory/source identity mismatch",
    )
    actual = {
        i
        for i, op in enumerate(parsed)
        if mq.attr_str(op, "prov.aten") == POOL_ATEN and mq.op_name(op) == "linalg.generic"
    }
    rows = [
        r
        for r in inventory.get("signatures", [])
        if r.get("frontend_op") == POOL_ATEN and r.get("mlir_operation") == "linalg.generic"
    ]
    listed: list[int] = []
    host_ordinals: list[int] = []
    device_ordinals: list[int] = []
    for row in rows:
        ordinals = row.get("ordinals")
        _need(
            row.get("disposition") in {"host_required", "hardware_admitted"}
            and isinstance(ordinals, list)
            and type(row.get("count")) is int
            and row["count"] == len(ordinals),
            "pool inventory has no exact host or device occurrences",
        )
        for ordinal in ordinals:
            _need(
                type(ordinal) is int and 0 <= ordinal < len(parsed),
                "pool occurrence ordinal is not an in-range integer",
            )
            listed.append(ordinal)
            if row["disposition"] == "host_required":
                host_ordinals.append(ordinal)
            else:
                device_ordinals.append(ordinal)
    _need(
        len(listed) == len(set(listed)) and set(listed) == actual,
        "pool source occurrences omitted, duplicated, or non-host",
    )
    if not host_ordinals:
        return {
            "status": PENDING,
            "scope": SCOPE,
            "raw_source_sha256": raw_sha256,
            "normalized_source_sha256": normalized_sha256,
            "count": 0,
            "occurrences": [],
            "device_ordinals": sorted(device_ordinals),
        }
    trace_path = capture / "frontend-trace.json"
    receipt = json.loads((capture / "capture_receipt.json").read_text(encoding="utf-8"))
    trace_sha = file_sha256(trace_path)
    _need(
        receipt.get("artifacts", {}).get("frontend-trace.json", {}).get("sha256") == trace_sha
        and receipt.get("artifacts", {}).get("model.mlir", {}).get("sha256") == raw_sha256,
        "trace or source not bound to capture receipt",
    )
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    graph = trace.get("graphs", {}).get("prepared", {})
    _need(
        trace.get("schema") == "m2m.frontend_trace.v1"
        and trace.get("status") == "complete"
        and trace.get("mlir", {}).get("sha256") == raw_sha256
        and graph.get("status") == "complete",
        "frontend trace incomplete or source-mismatched",
    )
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    correspondence = trace.get("mlir", {}).get("source_correspondence")
    _need(
        isinstance(nodes, list) and isinstance(edges, list) and isinstance(correspondence, list),
        "trace has no prepared graph or source join",
    )
    by_id = {node.get("id"): node for node in nodes if isinstance(node, Mapping)}
    _need(len(by_id) == len(nodes), "duplicate prepared node identity")
    occurrences = []
    for ordinal in sorted(host_ordinals):
        op = parsed[ordinal]
        ids = op.attributes.get("prov.source_node_ids")
        node_ids = tuple(getattr(item, "data", None) for item in getattr(ids, "data", ()))
        _need(len(node_ids) == 1 and node_ids[0] in by_id, "pool has no unique prepared source node")
        node_id = node_ids[0]
        node = by_id[node_id]
        _need(node.get("target") == POOL_ATEN and node.get("op") == "call_function", "pool source node changed")
        joined = [
            item
            for item in correspondence
            if item.get("node_id") == node_id
            and ordinal in item.get("mlir_ordinals", [])
            and item.get("status") == "lowered"
        ]
        other_claims = [
            item
            for item in correspondence
            if item.get("node_id") != node_id and ordinal in item.get("mlir_ordinals", [])
        ]
        _need(len(joined) == 1 and not other_claims, "pool ordinal is not uniquely joined to frontend node")
        results = node.get("results")
        output_shape = tuple(op.results[0].type.get_shape())
        _need(
            isinstance(results, list)
            and len(results) == 2
            and results[0].get("id") == f"{node_id}:v0"
            and results[0].get("dtype") == "float32"
            and results[1].get("id") == f"{node_id}:v1"
            and results[1].get("dtype") == "int64"
            and all(result.get("shape") == list(output_shape) for result in results),
            "pool trace has no typed values-and-indices result",
        )
        node_args = node.get("args")
        _need(isinstance(node_args, list) and bool(node_args), "pool trace has no input argument")
        input_ref = node_args[0]
        incoming = [
            edge for edge in edges if edge.get("consumer_node_id") == node_id and edge.get("argument_path") == "args/0"
        ]
        input_shape = _shape(op.operands[0], 4)
        if getattr(op.operands[0].owner, "name", None) == "tensor.insert_slice":
            input_shape = _shape(op.operands[0].owner.operands[0], 4)
        _need(
            isinstance(input_ref, Mapping)
            and len(incoming) == 1
            and incoming[0].get("producer_node_id") == input_ref.get("node_id")
            and incoming[0].get("producer_value_id") == input_ref.get("value_id")
            and incoming[0].get("dtype") == "float32"
            and incoming[0].get("shape") == list(input_shape),
            "pool trace input differs from lowered source",
        )
        consumers = [edge for edge in edges if edge.get("producer_node_id") == node_id]
        _need(
            bool(consumers)
            and all(
                edge.get("producer_value_id") == f"{node_id}:v0" and edge.get("argument_path") == "args/0"
                for edge in consumers
            ),
            "pool indices or tuple are live",
        )
        for edge in consumers:
            consumer = by_id.get(edge.get("consumer_node_id")) or {}
            _need(
                consumer.get("target") == "<built-in function getitem>"
                and consumer.get("args") == [{"node_id": node_id, "value_id": f"{node_id}:v0"}, 0]
                and consumer.get("kwargs") == {},
                "pool consumer is not value-only getitem(0)",
            )
        geometry = verify_pool_generic(op, node)
        occurrences.append({"ordinal": ordinal, "source_node_id": node_id, "geometry": geometry})
    _need(is_sha256(raw_sha256), "invalid source digest")
    return {
        "status": PENDING,
        "scope": SCOPE,
        "raw_source_sha256": raw_sha256,
        "normalized_source_sha256": normalized_sha256,
        "frontend_trace_sha256": trace_sha,
        "capture_receipt_sha256": file_sha256(capture / "capture_receipt.json"),
        "count": len(occurrences),
        "occurrences": occurrences,
        "device_ordinals": sorted(device_ordinals),
    }


def linked_pool_complete(source: Mapping[str, Any], entry: Mapping[str, Any], candidate_sha256: str) -> bool:
    """Do not credit source structure without its exact linked whole-program bytes."""
    proof = source.get("pool_value_support")
    if not isinstance(proof, Mapping) or proof.get("status") != LINKED or proof.get("scope") != SCOPE:
        return False
    occurrences = proof.get("occurrences")
    device_ordinals = proof.get("device_ordinals")
    if (
        not isinstance(occurrences, list)
        or not isinstance(device_ordinals, list)
        or any(type(ordinal) is not int or ordinal < 0 for ordinal in device_ordinals)
        or device_ordinals != sorted(set(device_ordinals))
        or type(proof.get("count")) is not int
        or proof["count"] != len(occurrences)
        or any(
            not isinstance(item, Mapping)
            or type(item.get("ordinal")) is not int
            or item["ordinal"] < 0
            or not isinstance(item.get("source_node_id"), str)
            or not isinstance(item.get("geometry"), Mapping)
            for item in occurrences
        )
        or [item["ordinal"] for item in occurrences] != sorted({item["ordinal"] for item in occurrences})
    ):
        return False
    if (
        not is_sha256(source.get("source_sha256"))
        or proof.get("raw_source_sha256") != source["source_sha256"]
        or entry.get("source_sha256") != source["source_sha256"]
        or not is_sha256(proof.get("normalized_source_sha256"))
        or proof.get("normalized_source_sha256") != source.get("normalized_source_sha256")
    ):
        return False
    if occurrences and (
        not is_sha256(proof.get("frontend_trace_sha256"))
        or not is_sha256(proof.get("capture_receipt_sha256"))
        or proof.get("capture_receipt_sha256") != source.get("capture_receipt_sha256")
    ):
        return False
    linked = proof.get("linked_build")
    return bool(
        isinstance(linked, Mapping)
        and set(linked) == {"candidate_tree_sha256", "capture_tree_sha256", "elf_sha256"}
        and all(is_sha256(linked[key]) for key in linked)
        and linked["candidate_tree_sha256"] == candidate_sha256
        and linked["capture_tree_sha256"] == entry.get("capture_tree_sha256")
        and linked["elf_sha256"] == entry.get("elf_sha256")
    )
