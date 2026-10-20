"""Host-private source witness for prepared literal signed-i64 ``aten.arange``.

This checks prepared graph literals against their exact typed MLIR lowering.
Original graph ancestry is recorded, not proved numerically equivalent to the
prepared graph. It grants no host operation, numerical result, selected index
width, or linked compiler claim. The width is an explicit caller premise until
the selected-build index observation is joined to the actual linked program.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from merlin.common import mlir_query as mq
from merlin.frontends.capture_normalization import CaptureNormalizationError, normalize_capture_mlir
from merlin.targetgen.application_inventory import verify_capture_receipt
from merlin_experiments.phase1.feedback import private_linked_source_preparation as preparation

PENDING = "source_literal_i64_arange_pending_build"
SCOPE = (
    "prepared literal i64 range and typed source lowering under caller-supplied index width; "
    "no frontend equivalence, host admission or compiler equivalence"
)
_SIGNED_I64 = (-(1 << 63), (1 << 63) - 1)
#: The only spelling each admitted ``aten.arange`` source option may carry.
_RANGE_SOURCE_OPTIONS: dict[str, Any] = {
    "dtype": {"kind": "dtype", "value": "torch.int64"},
    "device": {"kind": "device", "value": "cpu"},
    "layout": {"kind": "layout", "value": "torch.strided"},
    "pin_memory": False,
}
_BODY = (
    "linalg.index",
    "arith.index_cast",
    "arith.constant",
    "arith.muli",
    "arith.constant",
    "arith.addi",
    "linalg.yield",
)
_LOWERING = ("tensor.empty", "linalg.generic", *_BODY)


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(f"literal arange source refusal: {reason}")


def _ids(op: Any, key: str) -> tuple[str, ...]:
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    value = op.attributes.get(key)
    if not isinstance(value, ArrayAttr) or any(not isinstance(item, StringAttr) for item in value.data):
        return ()
    return tuple(item.data for item in value.data)


def _range(node: dict[str, Any]) -> tuple[int, int, int, int]:
    """Read only CPU, contiguous, literal signed-i64 graph results."""
    target, args, kwargs, results = (node.get("target"), node.get("args"), node.get("kwargs"), node.get("results"))
    arity = {"aten.arange.default": (1,), "aten.arange.start": (2,), "aten.arange.start_step": (2, 3)}
    _need(
        node.get("op") == "call_function"
        and target in arity
        and isinstance(args, list)
        and len(args) in arity[target]
        and all(type(value) is int and _SIGNED_I64[0] <= value <= _SIGNED_I64[1] for value in args),
        "range arguments are not fixed signed-i64 literals",
    )
    # An omitted option is not given a value here. Only a spelled option is
    # compared; the single result below must show the i64, CPU, strided and
    # contiguous storage that an omitted one leaves to the frontend.
    _need(
        isinstance(kwargs, dict)
        and set(kwargs) <= set(_RANGE_SOURCE_OPTIONS)
        and all(
            type(kwargs[name]) is type(expected) and kwargs[name] == expected
            for name, expected in _RANGE_SOURCE_OPTIONS.items()
            if name in kwargs
        ),
        "range has dynamic or non-CPU source options",
    )
    _need(
        isinstance(results, list) and len(results) == 1 and isinstance(results[0], dict),
        "range has no single result",
    )
    result = results[0]
    _need(
        result.get("kind") == "tensor"
        and result.get("dtype") == result.get("storage_dtype") == "int64"
        and result.get("device") == "cpu"
        and result.get("layout") == "torch.strided"
        and result.get("stride") == [1]
        and isinstance(result.get("shape"), list)
        and len(result["shape"]) == 1
        and type(result["shape"][0]) is int
        and result["shape"][0] >= 0,
        "range has no static contiguous i64 result",
    )
    start, end, step = (0, args[0], 1) if len(args) == 1 else (args[0], args[1], args[2] if len(args) == 3 else 1)
    _need(step != 0, "range step is zero")
    _need(
        start == end or (start < end and step > 0) or (start > end and step < 0),
        "range direction is inconsistent with source operator semantics",
    )
    extent = max(0, (end - start + step - 1) // step) if step > 0 else max(0, (start - end - step - 1) // -step)
    _need(extent == result["shape"][0], "range extent differs from half-open literal semantics")
    if extent:
        last = start + (extent - 1) * step
        _need(
            _SIGNED_I64[0] <= min(start, last) <= max(start, last) <= _SIGNED_I64[1],
            "range final values exceed signed i64",
        )
    return start, end, step, extent


def _body(op: Any, start: int, step: int, extent: int, index_bits: int) -> None:
    """Recognize only the producer's closed index/cast/mul/add iota body."""
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import (
        AffineMapAttr,
        ArrayAttr,
        DenseArrayBase,
        IndexType,
        IntegerAttr,
        NoneAttr,
        TensorType,
    )
    from xdsl.dialects.linalg.attrs import IteratorType
    from xdsl.dialects.linalg.ops import GenericOp, IndexOp, YieldOp
    from xdsl.ir.affine import AffineMap

    _need(
        type(op) is GenericOp
        and len(op.inputs) == 0
        and len(op.outputs) == len(op.results) == 1
        and not op.successors
        and len(list(op.results[0].uses)) >= 1,
        "range is not a zero-input, single-output registered generic",
    )
    _need(
        set(op.properties) == {"indexing_maps", "iterator_types", "operandSegmentSizes"}
        and all(key.startswith("prov.") for key in op.attributes)
        and isinstance(op.outputs[0].type, TensorType)
        and op.outputs[0].type == op.results[0].type
        and isinstance(op.results[0].type.encoding, NoneAttr)
        and str(op.results[0].type.element_type) == "i64"
        and tuple(op.results[0].type.get_shape()) == (extent,),
        "range has changed properties or typed output",
    )
    maps, iterators, segments = (op.indexing_maps, op.iterator_types, op.properties["operandSegmentSizes"])
    _need(
        isinstance(maps, ArrayAttr)
        and len(maps.data) == 1
        and isinstance(maps.data[0], AffineMapAttr)
        and maps.data[0].data == AffineMap.identity(1)
        and isinstance(iterators, ArrayAttr)
        and len(iterators.data) == 1
        and iterators.data[0].data == IteratorType.PARALLEL
        and isinstance(segments, DenseArrayBase)
        and tuple(segments.iter_values()) == (0, 1),
        "range maps, iterators or operand segments changed",
    )
    init = op.outputs[0].owner
    _need(
        type(init) is tensor.EmptyOp
        and len(init.dynamic_sizes) == 0
        and not init.successors
        and not init.properties
        and all(key.startswith("prov.") for key in init.attributes)
        and len(list(op.outputs[0].uses)) == 1,
        "range initializer is not a private static empty tensor",
    )
    _need(len(op.regions) == 1 and len(op.regions[0].blocks) == 1, "range has another region or block")
    block = op.regions[0].block
    operations = list(block.ops)
    expected_types = (
        IndexOp,
        arith.IndexCastOp,
        arith.ConstantOp,
        arith.MuliOp,
        arith.ConstantOp,
        arith.AddiOp,
        YieldOp,
    )
    _need(
        len(block.args) == 1
        and str(block.args[0].type) == "i64"
        and tuple(type(inner) for inner in operations) == expected_types
        and all(not inner.regions and not inner.successors for inner in operations)
        and all(all(key.startswith("prov.") for key in inner.attributes) for inner in operations),
        "range body contains other operations, effects or semantic attributes",
    )
    idx, cast, step_constant, mul, start_constant, add, yld = operations
    _need(
        set(idx.properties) == {"dim"}
        and idx.dim.value.data == 0
        and isinstance(idx.results[0].type, IndexType)
        and not cast.properties
        and set(step_constant.properties) == set(start_constant.properties) == {"value"}
        and all(
            isinstance(value.value, IntegerAttr)
            and value.value.type == value.result.type
            and str(value.result.type) == "i64"
            for value in (step_constant, start_constant)
        )
        and step_constant.value.value.data == step
        and start_constant.value.value.data == start
        and set(mul.properties) == set(add.properties) == {"overflowFlags"}
        and not mul.overflow_flags.data
        and not add.overflow_flags.data
        and not yld.properties
        and list(cast.operands) == [idx.results[0]]
        and str(cast.results[0].type) == "i64"
        and list(mul.operands) == [cast.results[0], step_constant.results[0]]
        and list(add.operands) == [start_constant.results[0], mul.results[0]]
        and list(yld.operands) == [add.results[0]]
        and all(str(value.results[0].type) == "i64" for value in (mul, add)),
        "range body does not compute the exact literal indexed sequence",
    )
    # Index casts may truncate a wider index, but every used index is nonnegative
    # and fits signed i64. Unflagged i64 mul/add are modulo 2^64: an intermediate
    # product may wrap, yet the final signed value agrees when its mathematical
    # endpoint fits i64. No no-wrap flag or intermediate bound is assumed.
    # The loop's upper bound/termination value also needs the selected signed
    # index width, even when the final element index would fit one slot wider.
    _need(extent <= min((1 << (index_bits - 1)) - 1, _SIGNED_I64[1]), "range extent exceeds selected width")


def prove_literal_arange_source(capture: Path, *, index_bits: int) -> dict[str, Any]:
    """Bind all captured i64 aranges to exact trace nodes and parsed source ordinals."""
    from xdsl.dialects.builtin import TensorType

    _need(type(index_bits) is int and 2 <= index_bits <= 128, "index width is not an explicit valid premise")
    source = capture / "model.mlir"
    trace_path = capture / "frontend-trace.json"
    receipt_path = capture / "capture_receipt.json"
    checked = verify_capture_receipt(source)
    _need(checked.get("status") == "verified_materialized", "capture artifact receipt is not verified")
    raw, trace_bytes = source.read_bytes(), trace_path.read_bytes()
    raw_sha, trace_sha = sha256(raw).hexdigest(), sha256(trace_bytes).hexdigest()
    receipt = json.loads(receipt_path.read_bytes())
    _need(
        receipt.get("artifacts", {}).get("model.mlir", {}).get("sha256") == raw_sha
        and receipt.get("artifacts", {}).get("frontend-trace.json", {}).get("sha256") == trace_sha,
        "source or trace changed after receipt verification",
    )
    try:
        normalized, normalization = normalize_capture_mlir(raw.decode("utf-8"))
    except (UnicodeError, CaptureNormalizationError) as exc:
        raise ValueError("literal arange source refusal: raw module is invalid") from exc
    normalized_sha = sha256(normalized.encode()).hexdigest()
    _need(
        normalization.get("input_sha256") == raw_sha and normalization.get("output_sha256") == normalized_sha,
        "source normalization is not bound to raw bytes",
    )
    try:
        module = preparation.parse(source, normalized, raw=raw)
        module.verify()
    except Exception as exc:
        raise ValueError("literal arange source refusal: normalized module is invalid") from exc
    parsed = tuple(mq.walk(module))
    trace = json.loads(trace_bytes)
    mlir, graphs = trace.get("mlir"), trace.get("graphs")
    _need(
        trace.get("schema") == "m2m.frontend_trace.v1"
        and trace.get("status") in {"complete", "diagnostic"}
        and isinstance(mlir, dict)
        and mlir.get("sha256") == raw_sha
        and mlir.get("bytes") == len(raw)
        and isinstance(graphs, dict)
        and isinstance(graphs.get("prepared"), dict)
        and graphs["prepared"].get("status") == "complete"
        and isinstance(graphs.get("original"), dict)
        and graphs["original"].get("status") == "complete",
        "frontend trace is not a complete matching captured graph",
    )
    operations, joins = mlir.get("operations"), mlir.get("source_correspondence")
    prepared, original = graphs["prepared"].get("nodes"), graphs["original"].get("nodes")
    _need(
        isinstance(operations, list)
        and len(operations) == len(parsed)
        and isinstance(joins, list)
        and isinstance(prepared, list)
        and isinstance(original, list),
        "trace operation or node roster is incomplete",
    )
    prepared_by_id = {node.get("id"): node for node in prepared if isinstance(node, dict)}
    original_by_id = {node.get("id"): node for node in original if isinstance(node, dict)}
    _need(
        len(prepared_by_id) == len(prepared) and len(original_by_id) == len(original),
        "trace has duplicate or malformed nodes",
    )
    occurrences = []
    listed: set[int] = set()
    for node_id, node in prepared_by_id.items():
        results = node.get("results")
        if (
            not isinstance(node.get("target"), str)
            or not node["target"].startswith("aten.arange.")
            or not isinstance(results, list)
            or not results
            or not isinstance(results[0], dict)
            or results[0].get("dtype") != "int64"
        ):
            continue
        _need(node["target"] == "aten.arange.start_step", "i64 range uses another unproved source form")
        _need(isinstance(node_id, str), "range has no prepared node identity")
        start, end, step, extent = _range(node)
        origins = node.get("origin_node_ids")
        _need(isinstance(origins, list), "range has no original source lineage")
        originals = [value for value in origins if isinstance(value, str) and value.startswith("g:original:")]
        _need(len(originals) == 1 and originals[0] in original_by_id, "range has no unique original source")
        original_node = original_by_id[originals[0]]
        _need(
            isinstance(original_node.get("op"), str) and isinstance(original_node.get("target"), str),
            "original ancestry has no operation identity",
        )
        matches = [item for item in joins if isinstance(item, dict) and item.get("node_id") == node_id]
        _need(len(matches) == 1 and matches[0].get("status") == "lowered", "range source join is ambiguous")
        ordinals = matches[0].get("mlir_ordinals")
        _need(
            isinstance(ordinals, list)
            and len(ordinals) == len(_LOWERING)
            and all(type(value) is int and 0 <= value < len(parsed) for value in ordinals)
            and len(set(ordinals)) == len(ordinals)
            and tuple(mq.op_name(parsed[value]) for value in ordinals) == _LOWERING,
            "range does not lower to exactly one closed i64 iota",
        )
        tagged = {i for i, op in enumerate(parsed) if node_id in _ids(op, "prov.source_node_ids")}
        _need(
            tagged == set(ordinals) and not listed.intersection(ordinals),
            "range trace omits, duplicates or escapes a source operation",
        )
        for ordinal in ordinals:
            entry = operations[ordinal]
            op = parsed[ordinal]
            _need(
                isinstance(entry, dict)
                and entry.get("ordinal") == ordinal
                and entry.get("operation") == mq.op_name(op)
                and entry.get("source_node_ids") == [node_id]
                and _ids(op, "prov.source_node_ids") == (node_id,)
                and not any(
                    item.get("node_id") != node_id and ordinal in item.get("mlir_ordinals", [])
                    for item in joins
                    if isinstance(item, dict)
                ),
                "range trace ordinal disagrees with parsed provenance",
            )
        generic_ordinal = ordinals[1]
        _need(
            set(_ids(parsed[generic_ordinal], "prov.origin_node_ids")) == set(origins),
            "range body has different original lineage",
        )
        _need(
            mq.attr_str(parsed[generic_ordinal], "prov.aten") == node["target"],
            "range body has different prepared operation identity",
        )
        _body(parsed[generic_ordinal], start, step, extent, index_bits)
        listed.update(ordinals)
        occurrences.append(
            {
                "source_ordinal": generic_ordinal,
                "source_node_id": node_id,
                "original_node_id": originals[0],
                "original_target": original_node["target"],
                "original_to_prepared_equivalence": "not_proved",
                "extent": extent,
                "literal_sha256": sha256(json.dumps([start, end, step], separators=(",", ":")).encode()).hexdigest(),
                "lowering_ordinals": ordinals,
            }
        )
    actual = {
        i
        for i, op in enumerate(parsed)
        if isinstance(mq.attr_str(op, "prov.aten"), str)
        and mq.attr_str(op, "prov.aten").startswith("aten.arange.")
        and mq.op_name(op) == "linalg.generic"
        and len(op.results) == 1
        and isinstance(op.results[0].type, TensorType)
        and str(op.results[0].type.element_type) == "i64"
    }
    _need(
        {item["source_ordinal"] for item in occurrences} == actual,
        "captured i64 range generic lacks a unique literal source",
    )
    return {
        "status": PENDING,
        "scope": SCOPE,
        "index_bits_premise": index_bits,
        "raw_source_sha256": raw_sha,
        "normalized_source_sha256": normalized_sha,
        "frontend_trace_sha256": trace_sha,
        "frontend_trace_status": trace["status"],
        "source_correspondence_scope": "all captured prepared i64 literal ranges",
        "capture_receipt_sha256": checked["receipt_sha256"],
        "count": len(occurrences),
        "occurrences": occurrences,
    }
