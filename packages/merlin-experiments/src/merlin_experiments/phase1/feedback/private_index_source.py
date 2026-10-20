"""Closed prepared-source observations for two distinct tensor-index forms.

This module proves a Boolean-mask count and a literal-bounded indexed extract.
It does not join those independent prepared nodes, admit host execution, prove
original-to-prepared equivalence, or certify generated-code numerics. The index
width remains a caller premise until a separate selected-build join is made.
"""

from __future__ import annotations

import json
from hashlib import sha256
from math import prod
from pathlib import Path
from typing import Any

from merlin.common import mlir_query as mq
from merlin.common.jsonio import strict_json_equal as _same_json_value
from merlin.frontends.capture_normalization import normalize_capture_mlir
from merlin.frontends.linalg_integer_reductions import recognize_static_integer_reduction
from merlin.frontends.linalg_patterns import (
    _checked_static_linalg_shell,
    recognize_static_projected_pointwise,
)
from merlin.frontends.prepared_index_source_body import PROOF_SCHEMA, PROOF_SCOPE, PROOF_STATUS
from merlin.targetgen.application_inventory import verify_capture_receipt
from merlin_experiments.phase1.feedback import private_control_support as control
from merlin_experiments.phase1.feedback import private_linked_source_preparation as preparation
from merlin_experiments.phase1.feedback.private_literal_arange import _range, prove_literal_arange_source

PENDING = PROOF_STATUS
SCOPE = PROOF_SCOPE


def _need(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(f"index source refusal: {reason}")


def _ids(op: Any) -> tuple[str, ...]:
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    value = op.attributes.get("prov.source_node_ids")
    if not isinstance(value, ArrayAttr) or any(not isinstance(item, StringAttr) for item in value.data):
        return ()
    return tuple(item.data for item in value.data)


def _static_tensor(value: Any, dtype: str, index_bits: int) -> tuple[int, ...]:
    from xdsl.dialects.builtin import NoneAttr, TensorType

    typ = value.type
    _need(isinstance(typ, TensorType) and isinstance(typ.encoding, NoneAttr), "tensor encoding is not default")
    shape = tuple(typ.get_shape())
    limit = (1 << (index_bits - 1)) - 1
    _need(
        bool(shape)
        and str(typ.element_type) == dtype
        and all(type(dim) is int and 0 < dim <= limit for dim in shape)
        and prod(shape) * (8 if dtype == "i64" else 1) <= limit,
        "tensor shape, type or byte span exceeds the selected-index premise",
    )
    return shape


def _reshape_source(value: Any, index_bits: int) -> Any:
    """Peel only verified, static, order-preserving tensor reassociations."""
    from xdsl.dialects import tensor
    from xdsl.dialects.builtin import DenseArrayBase

    _static_tensor(value, "i64", index_bits)
    while type(value.owner) in (tensor.CollapseShapeOp, tensor.ExpandShapeOp):
        op = value.owner
        expected = {"reassociation"}
        if type(op) is tensor.ExpandShapeOp:
            expected.add("static_output_shape")
            static = op.properties.get("static_output_shape")
            _need(
                isinstance(static, DenseArrayBase) and tuple(static.iter_values()) == tuple(value.type.get_shape()),
                "expanded index tensor has a different static output shape",
            )
        _need(
            set(op.properties) == expected
            and all(name.startswith("prov.") for name in op.attributes)
            and not op.regions
            and not op.successors
            and len(op.operands) == len(op.results) == 1
            and op.results[0] is value,
            "index reshape has dynamic operands or unknown semantics",
        )
        source = op.operands[0]
        _static_tensor(source, "i64", index_bits)
        _need(prod(source.type.get_shape()) == prod(value.type.get_shape()), "index reshape changes element count")
        try:
            op.verify()
        except Exception as exc:  # noqa: BLE001 - malformed source must have one refusing API
            raise ValueError("index source refusal: index reassociation is invalid") from exc
        value = source
    return value


def _scalar_splat(value: Any, index_bits: int) -> int:
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import IntegerAttr

    _static_tensor(value, "i64", index_bits)
    splat = value.owner
    _need(
        type(splat) is tensor.SplatOp
        and len(splat.operands) == len(splat.results) == 1
        and splat.results[0] is value
        and not splat.properties
        and not splat.regions
        and not splat.successors
        and all(name.startswith("prov.") for name in splat.attributes),
        "index offset is not one closed scalar splat",
    )
    constant = splat.operands[0].owner
    _need(
        type(constant) is arith.ConstantOp
        and not constant.operands
        and len(constant.results) == 1
        and constant.results[0] is splat.operands[0]
        and set(constant.properties) == {"value"}
        and not constant.regions
        and not constant.successors
        and all(name.startswith("prov.") for name in constant.attributes),
        "index offset is not one closed integer constant",
    )
    attr = constant.properties["value"]
    _need(
        isinstance(attr, IntegerAttr)
        and attr.type == constant.results[0].type
        and str(attr.type) == "i64"
        and type(attr.value.data) is int
        and -(1 << 63) <= attr.value.data < (1 << 63),
        "index offset has a different scalar type",
    )
    return attr.value.data


def _index_interval(
    value: Any,
    *,
    index_bits: int,
    ranges: dict[int, tuple[int, int, int, int]],
    range_literals: dict[int, str],
    ordinals: dict[int, int],
) -> tuple[int, int, int, str, tuple[int, ...], str | None]:
    """Return inclusive extrema, arange ordinal, and optional add ordinal."""
    from xdsl.dialects.linalg.ops import GenericOp

    value = _reshape_source(value, index_bits)
    owner = value.owner
    _need(type(owner) is GenericOp and owner.results[0] is value, "index has no closed range source")
    offset = 0
    additions: tuple[int, ...] = ()
    offset_sha: str | None = None
    if id(owner) not in ranges:
        pattern = recognize_static_projected_pointwise(owner)
        _need(
            pattern.operation == "arith.addi"
            and len(owner.inputs) == 2
            and pattern.ordered_types == ("i64", "i64", "i64", "i64"),
            "index offset is not unflagged signed-i64 elementwise addition",
        )
        candidates = [(i, inp) for i, inp in enumerate(owner.inputs) if id(inp.owner) in ranges]
        _need(len(candidates) == 1, "index addition does not have one literal range")
        i, base = candidates[0]
        offset = _scalar_splat(owner.inputs[1 - i], index_bits)
        offset_sha = sha256(json.dumps(offset).encode()).hexdigest()
        _need(tuple(base.type.get_shape()) == tuple(value.type.get_shape()), "offset changes range layout")
        additions = (ordinals[id(owner)],)
        owner = base.owner
    _need(id(owner) in ranges, "index source range is not trace-verified")
    start, _, step, extent = ranges[id(owner)]
    _need(extent > 0, "indexed extract has an empty range")
    last = start + (extent - 1) * step
    return (
        min(start, last) + offset,
        max(start, last) + offset,
        ordinals[id(owner)],
        range_literals[id(owner)],
        additions,
        offset_sha,
    )


def _indexed_extract(
    op: Any,
    *,
    index_bits: int,
    ranges: dict[int, tuple[int, int, int, int]],
    range_literals: dict[int, str],
    ordinals: dict[int, int],
) -> dict[str, Any]:
    """Prove that every static output cell reads only an in-bounds source cell."""
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.linalg.ops import YieldOp

    shell = _checked_static_linalg_shell(op, singleton_projection_inputs=True)
    _need(len(op.inputs) > 0 and len(shell.body) == len(op.inputs) + 2, "indexed extract body is not closed")
    *casts, extract, yielded = shell.body
    _need(type(extract) is tensor.ExtractOp and type(yielded) is YieldOp, "indexed body is not tensor.extract/yield")
    source = extract.operands[0]
    _need(
        source is not op.outputs[0] and type(source.owner) is not tensor.EmptyOp,
        "indexed extract reads uninitialized storage",
    )
    _need(shell.ordered_types[-1] == "i1", "indexed extract source/result type is not proved")
    source_shape = _static_tensor(source, shell.ordered_types[-1], index_bits)
    _need(
        _static_tensor(op.results[0], "i1", index_bits) == shell.shape,
        "indexed output extent or byte span exceeds the selected-index premise",
    )
    _need(len(op.inputs) == len(source_shape), "extract index arity differs from source rank")
    _need(
        not extract.properties
        and not extract.regions
        and not extract.successors
        and all(name.startswith("prov.") for name in extract.attributes)
        and tuple(extract.operands[1:]) == tuple(cast.results[0] for cast in casts)
        and len(extract.results) == 1
        and str(extract.results[0].type) == shell.ordered_types[-1],
        "extract source, index SSA or scalar type differs",
    )
    _need(
        tuple(yielded.operands) == tuple(extract.results)
        and not yielded.results
        and not yielded.properties
        and not yielded.regions
        and not yielded.successors
        and all(name.startswith("prov.") for name in yielded.attributes),
        "indexed body yields another value",
    )
    _need(shell.args[-1] not in tuple(extract.operands), "indexed body reads output initialization")
    limit = (1 << (index_bits - 1)) - 1
    index_sources = []
    for axis, (cast, argument, index_tensor, source_extent) in enumerate(
        zip(casts, shell.args[:-1], op.inputs, source_shape, strict=True)
    ):
        _need(
            type(cast) is arith.IndexCastOp
            and not cast.properties
            and not cast.regions
            and not cast.successors
            and all(name.startswith("prov.") for name in cast.attributes)
            and tuple(cast.operands) == (argument,)
            and len(cast.results) == 1
            and str(cast.results[0].type) == "index",
            "extract coordinate is not a closed i64-to-index cast",
        )
        low, high, range_ordinal, literal_sha, addition_ordinals, offset_sha = _index_interval(
            index_tensor,
            index_bits=index_bits,
            ranges=ranges,
            range_literals=range_literals,
            ordinals=ordinals,
        )
        _need(0 <= low <= high < source_extent and high <= limit, "literal index could be out of bounds")
        index_sources.append(
            {
                "axis": axis,
                "range_ordinal": range_ordinal,
                "range_literal_sha256": literal_sha,
                "addition_ordinals": list(addition_ordinals),
                "offset_literal_sha256": offset_sha,
            }
        )
    return {
        "kind": "literal_indexed_extract",
        "component_ordinals": sorted(ordinals[id(inner)] for inner in (op, *shell.body)),
        "source_shape": list(source_shape),
        "output_shape": list(shell.shape),
        "index_input_shapes": [list(value.type.get_shape()) for value in op.inputs],
        "input_maps": [str(mapping) for mapping in shell.input_maps],
        "index_sources": index_sources,
    }


def _mask_count(op: Any, node_ordinals: set[int], ordinals: dict[int, int], index_bits: int) -> dict[str, Any]:
    """Tie the exact i1 extension to its zero-seeded count, not other index nodes."""
    from xdsl.dialects import tensor

    _need(mq.op_name(op) == "linalg.reduce", "mask count is not a typed reduction")
    reduction = recognize_static_integer_reduction(op, index_bits=index_bits)
    _need(
        reduction.operation == "sum" and reduction.input_type == "i1",
        "mask index does not sum a proved unsigned Boolean extension",
    )
    input_generic = op.operands[0].owner
    _need(
        mq.op_name(input_generic) == "linalg.generic" and ordinals[id(input_generic)] in node_ordinals,
        "mask extension belongs to another prepared node",
    )
    uses = [use.operation for use in op.results[0].uses]
    _need(len(uses) == 1 and type(uses[0]) is tensor.ExtractOp, "mask count escapes its scalar extraction")
    scalar = uses[0]
    _need(ordinals[id(scalar)] in node_ordinals, "scalar count belongs to another prepared node")
    extent, _ = control._mask_count(scalar.results[0])
    _need(extent <= (1 << (index_bits - 1)) - 1, "mask count exceeds selected index width")
    return {
        "kind": "boolean_mask_count",
        "component_ordinals": sorted(
            ordinals[id(inner)]
            for inner in (
                input_generic,
                *input_generic.regions[0].block.ops,
                op,
                *op.regions[0].block.ops,
                scalar,
            )
        ),
        "extension_ordinal": ordinals[id(input_generic)],
        "reduction_ordinal": ordinals[id(op)],
        "scalar_ordinal": ordinals[id(scalar)],
        "count_interval": [0, extent],
        "compaction_and_index_data": "separate_control_source_obligation",
    }


def prove_index_source(capture: Path, *, index_bits: int) -> dict[str, Any]:
    """Bind every prepared tensor-index compute root to one of two closed forms."""
    from xdsl.dialects.linalg.ops import GenericOp, ReduceOp

    _need(type(index_bits) is int and 2 <= index_bits <= 128, "index width is not an explicit valid premise")
    capture = Path(capture)
    source, trace_path = capture / "model.mlir", capture / "frontend-trace.json"
    receipt = verify_capture_receipt(source)
    _need(receipt.get("status") == "verified_materialized", "capture receipt is not verified")
    raw, trace_bytes = source.read_bytes(), trace_path.read_bytes()
    raw_sha, trace_sha = sha256(raw).hexdigest(), sha256(trace_bytes).hexdigest()
    receipt_bytes = (capture / "capture_receipt.json").read_bytes()
    _need(sha256(receipt_bytes).hexdigest() == receipt["receipt_sha256"], "capture receipt changed during proof")
    document = json.loads(receipt_bytes)
    _need(
        document.get("artifacts", {}).get("model.mlir", {}).get("sha256") == raw_sha
        and document.get("artifacts", {}).get("frontend-trace.json", {}).get("sha256") == trace_sha,
        "source or trace is not bound to the receipt",
    )
    normalized, normalization = normalize_capture_mlir(raw.decode())
    normalized_sha = sha256(normalized.encode()).hexdigest()
    _need(
        normalization.get("input_sha256") == raw_sha and normalization.get("output_sha256") == normalized_sha,
        "source normalization is not bound to raw bytes",
    )
    module = preparation.parse(source, normalized, raw=raw)
    try:
        module.verify()
    except Exception as exc:  # noqa: BLE001 - malformed source must refuse
        raise ValueError("index source refusal: normalized module is invalid") from exc
    parsed = tuple(mq.walk(module))
    ordinals = {id(op): i for i, op in enumerate(parsed)}
    try:
        trace = json.loads(trace_bytes)
    except (UnicodeError, ValueError) as exc:
        raise ValueError("index source refusal: frontend trace is unreadable") from exc
    _need(isinstance(trace, dict), "frontend trace is malformed")
    mlir, graphs = trace.get("mlir"), trace.get("graphs")
    prepared = graphs.get("prepared") if isinstance(graphs, dict) else None
    nodes = prepared.get("nodes") if isinstance(prepared, dict) else None
    _need(
        trace.get("schema") == "m2m.frontend_trace.v1"
        and isinstance(prepared, dict)
        and trace.get("status") == prepared.get("status") == "complete"
        and isinstance(mlir, dict)
        and mlir.get("sha256") == raw_sha
        and mlir.get("bytes") == len(raw)
        and isinstance(nodes, list),
        "prepared trace is incomplete or selects another source",
    )
    entries, joins = mlir.get("operations"), mlir.get("source_correspondence")
    _need(
        isinstance(entries, list) and len(entries) == len(parsed) and isinstance(joins, list), "trace op roster differs"
    )
    prepared_by_id = {node.get("id"): node for node in nodes if isinstance(node, dict)}
    _need(len(prepared_by_id) == len(nodes), "prepared trace node identities are incomplete")
    index_nodes = {key: item for key, item in prepared_by_id.items() if item.get("target") == "aten.index.Tensor"}

    literal = prove_literal_arange_source(capture, index_bits=index_bits)
    _need(
        literal.get("raw_source_sha256") == raw_sha and literal.get("normalized_source_sha256") == normalized_sha,
        "literal range proof names another captured source",
    )
    _need(
        literal.get("frontend_trace_sha256") == trace_sha
        and literal.get("capture_receipt_sha256") == receipt["receipt_sha256"],
        "literal range proof names another trace or receipt",
    )
    ranges: dict[int, tuple[int, int, int, int]] = {}
    range_literals: dict[int, str] = {}
    for occurrence in literal["occurrences"]:
        node = prepared_by_id.get(occurrence["source_node_id"])
        _need(isinstance(node, dict), "literal range has no prepared trace node")
        root = id(parsed[occurrence["source_ordinal"]])
        ranges[root] = _range(node)
        range_literals[root] = occurrence["literal_sha256"]

    records = []
    witnessed: set[int] = set()
    for node_id, node in index_nodes.items():
        _need(isinstance(node_id, str) and node_id, "index node has no identity")
        found = [join for join in joins if isinstance(join, dict) and join.get("node_id") == node_id]
        _need(len(found) == 1 and found[0].get("status") == "lowered", "index node source join is ambiguous")
        roster = found[0].get("mlir_ordinals")
        _need(
            isinstance(roster, list)
            and bool(roster)
            and all(type(i) is int and 0 <= i < len(parsed) for i in roster)
            and len(roster) == len(set(roster)),
            "index node ordinal roster is malformed",
        )
        owned = set(roster)
        _need(
            owned == {i for i, op in enumerate(parsed) if node_id in _ids(op)} and not witnessed.intersection(owned),
            "index node source attributes omit, duplicate or alias an operation",
        )
        _need(
            not any(
                other.get("node_id") != node_id
                and isinstance(other.get("mlir_ordinals"), list)
                and owned.intersection(other["mlir_ordinals"])
                for other in joins
                if isinstance(other, dict)
            ),
            "index source ordinal is claimed by another prepared node",
        )
        for ordinal in owned:
            _need(
                _ids(parsed[ordinal]) == (node_id,)
                and isinstance(entries[ordinal], dict)
                and entries[ordinal].get("ordinal") == ordinal
                and entries[ordinal].get("operation") == mq.op_name(parsed[ordinal])
                and entries[ordinal].get("source_node_ids") == [node_id],
                "index trace operation differs from parsed provenance",
            )
        roots = [
            (i, parsed[i])
            for i in roster
            if isinstance(parsed[i], (GenericOp, ReduceOp)) and mq.attr_str(parsed[i], "prov.aten") == node["target"]
        ]
        if len(roots) == 2 and {mq.op_name(op) for _, op in roots} == {"linalg.generic", "linalg.reduce"}:
            reduction = next(op for _, op in roots if mq.op_name(op) == "linalg.reduce")
            fact = _mask_count(reduction, owned, ordinals, index_bits)
            selected = {fact["extension_ordinal"], fact["reduction_ordinal"]}
        elif len(roots) == 1 and mq.op_name(roots[0][1]) == "linalg.generic":
            fact = _indexed_extract(
                roots[0][1],
                index_bits=index_bits,
                ranges=ranges,
                range_literals=range_literals,
                ordinals=ordinals,
            )
            selected = {roots[0][0]}
        else:
            raise ValueError("index source refusal: prepared node has an unproved compute root")
        _need(
            set(fact["component_ordinals"]) <= owned,
            "index body component belongs to another prepared node",
        )
        witnessed.update(owned)
        records.append({"source_node_id": node_id, "compute_ordinals": sorted(selected), **fact})
    actual = {
        i
        for i, op in enumerate(parsed)
        if mq.attr_str(op, "prov.aten") == "aten.index.Tensor" and mq.op_name(op) in {"linalg.generic", "linalg.reduce"}
    }
    _need(
        {ordinal for record in records for ordinal in record["compute_ordinals"]} == actual,
        "tensor-index compute roots are incomplete or duplicated",
    )
    _need(
        sha256(source.read_bytes()).hexdigest() == raw_sha
        and sha256(trace_path.read_bytes()).hexdigest() == trace_sha
        and sha256((capture / "capture_receipt.json").read_bytes()).hexdigest() == receipt["receipt_sha256"],
        "capture source, trace or receipt changed during proof",
    )
    return {
        "schema": PROOF_SCHEMA,
        "status": PENDING,
        "scope": SCOPE,
        "index_bits_premise": index_bits,
        "raw_source_sha256": raw_sha,
        "normalized_source_sha256": normalized_sha,
        "frontend_trace_sha256": trace_sha,
        "capture_receipt_sha256": receipt["receipt_sha256"],
        "n_source_operations": len(parsed),
        "original_to_prepared_equivalence": "not_proved",
        "records": records,
    }


def verify_index_source_record(capture: Path, record: dict[str, Any]) -> None:
    """Recompute every serialized source, trace, literal and map fact exactly."""
    _need(isinstance(record, dict), "serialized index source proof is malformed")
    index_bits = record.get("index_bits_premise")
    _need(type(index_bits) is int, "serialized index width is malformed")
    _need(
        _same_json_value(record, prove_index_source(capture, index_bits=index_bits)),
        "serialized index source proof changed",
    )
