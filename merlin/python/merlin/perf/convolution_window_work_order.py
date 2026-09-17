"""Seal convolution-window elimination sites from one complete static portfolio iteration.

The global performance analysis already binds the source graph, global plan, emitted artifact,
compiler, and exact source-operation ownership.  This module consumes only that immutable record.
It does not reparse a model, ask a target what it supports, or infer a convolution from a capsule
name.  Two independently visible opportunity classes are admitted:

* a semantic convolution task whose emitted body still gathers one affine window row; and
* an explicit source im2col tensor that feeds a proved contraction through a bound dependency path.

Convolutions which already consume the source activation directly or use a native window mechanism
remain visible, but are not authoring sites.  Incomplete geometry, buffer flow, task ownership,
global-plan identity, or representation evidence fails closed.  The policy contains no workload
names, target names, tile constants, simulator identities, or timing claims.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from argparse import ArgumentParser
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.benchharness import hash_tree
from merlin.perf.compiler_edit_scope import validate_mechanism_catalog
from merlin.perf.rank_general_contraction_work_order import (
    _PIN_FIELDS,
    _analysis_parts,
    _digest,
    _embedded_edit_contract,
    _integer_dtype,
    _ordered_analyses,
    _pin,
)

MECHANISM_ID = "t01_03_convolution_window_materialization_elimination"
INVENTORY_SCHEMA = "portfolio_convolution_window_inventory_v1"

_MAC_PROOF = "one proved yielded MAC per static affine-domain point"
_CONVOLUTION_TASK_KIND = "convolution"
_CONTRACTION_TASK_KIND = "contraction"
_HOST_TASK_KIND = "host"

# These names describe compiler semantic surfaces, not a target or a workload.  Together they own
# source recognition/partitioning, the semantic task, capability-selected issue, and the currently
# emitted affine gather.  Omitting any one would hand the author a non-executable partial lever.
_REQUIRED_SURFACES = frozenset(
    {
        "source_convolution_global_partition",
        "source_convolution_mesh_eligibility",
        "source_convolution_semantic_task",
        "source_convolution_capability_selected_schedule",
        "affine_window_packing_specialization",
    }
)
_MECHANISM_SURFACES = _REQUIRED_SURFACES | frozenset(
    {
        "convolution_route",
        "global_issue_and_fences",
        "global_partition",
        "pipeline_issue",
    }
)


def _raw_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _shape(buffer: Any, *, label: str, reasons: list[str]) -> list[int] | None:
    if not isinstance(buffer, Mapping):
        reasons.append(f"{label} buffer is absent from the captured graph")
        return None
    value = buffer.get("shape")
    if not isinstance(value, list) or not value or any(type(extent) is not int or extent <= 0 for extent in value):
        reasons.append(f"{label} buffer lacks an exact positive static shape")
        return None
    return list(value)


def _dtype(buffer: Any, *, label: str, reasons: list[str]) -> str | None:
    if not isinstance(buffer, Mapping):
        reasons.append(f"{label} buffer is absent from the captured graph")
        return None
    value = buffer.get("dtype")
    if not isinstance(value, str) or not value:
        reasons.append(f"{label} buffer lacks an exact element dtype")
        return None
    return value


def _task_rows(analysis: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    diagnostics = analysis.get("diagnostics")
    evidence = diagnostics.get("task_instruction_evidence") if isinstance(diagnostics, Mapping) else None
    candidate = evidence.get("candidate") if isinstance(evidence, Mapping) else None
    rows = candidate.get("tasks") if isinstance(candidate, Mapping) else None
    return list(rows) if isinstance(rows, list) else []


def _placement_rows(parts: Mapping[str, Any]) -> tuple[dict[int, Mapping[str, Any]], list[str]]:
    by_source: dict[int, Mapping[str, Any]] = {}
    problems: list[str] = []
    for ordinal, raw in enumerate(parts["placement"].get("contractions", [])):
        if not isinstance(raw, Mapping):
            problems.append(f"contraction placement row {ordinal} is malformed")
            continue
        source_index = raw.get("source_op_index")
        if type(source_index) is not int or not 0 <= source_index < len(parts["nodes"]):
            problems.append(f"contraction placement row {ordinal} has no exact source operation")
        elif source_index in by_source:
            problems.append(f"source operation {source_index} has multiple contraction observations")
        else:
            by_source[source_index] = raw
    return by_source, problems


def _convolution_routes(
    analysis: Mapping[str, Any],
    task_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, Mapping[str, Any]], list[str]]:
    """Bind ordered semantic convolution tasks to ordered representation directives.

    The complete plan's task order and the command buffer's representation-directive order are both
    deterministic.  A mixed or incomplete denominator is refused rather than partially zipped.
    """
    problems: list[str] = []
    convolution_tasks = [row for row in task_rows if row.get("declared_task_kind") == _CONVOLUTION_TASK_KIND]
    diagnostics = analysis.get("diagnostics")
    arms = diagnostics.get("arms") if isinstance(diagnostics, Mapping) else None
    candidate = arms.get("candidate") if isinstance(arms, Mapping) else None
    activity = candidate.get("representation_activity") if isinstance(candidate, Mapping) else None
    if not isinstance(activity, Mapping) or activity.get("schema") != "command_buffer_representation_activity_v1":
        if convolution_tasks:
            problems.append("semantic convolution tasks lack representation activity evidence")
        return {}, problems
    directives = activity.get("representation_directives")
    if not isinstance(directives, list):
        if convolution_tasks:
            problems.append("semantic convolution tasks lack ordered representation directives")
        return {}, problems

    routes: list[Mapping[str, Any]] = []
    seen_indices: set[int] = set()
    for raw in directives:
        if not isinstance(raw, Mapping):
            problems.append("representation directive row is malformed")
            continue
        attributes, operands = raw.get("attributes"), raw.get("operands")
        # This is the target-dialect's semantic whole-convolution operation, not an ISA opcode.
        if raw.get("opcode") != "CONV2D":
            continue
        index = raw.get("index")
        if type(index) is not int or index < 0 or index in seen_indices:
            problems.append("convolution representation directive has no unique command index")
            continue
        seen_indices.add(index)
        if (
            not isinstance(attributes, Mapping)
            or not isinstance(operands, Mapping)
            or not isinstance(attributes.get("layout"), str)
            or not attributes["layout"]
            or set(operands) != {"ifm", "weight", "dst"}
            or any(not isinstance(value, str) or not value for value in operands.values())
        ):
            problems.append("convolution representation directive lacks exact layout or operands")
            continue
        routes.append(raw)
    if len(routes) != len(convolution_tasks):
        problems.append("semantic convolution task and representation-directive counts disagree")
        return {}, problems
    result: dict[int, Mapping[str, Any]] = {}
    for task, route in zip(convolution_tasks, routes, strict=True):
        task_index = task.get("task_index")
        if type(task_index) is not int or task_index < 0 or task_index in result:
            problems.append("semantic convolution task has no unique task index")
        else:
            result[task_index] = route
    return result, problems


def _mac_geometry(
    row: Mapping[str, Any],
    *,
    parallel_rank: int,
    reduction_rank: int,
    reasons: list[str],
) -> tuple[list[int] | None, list[int] | None]:
    parallel, reduction = row.get("parallel"), row.get("reduction")
    if (
        not isinstance(parallel, list)
        or len(parallel) != parallel_rank
        or not isinstance(reduction, list)
        or len(reduction) != reduction_rank
        or any(type(value) is not int or value <= 0 for value in [*parallel, *reduction])
    ):
        reasons.append("convolution geometry lacks exact positive affine-domain extents")
        return None, None
    if row.get("mac_status") != "derived" or row.get("mac_basis") != _MAC_PROOF:
        reasons.append("convolution geometry lacks a proved static MAC recurrence")
    product = math.prod([*parallel, *reduction])
    if type(row.get("macs")) is not int or row.get("macs") != product:
        reasons.append("convolution geometry and declared static MAC domain disagree")
    return list(parallel), list(reduction)


def _binding(parts: Mapping[str, Any]) -> dict[str, Any]:
    return {field: parts["binding"][field] for field in _PIN_FIELDS}


def _ownership(task: Mapping[str, Any], parts: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "proved",
        "task_index": task["task_index"],
        "declared_task_kind": task["declared_task_kind"],
        "task_binding_sha256": _digest(_binding(parts)),
        "owned_instruction_payload_sha256": task.get("owned_instruction_payload_sha256"),
    }


def _classify_semantic_task(
    source_index: int,
    row: Mapping[str, Any],
    parts: Mapping[str, Any],
    route: Mapping[str, Any] | None,
) -> tuple[str, dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    parallel, reduction = _mac_geometry(row, parallel_rank=4, reduction_rank=3, reasons=reasons)
    node = parts["nodes"][source_index]
    if not isinstance(node, Mapping) or node.get("kind") != "dispatch":
        reasons.append("source convolution is not an exact captured dispatch")
        node = {}
    provenance = node.get("prov") if isinstance(node.get("prov"), Mapping) else {}
    if (
        provenance.get("prov.conv_path") != "direct_contraction"
        or provenance.get("prov.role") != "contraction"
        or provenance.get("prov.region_id") != row.get("region")
    ):
        reasons.append("source convolution and contraction observer identities disagree")
    inputs, outputs = node.get("inputs"), node.get("outputs")
    if (
        not isinstance(inputs, list)
        or len(inputs) != 2
        or not isinstance(outputs, list)
        or len(outputs) != 1
        or any(not isinstance(value, str) or not value for value in [*inputs, *outputs])
    ):
        reasons.append("source convolution lacks exact activation/weight/result edges")
        inputs, outputs = [], []
    buffers = parts["buffers"]
    activation = buffers.get(inputs[0]) if inputs else None
    weight = buffers.get(inputs[1]) if len(inputs) > 1 else None
    output = buffers.get(outputs[0]) if outputs else None
    if inputs and outputs:
        for name, buffer, label in zip(
            [*inputs, *outputs], [activation, weight, output], ["activation", "weight", "result"], strict=True
        ):
            if isinstance(buffer, Mapping) and buffer.get("id") != name:
                reasons.append(f"{label} buffer identity disagrees with its graph edge")
    activation_shape = _shape(activation, label="activation", reasons=reasons)
    weight_shape = _shape(weight, label="weight", reasons=reasons)
    output_shape = _shape(output, label="result", reasons=reasons)
    dtypes = [
        _dtype(activation, label="activation", reasons=reasons),
        _dtype(weight, label="weight", reasons=reasons),
        _dtype(output, label="result", reasons=reasons),
    ]
    if any(dtype is not None and not _integer_dtype(dtype) for dtype in dtypes):
        reasons.append("source convolution is not an exact integer contraction")
    task = parts["owners"].get(source_index)
    if not isinstance(task, Mapping) or task.get("declared_task_kind") != _CONVOLUTION_TASK_KIND:
        reasons.append("source convolution has no unique semantic convolution-task owner")
        task = {}
    elif source_index not in task.get("source_op_indices", []):
        reasons.append("semantic convolution task does not own the source operation")
    task_index = task.get("task_index")
    if route is None or task_index is None:
        reasons.append("semantic convolution task lacks its exact representation directive")
        route = {}
    attributes = route.get("attributes") if isinstance(route, Mapping) else None
    operands = route.get("operands") if isinstance(route, Mapping) else None
    layout = attributes.get("layout") if isinstance(attributes, Mapping) else None
    if not isinstance(layout, str) or not layout:
        reasons.append("semantic convolution representation has no exact layout route")
    if (
        parallel is not None
        and reduction is not None
        and activation_shape is not None
        and weight_shape is not None
        and output_shape is not None
    ):
        batch, co, ho, wo = parallel
        ci, kh, kw = reduction
        if (
            len(activation_shape) != 4
            or len(weight_shape) != 4
            or len(output_shape) != 4
            or weight_shape != [co, ci, kh, kw]
            or output_shape != [batch, co, ho, wo]
            or activation_shape[:2] != [batch, ci]
        ):
            reasons.append("source buffers disagree with the proved convolution domain")
    if isinstance(operands, Mapping) and inputs and outputs:
        # Plan-local tensor names differ from graph-local buffer IDs.  Only the exact operand roles
        # are asserted here; source-to-plan value binding is already proved by task ownership.
        if set(operands) != {"ifm", "weight", "dst"}:
            reasons.append("representation directive lacks activation/weight/result roles")

    counts = task.get("static_operation_counts") if isinstance(task, Mapping) else None
    if not isinstance(counts, Mapping) or any(
        not isinstance(key, str) or type(value) is not int or value < 0
        for key, value in (counts.items() if isinstance(counts, Mapping) else [])
    ):
        reasons.append("semantic convolution task lacks exact static operation counts")
        counts = {}
    if reasons:
        return "refused", None, sorted(set(reasons))

    batch, co, ho, wo = parallel
    ci, kh, kw = reduction
    row_instances = batch * ho
    loads = counts.get("llvm.load", 0)
    stores = counts.get("llvm.store", 0)
    divisions = counts.get("llvm.udiv", 0)
    remainders = counts.get("llvm.urem", 0)
    route_lower = layout.lower()
    if "im2col" in route_lower or "stream" in route_lower:
        if loads == stores == row_instances and divisions > 0 and remainders > 0:
            classification = "streamed"
        elif loads == stores == divisions == remainders == 0 and kh == kw == 1 and activation_shape[2:] == [ho, wo]:
            classification = "already_direct"
        else:
            return "refused", None, ["declared streamed/direct route disagrees with exact window-work evidence"]
    elif loads == stores == divisions == remainders == 0:
        classification = "already_native"
    else:
        return "refused", None, ["native convolution route still contains unclassified scalar window work"]

    site = {
        "classification": classification,
        "eligible": classification == "streamed",
        "source_operation_id": source_index,
        "source_operation_ids": [source_index],
        "source_region_id": row["region"],
        "source_operation": row.get("op"),
        "geometry": {
            "batch": batch,
            "input_channels": ci,
            "input_height": activation_shape[2],
            "input_width": activation_shape[3],
            "output_channels": co,
            "kernel_height": kh,
            "kernel_width": kw,
            "output_height": ho,
            "output_width": wo,
            "window_row_instances": row_instances,
        },
        "buffers": {
            "activation": {"id": inputs[0], "shape": activation_shape, "dtype": dtypes[0]},
            "weight": {"id": inputs[1], "shape": weight_shape, "dtype": dtypes[1]},
            "result": {"id": outputs[0], "shape": output_shape, "dtype": dtypes[2]},
        },
        "plan_route": {
            "semantic_task_kind": _CONVOLUTION_TASK_KIND,
            "representation_command_index": route["index"],
            "layout": layout,
        },
        "window_work_evidence": {
            "static_load_sites": loads,
            "static_store_sites": stores,
            "static_unsigned_divide_sites": divisions,
            "static_unsigned_remainder_sites": remainders,
        },
        "ownership_evidence": _ownership(task, parts),
        "proof_scope": "static source geometry, buffers, plan route, and task ownership",
        "not_proven": ["replacement semantic equivalence", "runtime correctness", "cycle improvement"],
    }
    site["site_binding_sha256"] = _digest(site)
    return classification, site, []


def _producer_and_consumers(nodes: Sequence[Any]) -> tuple[dict[str, int], dict[str, list[int]]]:
    producers: dict[str, int] = {}
    consumers: dict[str, list[int]] = {}
    for index, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            continue
        for output in node.get("outputs", []):
            if isinstance(output, str) and output not in producers:
                producers[output] = index
        for source in node.get("inputs", []):
            if isinstance(source, str):
                consumers.setdefault(source, []).append(index)
    return producers, consumers


def _dependency_path(
    nodes: Sequence[Any],
    start_buffer: str,
    sink_index: int,
) -> list[int] | None:
    """Return one deterministic source-operation path from a buffer to the sink."""
    _producers, consumers = _producer_and_consumers(nodes)
    queue: deque[tuple[str, list[int]]] = deque([(start_buffer, [])])
    seen_buffers = {start_buffer}
    while queue:
        buffer, path = queue.popleft()
        for node_index in sorted(consumers.get(buffer, [])):
            if node_index == sink_index:
                return [*path, node_index]
            node = nodes[node_index]
            if not isinstance(node, Mapping):
                continue
            for output in node.get("outputs", []):
                if isinstance(output, str) and output not in seen_buffers:
                    seen_buffers.add(output)
                    queue.append((output, [*path, node_index]))
    return None


def _classify_materialized_region(
    region: str,
    region_nodes: Sequence[int],
    sink_index: int,
    placement: Mapping[str, Any],
    parts: Mapping[str, Any],
) -> tuple[str, dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    parallel, reduction = _mac_geometry(placement, parallel_rank=2, reduction_rank=1, reasons=reasons)
    nodes, buffers = parts["nodes"], parts["buffers"]
    sink = nodes[sink_index]
    sink_inputs = sink.get("inputs") if isinstance(sink, Mapping) else None
    sink_outputs = sink.get("outputs") if isinstance(sink, Mapping) else None
    if (
        not isinstance(sink_inputs, list)
        or len(sink_inputs) != 2
        or not isinstance(sink_outputs, list)
        or len(sink_outputs) != 1
    ):
        reasons.append("materialized convolution contraction lacks exact operand/result edges")
        sink_inputs, sink_outputs = [], []
    sink_shapes = [
        _shape(buffers.get(name), label=f"contraction operand {index}", reasons=reasons)
        for index, name in enumerate(sink_inputs)
    ]
    sink_output_shape = _shape(
        buffers.get(sink_outputs[0]) if sink_outputs else None, label="contraction result", reasons=reasons
    )
    sink_dtypes = [
        _dtype(buffers.get(name), label=f"contraction operand {index}", reasons=reasons)
        for index, name in enumerate(sink_inputs)
    ]
    sink_output_dtype = _dtype(
        buffers.get(sink_outputs[0]) if sink_outputs else None, label="contraction result", reasons=reasons
    )
    for name in [*sink_inputs, *sink_outputs]:
        buffer = buffers.get(name)
        if isinstance(buffer, Mapping) and buffer.get("id") != name:
            reasons.append("contraction buffer identity disagrees with its graph edge")
    sink_task = parts["owners"].get(sink_index)
    if not isinstance(sink_task, Mapping) or sink_task.get("declared_task_kind") != _CONTRACTION_TASK_KIND:
        reasons.append("materialized convolution has no proved contraction-task sink")
        sink_task = {}

    materializers: list[tuple[int, str, list[int], list[int], str, str]] = []
    for index in region_nodes:
        if index == sink_index:
            continue
        node = nodes[index]
        provenance = node.get("prov") if isinstance(node, Mapping) else None
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("prov.role") is not None
            or provenance.get("prov.conv_path") != "im2col_matmul"
        ):
            continue
        inputs, outputs = node.get("inputs"), node.get("outputs")
        local_reasons: list[str] = []
        if not isinstance(inputs, list) or len(inputs) != 1 or not isinstance(outputs, list) or len(outputs) != 1:
            continue
        source_shape = _shape(buffers.get(inputs[0]), label="window source", reasons=local_reasons)
        window_shape = _shape(buffers.get(outputs[0]), label="window tensor", reasons=local_reasons)
        source_dtype = _dtype(buffers.get(inputs[0]), label="window source", reasons=local_reasons)
        window_dtype = _dtype(buffers.get(outputs[0]), label="window tensor", reasons=local_reasons)
        if isinstance(buffers.get(inputs[0]), Mapping) and buffers[inputs[0]].get("id") != inputs[0]:
            local_reasons.append("window source buffer identity disagrees with its graph edge")
        if isinstance(buffers.get(outputs[0]), Mapping) and buffers[outputs[0]].get("id") != outputs[0]:
            local_reasons.append("window tensor buffer identity disagrees with its graph edge")
        if (
            not local_reasons
            and source_shape is not None
            and window_shape is not None
            and source_dtype is not None
            and window_dtype is not None
            and len(source_shape) == 4
            and len(window_shape) == 6
        ):
            materializers.append((index, outputs[0], source_shape, window_shape, source_dtype, window_dtype))
    if len(materializers) != 1:
        reasons.append("convolution region does not have one exact rank-4 to rank-6 window materializer")
        materializer = None
    else:
        materializer = materializers[0]

    path: list[int] | None = None
    if materializer is not None:
        path = _dependency_path(nodes, materializer[1], sink_index)
        if path is None:
            reasons.append("window tensor has no captured dependency path to its contraction")
        elif any(index not in parts["owners"] for index in [materializer[0], *path]):
            reasons.append("window-to-contraction path lacks exact source-task ownership")
        materializer_task = parts["owners"].get(materializer[0])
        if not isinstance(materializer_task, Mapping) or materializer_task.get("declared_task_kind") != _HOST_TASK_KIND:
            reasons.append("window materializer is not owned by one proved host task")
            materializer_task = {}
    else:
        materializer_task = {}

    if (
        parallel is not None
        and reduction is not None
        and materializer is not None
        and len(sink_shapes) == 2
        and all(shape is not None for shape in sink_shapes)
        and sink_output_shape is not None
    ):
        co, columns = parallel
        kdim = reduction[0]
        source_shape, window_shape = materializer[2], materializer[3]
        ci, kh, kw, batch, ho, wo = window_shape
        if (
            source_shape[:2] != [batch, ci]
            or kdim != ci * kh * kw
            or columns != batch * ho * wo
            or sink_shapes != [[co, kdim], [kdim, columns]]
            or sink_output_shape != [co, columns]
        ):
            reasons.append("window tensor, contraction buffers, and proved MAC geometry disagree")
    if reasons:
        return "refused", None, sorted(set(reasons))

    (materializer_index, _window_buffer, source_shape, window_shape, source_dtype, window_dtype) = materializer
    ci, kh, kw, batch, ho, wo = window_shape
    co, columns = parallel
    source_ids = sorted(set([materializer_index, *path]))
    site = {
        "classification": "materialized",
        "eligible": True,
        "source_operation_id": sink_index,
        "source_operation_ids": source_ids,
        "source_region_id": region,
        "source_operation": placement.get("op"),
        "geometry": {
            "batch": batch,
            "input_channels": ci,
            "input_height": source_shape[2],
            "input_width": source_shape[3],
            "output_channels": co,
            "kernel_height": kh,
            "kernel_width": kw,
            "output_height": ho,
            "output_width": wo,
            "flattened_columns": columns,
        },
        "buffers": {
            "window_source": {
                "id": nodes[materializer_index]["inputs"][0],
                "shape": source_shape,
                "dtype": source_dtype,
            },
            "materialized_window": {
                "id": nodes[materializer_index]["outputs"][0],
                "shape": window_shape,
                "dtype": window_dtype,
            },
            "contraction_lhs": {"id": sink_inputs[0], "shape": sink_shapes[0], "dtype": sink_dtypes[0]},
            "contraction_rhs": {"id": sink_inputs[1], "shape": sink_shapes[1], "dtype": sink_dtypes[1]},
            "result": {"id": sink_outputs[0], "shape": sink_output_shape, "dtype": sink_output_dtype},
        },
        "dependency_path": [materializer_index, *path],
        "ownership_evidence": {
            "status": "proved",
            "materializer": _ownership(materializer_task, parts),
            "contraction": _ownership(sink_task, parts),
        },
        "proof_scope": "static window/contraction geometry, dependency path, and task ownership",
        "not_proven": ["replacement semantic equivalence", "runtime correctness", "cycle improvement"],
    }
    site["site_binding_sha256"] = _digest(site)
    return "materialized", site, []


def _member_inventory(
    analysis: Mapping[str, Any],
    identity: Mapping[str, Any],
    candidate_sha256: str,
    member_index: int,
) -> dict[str, Any]:
    parts, problems = _analysis_parts(analysis, candidate_sha256)
    member = {
        "member_index": member_index,
        "identity": dict(identity),
        "identity_sha256": _digest(identity),
        "analysis_location": "/analysis" if member_index == 0 else f"/portfolio/members/{member_index}/analysis",
        "analysis_sha256": _digest(analysis),
        "status": "not_ready",
        "source_operation_ids": [],
        "streamed": [],
        "materialized": [],
        "already_direct": [],
        "already_native": [],
        "uncaptured": [],
        "refused": [],
        "problems": list(problems),
    }
    if not parts:
        member["inventory_sha256"] = _digest(member)
        return member
    placements, placement_problems = _placement_rows(parts)
    member["problems"].extend(placement_problems)
    tasks = _task_rows(analysis)
    routes, route_problems = _convolution_routes(analysis, tasks)
    member["problems"].extend(route_problems)

    regions: dict[str, list[int]] = {}
    direct_sinks: list[int] = []
    materialized_sinks: dict[str, int] = {}
    ambiguous_materialized_regions: set[str] = set()
    for index, node in enumerate(parts["nodes"]):
        provenance = node.get("prov") if isinstance(node, Mapping) else None
        if not isinstance(provenance, Mapping):
            continue
        path = provenance.get("prov.conv_path")
        region = provenance.get("prov.region_id")
        if path not in {"direct_contraction", "im2col_matmul"}:
            hint = provenance.get("prov._pattern_hint")
            if isinstance(hint, str) and "conv" in hint.lower():
                member["uncaptured"].append(
                    {
                        "source_operation_id": index,
                        "refusal_class": "uncaptured_convolution",
                        "reasons": ["convolution provenance has no exact lowering-path classification"],
                    }
                )
            continue
        if not isinstance(region, str) or not region:
            member["uncaptured"].append(
                {
                    "source_operation_id": index,
                    "refusal_class": "uncaptured_convolution",
                    "reasons": ["convolution provenance has no exact source region identity"],
                }
            )
            continue
        regions.setdefault(region, []).append(index)
        if provenance.get("prov.role") == "contraction":
            if path == "direct_contraction":
                direct_sinks.append(index)
            elif region in materialized_sinks:
                ambiguous_materialized_regions.add(region)
                member["refused"].append(
                    {
                        "source_operation_id": index,
                        "refusal_class": "ambiguous_convolution_region",
                        "reasons": ["convolution region has multiple contraction-role sinks"],
                    }
                )
            else:
                materialized_sinks[region] = index

    observed_regions: set[str] = set()
    for source_index in sorted(direct_sinks):
        node = parts["nodes"][source_index]
        region = node["prov"]["prov.region_id"]
        observed_regions.add(region)
        placement = placements.get(source_index)
        if placement is None:
            member["refused"].append(
                {
                    "source_operation_id": source_index,
                    "refusal_class": "missing_convolution_geometry",
                    "reasons": ["source convolution has no proved contraction-domain observation"],
                }
            )
            continue
        owner = parts["owners"].get(source_index)
        task_index = owner.get("task_index") if isinstance(owner, Mapping) else None
        classification, site, reasons = _classify_semantic_task(source_index, placement, parts, routes.get(task_index))
        if site is None:
            member["refused"].append(
                {
                    "source_operation_id": source_index,
                    "refusal_class": "incomplete_convolution_site",
                    "reasons": reasons,
                }
            )
        else:
            member[classification].append(site)

    for region, sink_index in sorted(materialized_sinks.items(), key=lambda item: item[1]):
        observed_regions.add(region)
        if region in ambiguous_materialized_regions:
            continue
        placement = placements.get(sink_index)
        if placement is None:
            member["refused"].append(
                {
                    "source_operation_id": sink_index,
                    "refusal_class": "missing_convolution_geometry",
                    "reasons": ["materialized convolution has no proved contraction-domain observation"],
                }
            )
            continue
        _classification, site, reasons = _classify_materialized_region(
            region, regions[region], sink_index, placement, parts
        )
        if site is None:
            member["refused"].append(
                {
                    "source_operation_id": sink_index,
                    "refusal_class": "incomplete_materialized_convolution_site",
                    "reasons": reasons,
                }
            )
        else:
            member["materialized"].append(site)

    for region, indices in regions.items():
        if region not in observed_regions:
            member["uncaptured"].append(
                {
                    "source_operation_id": min(indices),
                    "source_operation_ids": sorted(indices),
                    "refusal_class": "uncaptured_convolution",
                    "reasons": ["convolution region has no unique contraction-role sink"],
                }
            )

    for key in ("streamed", "materialized", "already_direct", "already_native"):
        member[key].sort(key=lambda row: row["source_operation_id"])
    for key in ("uncaptured", "refused"):
        member[key].sort(
            key=lambda row: (
                -1 if row.get("source_operation_id") is None else row["source_operation_id"],
                row["reasons"],
            )
        )
    opportunities = [*member["streamed"], *member["materialized"]]
    member["source_operation_ids"] = sorted(
        {source_id for site in opportunities for source_id in site["source_operation_ids"]}
    )
    member["problems"] = sorted(set(member["problems"]))
    if member["problems"]:
        # A contradictory portfolio denominator invalidates every otherwise useful assignment.
        member["source_operation_ids"] = []
        member["status"] = "not_ready"
    elif opportunities:
        member["status"] = "eligible_sites_bound"
    else:
        member["status"] = "no_eligible_sites"
    member["bindings"] = _binding(parts)
    member["inventory_sha256"] = _digest(member)
    return member


def inventory_portfolio_convolution_windows(
    record: Mapping[str, Any],
    *,
    iteration_record_sha256: str | None = None,
) -> dict[str, Any]:
    """Derive exact per-member convolution-window categories from one static iteration."""
    result = {
        "schema": INVENTORY_SCHEMA,
        "mechanism_id": MECHANISM_ID,
        "status": "not_ready",
        "candidate_sha256": record.get("candidate_sha256") if isinstance(record, Mapping) else None,
        "portfolio_sha256": None,
        "iteration_record_sha256": iteration_record_sha256,
        "iteration_payload_sha256": _digest(record) if isinstance(record, Mapping) else None,
        "ordered_portfolio": [],
        "members": [],
        "problems": [],
        "proof_scope": "static convolution-window source-site assignment only",
        "not_proven": ["replacement semantic equivalence", "runtime correctness", "performance improvement"],
    }
    if not isinstance(record, Mapping):
        result["problems"] = ["iteration record is missing"]
        result["sha256"] = _digest(result)
        return result
    if iteration_record_sha256 is not None and not _pin(iteration_record_sha256):
        result["problems"].append("iteration_record_sha256 is not an exact SHA-256 digest")
    identities, analyses = _ordered_analyses(record, result["problems"])
    portfolio = record.get("portfolio") if isinstance(record.get("portfolio"), Mapping) else {}
    result["portfolio_sha256"] = portfolio.get("portfolio_sha256")
    result["ordered_portfolio"] = identities
    if len(identities) != len(analyses) or not identities:
        result["sha256"] = _digest(result)
        return result
    for index, (identity, analysis) in enumerate(zip(identities, analyses, strict=True)):
        result["members"].append(_member_inventory(analysis, identity, record["candidate_sha256"], index))
    if any(member["status"] == "not_ready" for member in result["members"]):
        result["problems"].append("one or more portfolio member inventories are not ready")
    opportunities = sum(len(member["streamed"]) + len(member["materialized"]) for member in result["members"])
    result["status"] = (
        "ready_for_work_order"
        if not result["problems"] and opportunities
        else "no_eligible_sites"
        if not result["problems"]
        else "not_ready"
    )
    result["problems"] = sorted(set(result["problems"]))
    result["sha256"] = _digest(result)
    return result


def _definition_kinds(source: str) -> dict[str, str]:
    result: dict[str, str] = {}

    def visit(nodes: Sequence[ast.stmt], parents: tuple[str, ...] = ()) -> None:
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                name = ".".join((*parents, node.name))
                result[name] = "class"
                visit(node.body, (*parents, node.name))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = ".".join((*parents, node.name))
                result[name] = "method" if parents else "function"

    visit(ast.parse(source).body)
    return result


def _mechanism_catalog(contract: Mapping[str, Any], candidate: Path) -> dict[str, Any]:
    selected = [
        row
        for row in contract.get("existing_symbols", [])
        if isinstance(row, Mapping) and row.get("surface_id") in _MECHANISM_SURFACES
    ]
    found = {row.get("surface_id") for row in selected}
    if _REQUIRED_SURFACES - found:
        raise ValueError("edit contract lacks required convolution-window surfaces")
    selectors: list[dict[str, Any]] = []
    paths: set[str] = set()
    seen: set[tuple[str, str, str | None]] = set()
    for row in selected:
        path, symbol = row.get("path"), row.get("symbol")
        if not isinstance(path, str) or not isinstance(symbol, str):
            raise ValueError("convolution-window compiler surface identity is malformed")
        source = candidate / path
        if source.is_symlink() or not source.is_file():
            raise ValueError("convolution-window compiler surface source is missing")
        kind = _definition_kinds(source.read_text()).get(symbol)
        if kind not in {"class", "function", "method"}:
            raise ValueError("convolution-window compiler surface does not resolve to an AST unit")
        key = (kind, path, symbol)
        if key not in seen:
            selectors.append({"kind": kind, "path": path, "symbol": symbol})
            seen.add(key)
            paths.add(path)
    for path in sorted(paths):
        selectors.append({"kind": "imports", "path": path})
    for extension in contract.get("helper_extensions", []):
        if isinstance(extension, Mapping) and set(extension.get("surface_ids", [])) & _MECHANISM_SURFACES:
            selectors.append({"kind": "helper", "directory": extension["directory"]})
    selectors.sort(key=lambda row: (row["kind"], row.get("path", row.get("directory", "")), row.get("symbol", "")))
    catalog = {
        "schema": "compiler_mechanism_catalog_v1",
        "contract_sha256": contract["sha256"],
        "mechanisms": [{"id": MECHANISM_ID, "selectors": selectors}],
    }
    catalog["sha256"] = _digest(catalog)
    return validate_mechanism_catalog(catalog, candidate, contract)


def build_convolution_window_mechanism_documents(
    record: Mapping[str, Any],
    *,
    candidate: Path,
    expected_candidate_sha256: str,
    iteration_record_sha256: str,
) -> dict[str, Any]:
    """Build, but do not write, the exact contract/catalog/inventory/work order."""
    candidate = Path(candidate)
    if (
        not candidate.is_absolute()
        or candidate.resolve() != candidate
        or candidate.is_symlink()
        or not candidate.is_dir()
        or not _pin(expected_candidate_sha256)
        or hash_tree(candidate)["sha256"] != expected_candidate_sha256
        or record.get("candidate_sha256") != expected_candidate_sha256
    ):
        raise ValueError("explicit candidate path/hash does not match the static iteration")
    if not _pin(iteration_record_sha256):
        raise ValueError("static iteration requires an exact raw SHA-256 digest")
    protocol_problems: list[str] = []
    _identities, analyses = _ordered_analyses(record, protocol_problems)
    if protocol_problems:
        raise ValueError("static iteration portfolio is not exact: " + "; ".join(protocol_problems))
    contract = _embedded_edit_contract(record, analyses, candidate)
    inventory = inventory_portfolio_convolution_windows(record, iteration_record_sha256=iteration_record_sha256)
    if inventory["status"] != "ready_for_work_order":
        raise ValueError("static iteration has no fully evidenced convolution-window opportunity")
    catalog = _mechanism_catalog(contract, candidate)
    site_bindings = []
    for member in inventory["members"]:
        bindings = member["bindings"]
        opportunities = sorted(
            [*member["streamed"], *member["materialized"]],
            key=lambda row: row["source_operation_id"],
        )
        site_bindings.append(
            {
                "capsule": member["identity"]["capsule"],
                "capsule_sha256": member["identity"]["capsule_sha256"],
                "compiler_sha256": bindings["compiler_sha256"],
                "source_sha256": bindings["source_sha256"],
                "plan_digest": bindings["plan_digest"],
                "candidate_command_buffer_sha256": bindings["command_buffer_sha256"],
                "candidate_lowered_sha256": bindings["lowered_sha256"],
                "status": member["status"],
                "source_operation_ids": member["source_operation_ids"],
                "chains": opportunities,
                "inventory": {
                    "schema": INVENTORY_SCHEMA,
                    "member_index": member["member_index"],
                    "analysis_location": member["analysis_location"],
                    "analysis_sha256": member["analysis_sha256"],
                    "member_inventory_sha256": member["inventory_sha256"],
                    "portfolio_inventory_sha256": inventory["sha256"],
                    "iteration_record_sha256": iteration_record_sha256,
                    "streamed_count": len(member["streamed"]),
                    "materialized_count": len(member["materialized"]),
                    "already_direct_count": len(member["already_direct"]),
                    "already_native_count": len(member["already_native"]),
                    "uncaptured_count": len(member["uncaptured"]),
                    "refused_count": len(member["refused"]),
                },
            }
        )
    work_order = {
        "schema": "host_prepared_mechanism_work_order_v1",
        "status": "ready_for_authoring",
        "mechanism_id": MECHANISM_ID,
        "catalog_sha256": catalog["sha256"],
        "contract_sha256": contract["sha256"],
        "initial_candidate_sha256": expected_candidate_sha256,
        "round_start_candidate_sha256": expected_candidate_sha256,
        "portfolio_sha256": inventory["portfolio_sha256"],
        "ordered_portfolio": inventory["ordered_portfolio"],
        "source_operation_ids": [],
        "portfolio_site_bindings": site_bindings,
    }
    work_order["sha256"] = _digest(work_order)
    return {
        "edit_contract": contract,
        "catalog": catalog,
        "inventory": inventory,
        "work_order": work_order,
    }


def _write_once(path: Path, document: Mapping[str, Any]) -> str:
    raw = (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
    path.chmod(0o444)
    return _raw_sha256(raw)


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Seal one convolution-window elimination Phase-2 catalog/work order")
    parser.add_argument("iteration", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iteration-sha256", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    args = parser.parse_args(argv)
    iteration, candidate, output = args.iteration, args.candidate, args.output
    if (
        not iteration.is_absolute()
        or iteration.resolve() != iteration
        or iteration.is_symlink()
        or not iteration.is_file()
        or iteration.stat().st_mode & 0o222
    ):
        parser.error("iteration must be an absolute, read-only, non-symlink regular file")
    raw = iteration.read_bytes()
    if not _pin(args.iteration_sha256) or _raw_sha256(raw) != args.iteration_sha256:
        parser.error("iteration raw SHA-256 pin does not match")
    try:
        record = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        parser.error(f"iteration is not valid JSON: {exc}")
    try:
        documents = build_convolution_window_mechanism_documents(
            record,
            candidate=candidate,
            expected_candidate_sha256=args.candidate_sha256,
            iteration_record_sha256=args.iteration_sha256,
        )
        if hash_tree(candidate)["sha256"] != args.candidate_sha256:
            raise ValueError("candidate changed while deriving the sealed work order")
        output.mkdir(mode=0o755, parents=False, exist_ok=False)
        names = {
            "compiler_edit_contract.json": documents["edit_contract"],
            "mechanism_catalog.json": documents["catalog"],
            "mechanism_work_order.json": documents["work_order"],
            "source_site_inventory.json": documents["inventory"],
        }
        artifacts = {name: {"sha256": _write_once(output / name, document)} for name, document in names.items()}
        receipt = {
            "schema": "sealed_convolution_window_work_order_receipt_v1",
            "mechanism_id": MECHANISM_ID,
            "candidate_sha256": args.candidate_sha256,
            "iteration_record_sha256": args.iteration_sha256,
            "inventory_sha256": documents["inventory"]["sha256"],
            "catalog_sha256": documents["catalog"]["sha256"],
            "work_order_sha256": documents["work_order"]["sha256"],
            "artifacts": dict(artifacts),
        }
        receipt["sha256"] = _digest(receipt)
        artifacts["receipt.json"] = {"sha256": _write_once(output / "receipt.json", receipt)}
    except (FileExistsError, OSError, ValueError) as exc:
        parser.error(str(exc))
    report = {
        "status": "ready_for_authoring",
        "mechanism_id": MECHANISM_ID,
        "output": str(output),
        "streamed_counts": [len(member["streamed"]) for member in documents["inventory"]["members"]],
        "materialized_counts": [len(member["materialized"]) for member in documents["inventory"]["members"]],
        "already_direct_counts": [len(member["already_direct"]) for member in documents["inventory"]["members"]],
        "already_native_counts": [len(member["already_native"]) for member in documents["inventory"]["members"]],
        "uncaptured_counts": [len(member["uncaptured"]) for member in documents["inventory"]["members"]],
        "refused_counts": [len(member["refused"]) for member in documents["inventory"]["members"]],
        "artifacts": artifacts,
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
