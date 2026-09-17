"""Seal capability-proven narrow-epilogue source sites for one static portfolio iteration.

This policy joins two independently frozen inputs:

* a complete-graph iteration proving source edges, task ownership, types, representations, and
  candidate/plan/artifact identities; and
* target capability evidence proving one exact epilogue form, including every stage operand,
  representation, narrow output, and completed readout/store.

Neither source spellings nor instruction names imply hardware support.  Floating-point paths,
second-tensor residual adds, missing representations, partial ownership, and unproved capability
forms remain explicit refusals.  The module contains no workload names, target names, tile sizes,
simulator choices, or target constants.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from argparse import ArgumentParser
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.benchharness import hash_tree
from merlin.perf.compiler_edit_scope import validate_edit_contract, validate_mechanism_catalog
from merlin.perf.rank_general_contraction_work_order import (
    _PIN_FIELDS,
    _analysis_parts,
    _digest,
    _embedded_edit_contract,
    _ordered_analyses,
    _pin,
    _raw_sha256,
    _site,
)


MECHANISM_ID = "t01_04_capability_proven_narrow_epilogue_readout"
INVENTORY_SCHEMA = "portfolio_capability_proven_narrow_epilogue_inventory_v2"
CAPABILITY_SCHEMA = "target_narrow_epilogue_capability_evidence_v1"

_SOURCE_PLAN_SCHEMA = "source_plan_metadata_v1"
_SOURCE_STORAGE_SCHEMA = "source_buffer_physical_storage_v1"
_SOURCE_EPILOGUE_SCHEMA = "source_integer_epilogue_ownership_v1"

_SOURCE_STAGES = frozenset({"acc_scale", "bias", "activation", "requant", "narrow_store"})
_REQUIRED_SURFACES = frozenset({
    "epilogue_semantics",
    "target_readout",
    "target_readout_ir",
    "final_target_emission",
    "host_pointwise_and_residual",
})
_MECHANISM_SURFACES = _REQUIRED_SURFACES | frozenset({
    "global_partition",
    "contraction_placement",
    "convolution_route",
    "ranked_contraction_route",
    "pipeline_issue",
    "source_convolution_semantic_task",
    "retire_nested_host_epilogue_hazard_frontier",
})


def _integer_width(dtype: Any) -> int | None:
    if (not isinstance(dtype, str) or len(dtype) < 2 or dtype[0] != "i"
            or not dtype[1:].isdigit() or int(dtype[1:]) <= 0):
        return None
    return int(dtype[1:])


def _floating(dtype: Any) -> bool:
    return isinstance(dtype, str) and (dtype.startswith("f") or dtype.startswith("bf"))


def _source_buffer_witness(name: str, buffer: Mapping[str, Any]) -> dict[str, Any]:
    """Keep exact graph operands visible without making buffer names part of a target form."""
    return {
        "buffer": name,
        "dtype": buffer.get("dtype"),
        "shape": buffer.get("shape"),
        "kind": buffer.get("kind"),
        "encoding": buffer.get("encoding"),
        "layout": buffer.get("layout"),
    }


def _buffer_contract(buffer: Mapping[str, Any], *, role: str,
                     relation: str) -> tuple[dict[str, Any], list[str]]:
    problems: list[str] = []
    shape, dtype = buffer.get("shape"), buffer.get("dtype")
    encoding, layout = buffer.get("encoding"), buffer.get("layout")
    if (not isinstance(shape, list)
            or any(type(extent) is not int or extent < 0 for extent in shape)):
        problems.append("operand shape is not an exact static non-negative shape")
        shape = []
    if not isinstance(dtype, str) or not dtype:
        problems.append("operand dtype is absent")
    if not isinstance(encoding, str) or not encoding:
        problems.append("operand encoding is absent from the captured graph")
    if not isinstance(layout, str) or not layout:
        problems.append("operand layout is absent from the captured graph")
    return {
        "role": role,
        "dtype": dtype,
        "rank": len(shape),
        "encoding": encoding,
        "layout": layout,
        "relation": relation,
    }, problems


def _broadcast_relation(shape: Any, output_shape: Any) -> str:
    if not isinstance(shape, list) or not isinstance(output_shape, list):
        return "unproved"
    if shape == output_shape:
        return "exact"
    if not shape:
        return "scalar"
    if len(shape) > len(output_shape):
        return "unproved"
    padded = [1] * (len(output_shape) - len(shape)) + shape
    if all(left in (1, right) for left, right in zip(padded, output_shape, strict=True)):
        return "trailing_broadcast"
    return "unproved"


def _graph_links(parts: Mapping[str, Any]) -> tuple[dict[str, int], dict[str, list[int]], list[str]]:
    producer: dict[str, int] = {}
    uses: dict[str, list[int]] = defaultdict(list)
    problems: list[str] = []
    buffers, nodes = parts["buffers"], parts["nodes"]
    for name, buffer in buffers.items():
        if (not isinstance(name, str) or not name or not isinstance(buffer, Mapping)
                or buffer.get("id") != name or not isinstance(buffer.get("shape"), list)
                or any(type(extent) is not int or extent < 0
                       for extent in buffer.get("shape", []))
                or not isinstance(buffer.get("dtype"), str) or not buffer.get("dtype")):
            problems.append(f"logical graph buffer {name!r} lacks an exact structural contract")
    for index, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            problems.append(f"logical graph node {index} is malformed")
            continue
        inputs, outputs = node.get("inputs"), node.get("outputs")
        if (not isinstance(inputs, list) or not isinstance(outputs, list)
                or any(not isinstance(name, str) or not name for name in [*inputs, *outputs])):
            problems.append(f"logical graph node {index} has malformed buffer edges")
            continue
        for name in inputs:
            if name not in buffers:
                problems.append(f"logical graph node {index} reads absent buffer {name}")
            uses[name].append(index)
        for name in outputs:
            if name not in buffers:
                problems.append(f"logical graph node {index} writes absent buffer {name}")
            if name in producer:
                problems.append(f"logical graph buffer {name} has multiple producers")
            producer[name] = index
    for name, consumers in uses.items():
        if name in producer and any(producer[name] >= consumer for consumer in consumers):
            problems.append(f"logical graph edge for {name} is not in source order")
    return producer, dict(uses), sorted(set(problems))


def _source_plan_adapter(parts: Mapping[str, Any]
                         ) -> tuple[dict[str, Any] | None, dict[int, Mapping[str, Any]],
                                    list[str]]:
    """Expose only source facts backed by the verifier's physical/semantic join.

    The captured graph is logical: any ``encoding``/``layout`` strings or epilogue provenance in
    that payload are not physical or semantic proof.  This adapter removes those hints and adds
    them back only after checking the host-built ``source_plan_metadata`` against the same graph,
    task ownership, and verified storage-contract table.  Older records therefore fail closed
    instead of accidentally becoming executable work orders.
    """
    problems: list[str] = []
    plan = parts.get("plan")
    metadata = plan.get("source_plan_metadata") if isinstance(plan, Mapping) else None
    if not isinstance(metadata, Mapping):
        return None, {}, ["verified global plan has no source-bound plan metadata"]
    storage = metadata.get("physical_storage")
    epilogues = metadata.get("integer_epilogue_ownership")
    if (metadata.get("schema") != _SOURCE_PLAN_SCHEMA
            or metadata.get("status") not in {"verified", "partial"}
            or metadata.get("problems") != []
            or not isinstance(storage, Mapping) or not isinstance(epilogues, Mapping)):
        return None, {}, ["source-bound plan metadata schema, status, or join is invalid"]

    nodes = copy.deepcopy(parts["nodes"])
    buffers = copy.deepcopy(parts["buffers"])
    owners = parts["owners"]
    for buffer in buffers.values():
        if isinstance(buffer, dict):
            buffer.pop("encoding", None)
            buffer.pop("layout", None)
    for node in nodes:
        provenance = node.get("prov") if isinstance(node, Mapping) else None
        if isinstance(provenance, dict):
            for field in ("prov.epilogue_stage", "prov.epilogue_operation",
                          "prov.epilogue_operand_roles"):
                provenance.pop(field, None)

    rows, unknown = storage.get("rows"), storage.get("unknown")
    storage_encodings = plan.get("storage_encodings")
    if (storage.get("schema") != _SOURCE_STORAGE_SCHEMA
            or storage.get("status") not in {"complete", "partial"}
            or not isinstance(rows, list) or not isinstance(unknown, list)
            or not isinstance(storage_encodings, Mapping)
            or storage.get("materialized_source_values") != len(rows) + len(unknown)
            or storage.get("exact_physical_representations") != len(rows)):
        problems.append("source-bound physical-storage inventory is malformed")
        rows, unknown, storage_encodings = [], [], {}
    if ((storage.get("status") == "complete") != (not unknown)
            or (metadata.get("status") == "verified") != (storage.get("status") == "complete")):
        problems.append("source-bound metadata and physical-storage completeness disagree")

    represented: set[str] = set()
    represented_tensors: set[str] = set()
    for index, row in enumerate(rows):
        label = f"physical-storage row {index}"
        if not isinstance(row, Mapping):
            problems.append(f"{label} is malformed")
            continue
        name, tensor = row.get("source_buffer"), row.get("materialized_tensor")
        logical, physical = row.get("logical"), row.get("physical_tensor")
        encoding_sha, layout_sha = row.get("encoding_sha256"), row.get("layout_sha256")
        checked = storage_encodings.get(tensor) if isinstance(tensor, str) else None
        contract = checked.get("contract") if isinstance(checked, Mapping) else None
        layout_contract = row.get("layout_contract")
        source = buffers.get(name) if isinstance(name, str) else None
        expected_location = f"/storage_encodings/{tensor}/contract"
        if (set(row) != {"source_buffer", "source_origin", "materialized_tensor", "logical",
                         "physical_tensor", "encoding", "encoding_sha256",
                         "encoding_contract_location", "layout", "layout_sha256",
                         "layout_contract", "proof_scope", "caller_materialization",
                         "emitted_consumer_addressing"}
                or not isinstance(source, Mapping) or name in represented
                or not isinstance(tensor, str) or not tensor or tensor in represented_tensors
                or not isinstance(logical, Mapping)
                or logical != {"shape": source.get("shape"), "dtype": source.get("dtype")}
                or not isinstance(physical, Mapping)
                or not isinstance(contract, Mapping) or not _pin(encoding_sha)
                or contract.get("logical_shape") != source.get("shape")
                or contract.get("dtype") != source.get("dtype")
                or contract.get("physical_shape") != physical.get("shape")
                or contract.get("dtype") != physical.get("dtype")
                or _digest(contract) != encoding_sha
                or row.get("encoding") != f"{contract.get('schema')}@sha256:{encoding_sha}"
                or row.get("encoding_contract_location") != expected_location
                or not isinstance(layout_contract, Mapping) or not _pin(layout_sha)
                or _digest(layout_contract) != layout_sha
                or row.get("layout") != f"static_strided_elements_v1@sha256:{layout_sha}"):
            problems.append(f"{label} does not exactly bind graph and verified storage")
            continue
        represented.add(name)
        represented_tensors.add(tensor)
        buffers[name]["encoding"] = row["encoding"]
        buffers[name]["layout"] = row["layout"]
        buffers[name]["materialized_tensor"] = tensor
    for index, row in enumerate(unknown):
        name = row.get("source_buffer") if isinstance(row, Mapping) else None
        source = buffers.get(name) if isinstance(name, str) else None
        logical = row.get("logical") if isinstance(row, Mapping) else None
        if (not isinstance(row, Mapping) or not isinstance(source, Mapping) or name in represented
                or row.get("reason") != "no verified physical storage encoding"
                or logical != {"shape": source.get("shape"), "dtype": source.get("dtype")}):
            problems.append(f"unknown physical-storage row {index} is malformed")
            continue
        represented.add(name)
    if len(represented) != len(rows) + len(unknown):
        problems.append("source-bound physical-storage rows repeat a logical buffer")
    if represented_tensors != set(storage_encodings):
        problems.append("verified storage contracts do not exactly match represented tensors")

    roots = epilogues.get("roots")
    counts = epilogues.get("classification_counts")
    classes = ("complete_integer_epilogue", "partial_integer_epilogue", "unclassified")
    if (epilogues.get("schema") != _SOURCE_EPILOGUE_SCHEMA
            or epilogues.get("status") != "verified" or not isinstance(roots, list)
            or epilogues.get("contraction_roots") != len(roots)
            or not isinstance(counts, Mapping)
            or set(counts) != set(classes)
            or any(counts[name] != sum(isinstance(root, Mapping)
                                       and root.get("classification") == name for root in roots)
                   for name in classes)):
        problems.append("source-bound integer-epilogue inventory is malformed")
        roots = []

    roots_by_id: dict[int, Mapping[str, Any]] = {}
    for root_index, root in enumerate(roots):
        label = f"integer-epilogue root {root_index}"
        source_index = root.get("producer_source_operation_id") \
            if isinstance(root, Mapping) else None
        if (type(source_index) is not int or not 0 <= source_index < len(nodes)
                or source_index in roots_by_id):
            problems.append(f"{label} has no unique source producer")
            continue
        node = nodes[source_index]
        provenance = node.get("prov") if isinstance(node, Mapping) else None
        owner = owners.get(source_index)
        accumulator_name = (node.get("outputs", [None])[0]
                            if isinstance(node, Mapping) and len(node.get("outputs", [])) == 1
                            else None)
        accumulator = buffers.get(accumulator_name)
        source_ids = root.get("source_operation_ids")
        stages = root.get("stages")
        if (set(root) != {"producer_source_operation_id", "producer_task_index",
                          "producer_task_kind", "accumulator_source_buffer", "accumulator",
                          "source_operation_ids", "stages", "reasons", "classification"}
                or root.get("classification") not in classes
                or not isinstance(provenance, Mapping)
                or provenance.get("prov.family") != "contraction"
                or not isinstance(root.get("reasons"), list)
                or any(not isinstance(reason, str) or not reason for reason in root["reasons"])
                or not isinstance(owner, Mapping)
                or root.get("producer_task_index") != owner["task_index"]
                or root.get("producer_task_kind") != owner["declared_task_kind"]
                or root.get("accumulator_source_buffer") != accumulator_name
                or not isinstance(accumulator, Mapping)
                or root.get("accumulator") != {
                    "shape": accumulator.get("shape"), "dtype": accumulator.get("dtype")}
                or not isinstance(stages, list) or not isinstance(source_ids, list)
                or source_ids != [source_index, *[
                    stage.get("source_operation_id") for stage in stages
                    if isinstance(stage, Mapping)]]):
            problems.append(f"{label} disagrees with graph or task ownership")
            continue
        current = accumulator_name
        valid = True
        for stage_index, stage in enumerate(stages):
            operation_id = stage.get("source_operation_id") if isinstance(stage, Mapping) else None
            operation = (nodes[operation_id] if type(operation_id) is int
                         and 0 <= operation_id < len(nodes) else None)
            stage_owner = owners.get(operation_id)
            inputs = stage.get("inputs") if isinstance(stage, Mapping) else None
            output = stage.get("output") if isinstance(stage, Mapping) else None
            semantic = ({key: stage.get(key) for key in (
                "stage", "operation", "inputs", "output", "indexing_maps", "scalar_operations")}
                        if isinstance(stage, Mapping) else {})
            operation_inputs = operation.get("inputs") if isinstance(operation, Mapping) else None
            operation_outputs = operation.get("outputs") if isinstance(operation, Mapping) else None
            primary = inputs[0] if isinstance(inputs, list) and inputs else None
            if (not isinstance(stage, Mapping)
                    or set(stage) != {"stage", "operation", "inputs", "output",
                                      "indexing_maps", "scalar_operations", "semantic_sha256",
                                      "classification_source", "source_operation_id",
                                      "task_index", "task_kind"}
                    or not isinstance(operation, Mapping) or not isinstance(stage_owner, Mapping)
                    or stage.get("task_index") != stage_owner["task_index"]
                    or stage.get("task_kind") != stage_owner["declared_task_kind"]
                    or stage.get("stage") not in _SOURCE_STAGES
                    or not isinstance(stage.get("operation"), str) or not stage["operation"]
                    or not isinstance(inputs, list) or not inputs
                    or not isinstance(output, Mapping)
                    or not isinstance(operation_inputs, list)
                    or len(inputs) != len(operation_inputs)
                    or [operand.get("operand_index") for operand in inputs
                        if isinstance(operand, Mapping)] != list(range(len(operation_inputs)))
                    or not isinstance(operation_outputs, list) or len(operation_outputs) != 1
                    or operation_outputs[0] != output.get("source_buffer")
                    or not isinstance(primary, Mapping)
                    or primary.get("source_buffer") != current
                    or primary.get("role") != "accumulator"
                    or primary.get("relation") != "exact"
                    or not isinstance(stage.get("indexing_maps"), list)
                    or any(not isinstance(item, str) or not item
                           for item in stage["indexing_maps"])
                    or not isinstance(stage.get("scalar_operations"), list)
                    or any(not isinstance(item, str) or not item
                           for item in stage["scalar_operations"])
                    or _digest(semantic) != stage.get("semantic_sha256")):
                problems.append(f"{label} stage {stage_index} identity is invalid")
                valid = False
                break
            seen_roles: set[str] = set()
            for operand in inputs:
                operand_index = operand.get("operand_index") if isinstance(operand, Mapping) else None
                name = operand.get("source_buffer") if isinstance(operand, Mapping) else None
                buffer = buffers.get(name) if isinstance(name, str) else None
                role = operand.get("role") if isinstance(operand, Mapping) else None
                if (not isinstance(operand, Mapping)
                        or set(operand) != {"source_buffer", "operand_index", "role", "relation",
                                            "shape", "dtype"}
                        or type(operand_index) is not int
                        or not 0 <= operand_index < len(operation_inputs)
                        or operation_inputs[operand_index] != name or not isinstance(buffer, Mapping)
                        or operand.get("shape") != buffer.get("shape")
                        or operand.get("dtype") != buffer.get("dtype")
                        or operand.get("relation") not in {"exact", "scalar", "trailing_broadcast"}
                        or not isinstance(role, str) or not role or role in seen_roles):
                    problems.append(f"{label} stage {stage_index} operand contract is invalid")
                    valid = False
                    break
                seen_roles.add(role)
            final_buffer = buffers.get(output.get("source_buffer"))
            if (not valid or set(output) != {"source_buffer", "shape", "dtype"}
                    or not isinstance(final_buffer, Mapping)
                    or output.get("shape") != final_buffer.get("shape")
                    or output.get("dtype") != final_buffer.get("dtype")):
                if valid:
                    problems.append(f"{label} stage {stage_index} output contract is invalid")
                valid = False
                break
            provenance = operation.setdefault("prov", {})
            provenance["prov.epilogue_stage"] = stage["stage"]
            provenance["prov.epilogue_operation"] = stage["operation"]
            provenance["prov.epilogue_operand_roles"] = [
                operand["role"] for operand in inputs[1:]]
            current = output["source_buffer"]
        accumulator_width = _integer_width(accumulator.get("dtype"))
        final_width = _integer_width(buffers.get(current, {}).get("dtype"))
        completed = (bool(stages) and accumulator_width is not None and final_width is not None
                     and final_width < accumulator_width)
        expected_classification = ("complete_integer_epilogue" if completed else
                                   "partial_integer_epilogue" if stages else "unclassified")
        if valid and root.get("classification") != expected_classification:
            problems.append(f"{label} completion classification is inconsistent")
            valid = False
        if valid:
            roots_by_id[source_index] = root

    if problems:
        return None, {}, sorted(set(problems))
    adapted = dict(parts)
    adapted.update({"nodes": nodes, "buffers": buffers,
                    "source_plan_epilogue_roots": roots_by_id})
    program = dict(parts["program"])
    program.update({"nodes": nodes, "buffers": buffers})
    adapted["program"] = program
    return adapted, roots_by_id, []


def _fallback_residual(node: Mapping[str, Any], current: str,
                       buffers: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    provenance = node.get("prov") if isinstance(node.get("prov"), Mapping) else {}
    if (node.get("kind") != "dispatch" or provenance.get("prov.op") != "add"
            or current not in node.get("inputs", []) or len(node.get("outputs", [])) != 1):
        return None
    output = buffers.get(node["outputs"][0])
    primary = buffers.get(current)
    if not isinstance(output, Mapping) or not isinstance(primary, Mapping):
        return None
    for name in node["inputs"]:
        extra = buffers.get(name)
        if (name != current and isinstance(extra, Mapping)
                and extra.get("shape") == primary.get("shape") == output.get("shape")):
            return {
                "buffer": name,
                "dtype": extra.get("dtype"),
                "shape": extra.get("shape"),
                "reason": "add consumes a second full-shape tensor operand",
            }
    return None


def _explicit_stage(node: Mapping[str, Any], *, source_index: int, current: str,
                    buffers: Mapping[str, Mapping[str, Any]], first: bool,
                    ) -> tuple[dict[str, Any] | None, list[str], bool, dict[str, Any] | None]:
    """Return an exact normalized stage, refusal reasons, float use, and residual witness."""
    reasons: list[str] = []
    residual = _fallback_residual(node, current, buffers)
    inputs, outputs = node.get("inputs"), node.get("outputs")
    if node.get("kind") != "dispatch":
        return None, ["epilogue path contains a representation/view operation"], False, residual
    if (not isinstance(inputs, list) or inputs.count(current) != 1
            or not isinstance(outputs, list) or len(outputs) != 1):
        return None, ["epilogue stage lacks one exact chain input and output"], False, residual
    provenance = node.get("prov") if isinstance(node.get("prov"), Mapping) else {}
    stage = provenance.get("prov.epilogue_stage")
    operation = provenance.get("prov.epilogue_operation")
    roles = provenance.get("prov.epilogue_operand_roles")
    extras = [name for name in inputs if name != current]
    if stage not in _SOURCE_STAGES:
        reasons.append("source operation has no supported explicit epilogue-stage identity")
    if not isinstance(operation, str) or not operation:
        reasons.append("source operation has no exact epilogue operation identity")
    if (not isinstance(roles, list) or len(roles) != len(extras)
            or any(not isinstance(role, str) or not role for role in roles)
            or len(set(roles)) != len(roles)):
        reasons.append("epilogue additional-operand roles are absent or ambiguous")
        roles = [f"unresolved_{index}" for index in range(len(extras))]

    output = buffers.get(outputs[0])
    primary = buffers.get(current)
    if not isinstance(primary, Mapping) or not isinstance(output, Mapping):
        return None, sorted(set([*reasons, "epilogue buffer contract is absent"])), False, residual
    primary_contract, contract_problems = _buffer_contract(
        primary, role="accumulator" if first else "chain_value", relation="exact")
    reasons.extend(contract_problems)
    operand_contracts = [primary_contract]
    for name, role in zip(extras, roles, strict=True):
        extra = buffers.get(name)
        if not isinstance(extra, Mapping):
            reasons.append(f"additional operand {name!r} has no buffer contract")
            continue
        relation = _broadcast_relation(extra.get("shape"), output.get("shape"))
        contract, problems = _buffer_contract(extra, role=role, relation=relation)
        operand_contracts.append(contract)
        reasons.extend(problems)
    output_contract, output_problems = _buffer_contract(output, role="stage_output", relation="exact")
    output_contract.pop("role")
    output_contract.pop("relation")
    reasons.extend(output_problems)
    if primary.get("shape") != output.get("shape"):
        reasons.append("stage changes logical shape without an exact supported stage contract")
    if stage == "bias":
        if len(extras) != 1:
            reasons.append("bias stage does not have exactly one additional operand")
        elif operand_contracts[-1]["relation"] not in {"scalar", "trailing_broadcast"}:
            reasons.append("bias operand broadcast is not an exact trailing/scalar broadcast")
    elif stage == "acc_scale" and not extras:
        reasons.append("acc_scale stage has no exact scale operand")
    elif stage in {"activation", "narrow_store"} and extras:
        reasons.append(f"{stage} stage unexpectedly consumes additional operands")
    if stage == "narrow_store":
        left, right = _integer_width(primary.get("dtype")), _integer_width(output.get("dtype"))
        if left is None or right is None or right >= left:
            reasons.append("narrow_store does not reduce an integer container width")
    float_used = any(_floating(contract.get("dtype")) for contract in
                     [*operand_contracts, output_contract])
    normalized = {
        "source_operation_id": source_index,
        "stage": stage,
        "operation": operation,
        "inputs": operand_contracts,
        "output": output_contract,
        "source_input_buffers": [
            _source_buffer_witness(name, buffers[name]) for name in inputs if name in buffers
        ],
        "source_output_buffer": _source_buffer_witness(outputs[0], output),
    }
    normalized["stage_binding_sha256"] = _digest(normalized)
    return normalized, sorted(set(reasons)), float_used, residual


def _exact_contract(value: Any, *, fields: set[str], label: str,
                    role: str | None = None, relation: str | None = None) -> list[str]:
    if not isinstance(value, Mapping) or set(value) != fields:
        return [f"{label} does not have the exact representation fields"]
    problems = []
    if _integer_width(value.get("dtype")) is None:
        problems.append(f"{label} is not an integer representation")
    if type(value.get("rank")) is not int or value["rank"] < 0:
        problems.append(f"{label} rank is not exact")
    for field in ("encoding", "layout"):
        if not isinstance(value.get(field), str) or not value[field]:
            problems.append(f"{label} {field} is not exact")
    if role is not None and value.get("role") != role:
        problems.append(f"{label} role is not {role}")
    if relation is not None and value.get("relation") != relation:
        problems.append(f"{label} relation is not {relation}")
    return problems


def _representation(value: Mapping[str, Any]) -> dict[str, Any]:
    return {field: value.get(field) for field in ("dtype", "rank", "encoding", "layout")}


def _proof_form(form: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if set(form) != {"producer", "stage_sequence", "stages", "output", "completion"}:
        return ["capability form does not have the exact normalized fields"]
    producer, stages, output = form.get("producer"), form.get("stages"), form.get("output")
    if (not isinstance(producer, Mapping)
            or set(producer) != {"family", "inputs", "accumulator"}
            or producer.get("family") != "contraction"):
        problems.append("capability form lacks an exact contraction producer")
        producer = {}
    producer_inputs = producer.get("inputs")
    if not isinstance(producer_inputs, list) or len(producer_inputs) != 2:
        problems.append("capability producer does not prove exactly two input operands")
        producer_inputs = []
    for index, role in enumerate(("lhs", "rhs")):
        if index < len(producer_inputs):
            problems.extend(_exact_contract(
                producer_inputs[index],
                fields={"role", "dtype", "rank", "encoding", "layout", "relation"},
                label=f"capability producer {role}", role=role, relation="exact"))
    accumulator = producer.get("accumulator")
    problems.extend(_exact_contract(
        accumulator,
        fields={"dtype", "rank", "encoding", "layout"},
        label="capability accumulator"))
    if not isinstance(stages, list) or not stages:
        problems.append("capability form has no exact stage sequence")
        stages = []
    sequence = form.get("stage_sequence")
    if (not isinstance(sequence, list) or sequence != [stage.get("stage") for stage in stages
                                                       if isinstance(stage, Mapping)]
            or any(stage not in _SOURCE_STAGES for stage in sequence)):
        problems.append("capability stage sequence is absent, unknown, or inconsistent")
    previous = accumulator if isinstance(accumulator, Mapping) else None
    for index, stage in enumerate(stages):
        if (not isinstance(stage, Mapping)
                or set(stage) != {"stage", "operation", "inputs", "output"}
                or not isinstance(stage.get("operation"), str) or not stage["operation"]
                or not isinstance(stage.get("inputs"), list) or not stage["inputs"]
                or not isinstance(stage.get("output"), Mapping)):
            problems.append("capability stage does not prove exact operation/operands/output")
            continue
        stage_name = stage["stage"]
        inputs = stage["inputs"]
        expected_primary_role = "accumulator" if index == 0 else "chain_value"
        for operand_index, operand in enumerate(inputs):
            problems.extend(_exact_contract(
                operand,
                fields={"role", "dtype", "rank", "encoding", "layout", "relation"},
                label=f"capability stage {index} operand {operand_index}",
                role=expected_primary_role if operand_index == 0 else None,
                relation="exact" if operand_index == 0 else None))
        if len({operand.get("role") for operand in inputs if isinstance(operand, Mapping)}) != len(inputs):
            problems.append(f"capability stage {index} operand roles are not unique")
        if any(isinstance(operand, Mapping)
               and operand.get("relation") not in {"exact", "scalar", "trailing_broadcast"}
               for operand in inputs):
            problems.append(f"capability stage {index} has an unproved operand relation")
        primary = inputs[0]
        if (isinstance(previous, Mapping) and isinstance(primary, Mapping)
                and _representation(previous) != _representation(primary)):
            problems.append(f"capability stage {index} breaks representation continuity")
        stage_output = stage["output"]
        problems.extend(_exact_contract(
            stage_output,
            fields={"dtype", "rank", "encoding", "layout"},
            label=f"capability stage {index} output"))
        if stage_name == "bias":
            if (len(inputs) != 2 or not isinstance(inputs[1], Mapping)
                    or inputs[1].get("role") != "bias"
                    or inputs[1].get("relation") not in {"scalar", "trailing_broadcast"}):
                problems.append("capability bias does not prove one broadcast bias operand")
        elif stage_name == "acc_scale" and len(inputs) < 2:
            problems.append("capability acc_scale does not prove a scale operand")
        elif stage_name in {"activation", "narrow_store"} and len(inputs) != 1:
            problems.append(f"capability {stage_name} has unsupported additional operands")
        if stage_name == "narrow_store" and (
                _integer_width(stage_output.get("dtype")) is None
                or _integer_width(primary.get("dtype")) is None
                or _integer_width(stage_output["dtype"]) >= _integer_width(primary["dtype"])):
            problems.append("capability narrow_store does not narrow its chain value")
        previous = stage_output
    if stages and isinstance(stages[-1], Mapping) \
            and stages[-1].get("stage") not in {"requant", "narrow_store"}:
        problems.append("capability form does not end in requant or narrow_store")
    if (not isinstance(output, Mapping)
            or set(output) != {"dtype", "rank", "encoding", "layout", "width_bits"}
            or _integer_width(output.get("dtype")) != output.get("width_bits")
            or not isinstance(output.get("encoding"), str) or not output["encoding"]
            or not isinstance(output.get("layout"), str) or not output["layout"]):
        problems.append("capability output width, encoding, or layout is not exact")
    if (isinstance(output, Mapping) and isinstance(previous, Mapping)
            and _representation(output) != _representation(previous)):
        problems.append("capability final output does not match the final stage representation")
    if (not isinstance(accumulator, Mapping)
            or _integer_width(accumulator.get("dtype")) is None
            or _integer_width(output.get("dtype") if isinstance(output, Mapping) else None) is None
            or _integer_width(output["dtype"]) >= _integer_width(accumulator["dtype"])):
        problems.append("capability form does not prove a narrower integer result")
    if form.get("completion") != "narrow_readout_and_store":
        problems.append("capability form does not prove completed narrow readout and store")
    return sorted(set(problems))


def _capability_forms(record: Mapping[str, Any], analyses: Sequence[Mapping[str, Any]],
                      evidence: Mapping[str, Any] | None,
                      file_sha256: str | None) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    if evidence is None:
        return {}, ([] if file_sha256 is None else
                    ["capability file pin exists without capability evidence"])
    problems: list[str] = []
    if not _pin(file_sha256):
        problems.append("capability evidence requires an exact raw file SHA-256 pin")
    declared = evidence.get("sha256")
    if (evidence.get("schema") != CAPABILITY_SCHEMA or evidence.get("status") != "verified"
            or not _pin(declared)
            or _digest({key: value for key, value in evidence.items() if key != "sha256"}) != declared):
        problems.append("capability evidence schema, status, or canonical identity is invalid")
    binding = record.get("cross_run_static_analysis_binding")
    target_descriptor = binding.get("target_descriptor_sha256") if isinstance(binding, Mapping) else None
    if (not _pin(target_descriptor)
            or evidence.get("target_descriptor_sha256") != target_descriptor):
        problems.append("capability evidence does not match the iteration target descriptor")
    target_facts = set()
    for analysis in analyses:
        try:
            target_facts.add(analysis["diagnostics"]["task_instruction_evidence"]["candidate"][
                "binding"]["target_facts_sha256"])
        except (KeyError, TypeError):
            target_facts.add(None)
    if (len(target_facts) != 1 or not _pin(next(iter(target_facts)))
            or evidence.get("target_instruction_facts_sha256") != next(iter(target_facts))):
        problems.append("capability evidence does not match every member's target instruction facts")

    forms: dict[str, Mapping[str, Any]] = {}
    proofs = evidence.get("proofs")
    if not isinstance(proofs, list) or not proofs:
        problems.append("capability evidence has no exact forms")
        proofs = []
    for index, proof in enumerate(proofs):
        if (not isinstance(proof, Mapping)
                or set(proof) != {"id", "status", "form", "form_sha256", "evidence"}
                or not isinstance(proof.get("id"), str) or not proof["id"]
                or proof.get("status") != "verified" or not isinstance(proof.get("form"), Mapping)
                or _digest(proof.get("form")) != proof.get("form_sha256")):
            problems.append(f"capability proof {index} identity or exact form is invalid")
            continue
        problems.extend(f"capability proof {index}: {reason}"
                        for reason in _proof_form(proof["form"]))
        citations = proof.get("evidence")
        if (not isinstance(citations, list) or not citations
                or any(not isinstance(item, Mapping)
                       or set(item) != {"kind", "locator", "sha256", "scope"}
                       or not isinstance(item.get("kind"), str) or not item["kind"]
                       or not isinstance(item.get("locator"), str) or not item["locator"]
                       or not _pin(item.get("sha256"))
                       or not isinstance(item.get("scope"), str) or not item["scope"]
                       for item in citations)):
            problems.append(f"capability proof {index} lacks pinned completion evidence")
        form_sha = proof.get("form_sha256")
        if form_sha in forms:
            problems.append("capability evidence repeats one exact form")
        else:
            forms[form_sha] = proof
    return ({} if problems else forms), sorted(set(problems))


def _source_form(root: Mapping[str, Any], root_node: Mapping[str, Any],
                 stages: list[Mapping[str, Any]], output_buffer: Mapping[str, Any],
                 buffers: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    producer_inputs = []
    for role, name in zip(("lhs", "rhs"), root_node.get("inputs", [])[:2], strict=True):
        buffer = buffers.get(name)
        if not isinstance(buffer, Mapping):
            reasons.append("contraction input buffer contract is absent")
            continue
        contract, problems = _buffer_contract(buffer, role=role, relation="exact")
        producer_inputs.append(contract)
        reasons.extend(problems)
    accumulator_name = root_node.get("outputs", [None])[0]
    accumulator = buffers.get(accumulator_name)
    if not isinstance(accumulator, Mapping):
        reasons.append("contraction accumulator buffer contract is absent")
        accumulator_contract = {}
    else:
        accumulator_contract, problems = _buffer_contract(
            accumulator, role="accumulator", relation="exact")
        accumulator_contract.pop("role")
        accumulator_contract.pop("relation")
        reasons.extend(problems)
    final, problems = _buffer_contract(output_buffer, role="output", relation="exact")
    final.pop("role")
    final.pop("relation")
    reasons.extend(problems)
    width = _integer_width(final.get("dtype"))
    accumulator_width = _integer_width(accumulator_contract.get("dtype"))
    if width is None or accumulator_width is None or width >= accumulator_width:
        reasons.append("source path has no exact narrower integer output")
    output = {**final, "width_bits": width}
    normalized_stages = [{key: value for key, value in stage.items()
                          if key not in {"source_operation_id", "stage_binding_sha256",
                                         "source_input_buffers", "source_output_buffer"}}
                         for stage in stages]
    form = {
        "producer": {
            "family": "contraction",
            "inputs": producer_inputs,
            "accumulator": accumulator_contract,
        },
        "stage_sequence": [stage["stage"] for stage in normalized_stages],
        "stages": normalized_stages,
        "output": output,
        "completion": "narrow_readout_and_store",
    }
    reasons.extend(_proof_form(form))
    return (None if reasons else form), sorted(set(reasons))


def _trace(root: Mapping[str, Any], parts: Mapping[str, Any],
           uses: Mapping[str, list[int]], results: set[str]) -> dict[str, Any]:
    index = root["source_operation_id"]
    node = parts["nodes"][index]
    buffers, owners = parts["buffers"], parts["owners"]
    output_name = node["outputs"][0]
    accumulator = buffers[output_name]
    accumulator_width = _integer_width(accumulator.get("dtype"))
    input_widths = [_integer_width(buffers[name].get("dtype"))
                    for name in node.get("inputs", [])[:2]]
    base = {
        "producer_source_operation_id": index,
        "producer_region_id": root["source_region_id"],
        "producer_task_index": root["ownership_evidence"]["task_index"],
        "producer_task_kind": root["ownership_evidence"]["declared_task_kind"],
        "accumulator_buffer": output_name,
        "accumulator_dtype": accumulator.get("dtype"),
        "producer_input_buffers": [
            _source_buffer_witness(name, buffers[name])
            for name in node.get("inputs", [])[:2] if name in buffers
        ],
        "source_operation_ids": [index],
        "stages": [],
        "path": [],
        "reasons": [],
    }
    if (accumulator_width is not None and input_widths
            and all(width is not None for width in input_widths)
            and accumulator_width <= max(input_widths)):
        return {**base, "classification": "already_narrow",
                "reasons": ["accelerator-owned contraction has no wider logical accumulator result"]}

    current = output_name
    stages: list[dict[str, Any]] = []
    float_used = False
    uncaptured: list[str] = []
    unsupported: list[str] = []
    residual: dict[str, Any] | None = None
    completion = False
    for _step in range(512):
        consumers = uses.get(current, [])
        if len(consumers) != 1:
            uncaptured.append(
                "epilogue chain has fanout" if consumers else
                "epilogue chain terminates before an exact narrow readout")
            break
        source_index = consumers[0]
        consumer = parts["nodes"][source_index]
        owner = owners.get(source_index)
        if not isinstance(owner, Mapping):
            return {**base, "classification": "ownership_failure",
                    "source_operation_ids": [*base["source_operation_ids"], source_index],
                    "reasons": ["epilogue consumer has no unique source-task owner"]}
        if owner["declared_task_kind"] != "host":
            uncaptured.append("wide accumulator reaches another non-host task before narrow readout")
            break
        path_row = {
            "source_operation_id": source_index,
            "kind": consumer.get("kind"),
            "operation": consumer.get("op"),
            "task_index": owner["task_index"],
            "task_kind": owner["declared_task_kind"],
        }
        base["path"].append(path_row)
        base["source_operation_ids"].append(source_index)
        stage, reasons, uses_float, second_operand = _explicit_stage(
            consumer, source_index=source_index, current=current, buffers=buffers,
            first=not stages)
        if second_operand is not None and residual is None:
            residual = {"source_operation_id": source_index, **second_operand}
        float_used = float_used or uses_float
        if stage is None:
            uncaptured.extend(reasons)
        else:
            stages.append(stage)
            unsupported.extend(reasons)
        outputs = consumer.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != 1 or outputs[0] not in buffers:
            uncaptured.append("epilogue consumer has no exact single output")
            break
        current = outputs[0]
        width = _integer_width(buffers[current].get("dtype"))
        if width is not None and accumulator_width is not None and width < accumulator_width:
            completion = True
            break
    else:
        uncaptured.append("epilogue source path exceeds the bounded acyclic trace length")

    base["stages"] = stages
    base["final_buffer"] = current
    final = buffers.get(current, {})
    base["final_dtype"] = final.get("dtype")
    base["final_shape"] = final.get("shape")
    if residual is not None:
        return {**base, "classification": "residual_second_operand",
                "second_operand": residual,
                "reasons": ["residual add needs a second full-shape tensor operand"]}
    if float_used or any(_floating(buffers[name].get("dtype"))
                         for row in base["path"] for name in
                         parts["nodes"][row["source_operation_id"]].get("inputs", [])
                         if name in buffers):
        return {**base, "classification": "float_or_unsupported_stage",
                "reasons": sorted(set([*unsupported, *uncaptured,
                                       "epilogue path enters floating-point arithmetic"]))}
    if unsupported:
        return {**base, "classification": "float_or_unsupported_stage",
                "reasons": sorted(set([*unsupported, *uncaptured]))}
    if not completion or uncaptured or not stages:
        return {**base, "classification": "uncaptured",
                "reasons": sorted(set(uncaptured or ["exact epilogue stages are unavailable"]))}
    form, form_problems = _source_form(root, node, stages, final, buffers)
    if form is None:
        return {**base, "classification": "uncaptured", "reasons": form_problems}
    return {**base, "classification": "capability_pending", "capability_form": form,
            "capability_form_sha256": _digest(form), "reasons": []}


def _member_inventory(analysis: Mapping[str, Any], identity: Mapping[str, Any],
                      candidate_sha256: str, member_index: int,
                      capability_forms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    parts, problems = _analysis_parts(analysis, candidate_sha256)
    member = {
        "member_index": member_index,
        "identity": dict(identity),
        "identity_sha256": _digest(identity),
        "analysis_location": "/analysis" if member_index == 0 else
                             f"/portfolio/members/{member_index}/analysis",
        "analysis_sha256": _digest(analysis),
        "status": "not_ready",
        "source_operation_ids": [],
        "eligible": [],
        "already_narrow": [],
        "missing_representation": [],
        "missing_capability": [],
        "residual_second_operand": [],
        "float_or_unsupported_stage": [],
        "uncaptured": [],
        "ownership_failures": [],
        "problems": list(problems),
    }
    if not parts:
        member["inventory_sha256"] = _digest(member)
        return member
    ownership_broken = bool(parts["missing_owners"] or any(
        "owner" in reason or "ownership" in reason for reason in problems))
    if ownership_broken:
        for raw in parts["placement"].get("contractions", []):
            member["ownership_failures"].append({
                "producer_source_operation_id": raw.get("source_op_index")
                if isinstance(raw, Mapping) else None,
                "classification": "ownership_failure",
                "reasons": ["full source graph ownership is incomplete or contradictory"],
            })
        member["problems"] = sorted(set(member["problems"]))
        member["inventory_sha256"] = _digest(member)
        return member
    parts, metadata_roots, metadata_problems = _source_plan_adapter(parts)
    member["problems"].extend(metadata_problems)
    if not parts:
        member["problems"] = sorted(set(member["problems"]))
        member["inventory_sha256"] = _digest(member)
        return member
    _producer, uses, link_problems = _graph_links(parts)
    member["problems"].extend(link_problems)
    seen: set[int] = set()
    for raw in parts["placement"].get("contractions", []):
        source_index = raw.get("source_op_index") if isinstance(raw, Mapping) else None
        if type(source_index) is int and source_index in seen:
            member["uncaptured"].append({
                "producer_source_operation_id": source_index,
                "classification": "uncaptured",
                "reasons": ["contraction placement repeats one source operation"],
            })
            continue
        if type(source_index) is int:
            seen.add(source_index)
        if not isinstance(raw, Mapping):
            member["uncaptured"].append({
                "producer_source_operation_id": None, "classification": "uncaptured",
                "reasons": ["contraction placement row is malformed"],
            })
            continue
        site, reasons = _site(raw, parts)
        if site is None:
            node = (parts["nodes"][source_index] if type(source_index) is int
                    and 0 <= source_index < len(parts["nodes"]) else {})
            output_dtypes = [parts["buffers"].get(name, {}).get("dtype")
                             for name in node.get("outputs", [])]
            bucket = ("float_or_unsupported_stage" if any(_floating(dtype)
                                                           for dtype in output_dtypes)
                      else "uncaptured")
            member[bucket].append({
                "producer_source_operation_id": source_index,
                "classification": bucket,
                "reasons": reasons,
            })
            continue
        if site["ownership_evidence"]["declared_task_kind"] == "host":
            member["uncaptured"].append({
                "producer_source_operation_id": source_index,
                "classification": "uncaptured",
                "reasons": ["contraction producer is not owned by an accelerator task"],
            })
            continue
        traced = _trace(site, parts, uses, set(parts["program"].get("results", [])))
        classification = traced.pop("classification")
        source_metadata = metadata_roots.get(source_index)
        metadata_reasons = (source_metadata.get("reasons", [])
                            if isinstance(source_metadata, Mapping) else [])
        representation_reasons = sorted({reason for reason in traced.get("reasons", [])
                                         if "encoding is absent" in reason
                                         or "layout is absent" in reason})
        if representation_reasons:
            classification = "missing_representation"
            traced["reasons"] = [
                "one or more exact epilogue buffers have no verified physical representation",
                *representation_reasons,
            ]
        elif classification in {"uncaptured", "float_or_unsupported_stage"} \
                and isinstance(metadata_reasons, list):
            traced["reasons"] = sorted(set([*traced.get("reasons", []),
                                             *[reason for reason in metadata_reasons
                                               if isinstance(reason, str) and reason]]))
        if classification == "capability_pending":
            proof = capability_forms.get(traced["capability_form_sha256"])
            if proof is None:
                classification = "missing_capability"
                traced["reasons"] = [
                    "no pinned target proof matches the exact stage/operand/encoding/readout form"]
            else:
                classification = "eligible"
                traced["capability_proof_id"] = proof["id"]
                traced["capability_proof_sha256"] = _digest(proof)
                traced["proof_scope"] = (
                    "exact target form availability; source rewrite equivalence remains mandatory")
                traced["not_proven"] = [
                    "replacement semantic equivalence", "emitted work deletion",
                    "runtime correctness", "cycle improvement",
                ]
        traced["classification"] = classification
        member[classification].append(traced)

    for raw in parts["placement"].get("unresolved", []):
        member["uncaptured"].append({
            "producer_source_operation_id": raw.get("source_op_index")
            if isinstance(raw, Mapping) else None,
            "classification": "uncaptured",
            "reasons": [raw.get("mac_basis") if isinstance(raw, Mapping)
                        and isinstance(raw.get("mac_basis"), str) else
                        "contraction observer left the source operation unresolved"],
        })
    for key in ("eligible", "already_narrow", "missing_representation", "missing_capability",
                "residual_second_operand", "float_or_unsupported_stage", "uncaptured",
                "ownership_failures"):
        member[key].sort(key=lambda row: (-1 if row["producer_source_operation_id"] is None
                                         else row["producer_source_operation_id"]))
    member["source_operation_ids"] = sorted({source_index for chain in member["eligible"]
                                              for source_index in chain["source_operation_ids"]})
    member["problems"] = sorted(set(member["problems"]))
    member["status"] = (
        "not_ready" if member["problems"] else
        "eligible_sites_bound" if member["eligible"] else "no_eligible_sites"
    )
    member["bindings"] = {field: parts["binding"][field] for field in _PIN_FIELDS}
    member["inventory_sha256"] = _digest(member)
    return member


def inventory_portfolio_narrow_epilogues(
        record: Mapping[str, Any], *, capability_evidence: Mapping[str, Any] | None = None,
        capability_evidence_file_sha256: str | None = None,
        iteration_record_sha256: str | None = None,
) -> dict[str, Any]:
    """Derive exactly classified narrow-readout sites from a static portfolio iteration."""
    result = {
        "schema": INVENTORY_SCHEMA,
        "mechanism_id": MECHANISM_ID,
        "status": "not_ready",
        "candidate_sha256": record.get("candidate_sha256") if isinstance(record, Mapping) else None,
        "portfolio_sha256": None,
        "iteration_record_sha256": iteration_record_sha256,
        "iteration_payload_sha256": _digest(record) if isinstance(record, Mapping) else None,
        "capability_evidence_file_sha256": capability_evidence_file_sha256,
        "capability_evidence_sha256": capability_evidence.get("sha256")
        if isinstance(capability_evidence, Mapping) else None,
        "ordered_portfolio": [],
        "members": [],
        "problems": [],
        "proof_scope": "static source-site and exact target-capability-form assignment only",
        "not_proven": [
            "replacement semantic equivalence", "emitted work deletion",
            "runtime correctness", "performance improvement",
        ],
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
    forms, capability_problems = _capability_forms(
        record, analyses, capability_evidence, capability_evidence_file_sha256)
    result["problems"].extend(capability_problems)
    for index, (identity, analysis) in enumerate(zip(identities, analyses, strict=True)):
        result["members"].append(_member_inventory(
            analysis, identity, record["candidate_sha256"], index, forms))
    if any(member["status"] == "not_ready" for member in result["members"]):
        result["problems"].append("one or more portfolio member inventories are not ready")
    opportunities = sum(len(member["eligible"]) for member in result["members"])
    result["problems"] = sorted(set(result["problems"]))
    result["status"] = (
        "not_ready" if result["problems"] else
        "ready_for_work_order" if opportunities else "no_eligible_sites"
    )
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
    selected = [row for row in contract.get("existing_symbols", [])
                if isinstance(row, Mapping) and row.get("surface_id") in _MECHANISM_SURFACES]
    found = {row.get("surface_id") for row in selected}
    if _REQUIRED_SURFACES - found:
        raise ValueError("edit contract lacks required narrow-epilogue surfaces")
    selectors: list[dict[str, Any]] = []
    paths: set[str] = set()
    seen: set[tuple[str, str, str | None]] = set()
    for row in selected:
        path, symbol = row.get("path"), row.get("symbol")
        if not isinstance(path, str) or not isinstance(symbol, str):
            raise ValueError("narrow-epilogue compiler surface identity is malformed")
        source = candidate / path
        if source.is_symlink() or not source.is_file():
            raise ValueError("narrow-epilogue compiler surface source is missing")
        kind = _definition_kinds(source.read_text()).get(symbol)
        if kind not in {"class", "function", "method"}:
            raise ValueError("narrow-epilogue compiler surface is not an exact AST unit")
        key = kind, path, symbol
        if key not in seen:
            selectors.append({"kind": kind, "path": path, "symbol": symbol})
            seen.add(key)
            paths.add(path)
    for path in sorted(paths):
        selectors.append({"kind": "imports", "path": path})
    for extension in contract.get("helper_extensions", []):
        if (isinstance(extension, Mapping)
                and set(extension.get("surface_ids", [])) & _MECHANISM_SURFACES):
            selectors.append({"kind": "helper", "directory": extension["directory"]})
    selectors.sort(key=lambda row: (row["kind"], row.get("path", row.get("directory", "")),
                                    row.get("symbol", "")))
    catalog = {
        "schema": "compiler_mechanism_catalog_v1",
        "contract_sha256": contract["sha256"],
        "mechanisms": [{"id": MECHANISM_ID, "selectors": selectors}],
    }
    catalog["sha256"] = _digest(catalog)
    return validate_mechanism_catalog(catalog, candidate, contract)


def build_narrow_epilogue_mechanism_documents(
        record: Mapping[str, Any], *, candidate: Path, expected_candidate_sha256: str,
        iteration_record_sha256: str, capability_evidence: Mapping[str, Any],
        capability_evidence_file_sha256: str,
) -> dict[str, Any]:
    """Build, without writing, the exact contract/catalog/inventory/work-order documents."""
    candidate = Path(candidate)
    if (not candidate.is_absolute() or candidate.resolve() != candidate or candidate.is_symlink()
            or not candidate.is_dir() or not _pin(expected_candidate_sha256)
            or hash_tree(candidate)["sha256"] != expected_candidate_sha256
            or record.get("candidate_sha256") != expected_candidate_sha256):
        raise ValueError("explicit candidate path/hash does not match the static iteration")
    if not _pin(iteration_record_sha256) or not _pin(capability_evidence_file_sha256):
        raise ValueError("static iteration and capability evidence require exact raw SHA-256 pins")
    protocol_problems: list[str] = []
    _identities, analyses = _ordered_analyses(record, protocol_problems)
    if protocol_problems:
        raise ValueError("static iteration portfolio is not exact: " + "; ".join(protocol_problems))
    contract = _embedded_edit_contract(record, analyses, candidate)
    inventory = inventory_portfolio_narrow_epilogues(
        record,
        capability_evidence=capability_evidence,
        capability_evidence_file_sha256=capability_evidence_file_sha256,
        iteration_record_sha256=iteration_record_sha256,
    )
    if inventory["status"] != "ready_for_work_order":
        raise ValueError("static iteration has no capability-proven narrow-epilogue opportunity")
    catalog = _mechanism_catalog(contract, candidate)
    site_bindings = []
    for member in inventory["members"]:
        bindings = member["bindings"]
        site_bindings.append({
            "capsule": member["identity"]["capsule"],
            "capsule_sha256": member["identity"]["capsule_sha256"],
            "compiler_sha256": bindings["compiler_sha256"],
            "source_sha256": bindings["source_sha256"],
            "plan_digest": bindings["plan_digest"],
            "candidate_command_buffer_sha256": bindings["command_buffer_sha256"],
            "candidate_lowered_sha256": bindings["lowered_sha256"],
            "status": member["status"],
            "source_operation_ids": member["source_operation_ids"],
            "chains": member["eligible"],
            "inventory": {
                "schema": INVENTORY_SCHEMA,
                "member_index": member["member_index"],
                "analysis_location": member["analysis_location"],
                "analysis_sha256": member["analysis_sha256"],
                "member_inventory_sha256": member["inventory_sha256"],
                "portfolio_inventory_sha256": inventory["sha256"],
                "iteration_record_sha256": iteration_record_sha256,
                "capability_evidence_file_sha256": capability_evidence_file_sha256,
                "already_narrow_count": len(member["already_narrow"]),
                "missing_representation_count": len(member["missing_representation"]),
                "missing_capability_count": len(member["missing_capability"]),
                "residual_second_operand_count": len(member["residual_second_operand"]),
                "float_or_unsupported_stage_count": len(member["float_or_unsupported_stage"]),
                "uncaptured_count": len(member["uncaptured"]),
                "ownership_failure_count": len(member["ownership_failures"]),
            },
        })
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
    return {"edit_contract": contract, "catalog": catalog,
            "inventory": inventory, "work_order": work_order}


def _write_once(path: Path, document: Mapping[str, Any]) -> str:
    raw = (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
    path.chmod(0o444)
    return _raw_sha256(raw)


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(
        description="Seal one capability-proven narrow-epilogue Phase-2 catalog/work order")
    parser.add_argument("iteration", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("capability_evidence", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iteration-sha256", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--capability-sha256", required=True)
    args = parser.parse_args(argv)
    for label, path in (("iteration", args.iteration),
                        ("capability evidence", args.capability_evidence)):
        if (not path.is_absolute() or path.resolve() != path or path.is_symlink()
                or not path.is_file() or path.stat().st_mode & 0o222):
            parser.error(f"{label} must be an absolute, read-only, non-symlink regular file")
    iteration_raw = args.iteration.read_bytes()
    capability_raw = args.capability_evidence.read_bytes()
    if not _pin(args.iteration_sha256) or _raw_sha256(iteration_raw) != args.iteration_sha256:
        parser.error("iteration raw SHA-256 pin does not match")
    if not _pin(args.capability_sha256) or _raw_sha256(capability_raw) != args.capability_sha256:
        parser.error("capability evidence raw SHA-256 pin does not match")
    try:
        record, capability = json.loads(iteration_raw), json.loads(capability_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        parser.error(f"iteration or capability evidence is not valid JSON: {exc}")
    try:
        documents = build_narrow_epilogue_mechanism_documents(
            record,
            candidate=args.candidate,
            expected_candidate_sha256=args.candidate_sha256,
            iteration_record_sha256=args.iteration_sha256,
            capability_evidence=capability,
            capability_evidence_file_sha256=args.capability_sha256,
        )
        if hash_tree(args.candidate)["sha256"] != args.candidate_sha256:
            raise ValueError("candidate changed while deriving the sealed work order")
        args.output.mkdir(mode=0o755, parents=False, exist_ok=False)
        names = {
            "compiler_edit_contract.json": documents["edit_contract"],
            "mechanism_catalog.json": documents["catalog"],
            "mechanism_work_order.json": documents["work_order"],
            "source_site_inventory.json": documents["inventory"],
        }
        artifacts = {name: {"sha256": _write_once(args.output / name, document)}
                     for name, document in names.items()}
        receipt = {
            "schema": "sealed_narrow_epilogue_work_order_receipt_v1",
            "mechanism_id": MECHANISM_ID,
            "candidate_sha256": args.candidate_sha256,
            "iteration_record_sha256": args.iteration_sha256,
            "capability_evidence_file_sha256": args.capability_sha256,
            "capability_evidence_sha256": capability.get("sha256"),
            "inventory_sha256": documents["inventory"]["sha256"],
            "catalog_sha256": documents["catalog"]["sha256"],
            "work_order_sha256": documents["work_order"]["sha256"],
            "artifacts": dict(artifacts),
        }
        receipt["sha256"] = _digest(receipt)
        artifacts["receipt.json"] = {"sha256": _write_once(args.output / "receipt.json", receipt)}
    except (FileExistsError, OSError, ValueError) as exc:
        parser.error(str(exc))
    report = {
        "status": "ready_for_authoring",
        "mechanism_id": MECHANISM_ID,
        "output": str(args.output),
        "eligible_counts": [len(member["eligible"])
                            for member in documents["inventory"]["members"]],
        "already_narrow_counts": [len(member["already_narrow"])
                                  for member in documents["inventory"]["members"]],
        "missing_representation_counts": [len(member["missing_representation"])
                                           for member in documents["inventory"]["members"]],
        "missing_capability_counts": [len(member["missing_capability"])
                                      for member in documents["inventory"]["members"]],
        "residual_second_operand_counts": [len(member["residual_second_operand"])
                                           for member in documents["inventory"]["members"]],
        "float_or_unsupported_stage_counts": [len(member["float_or_unsupported_stage"])
                                              for member in documents["inventory"]["members"]],
        "uncaptured_counts": [len(member["uncaptured"])
                              for member in documents["inventory"]["members"]],
        "artifacts": artifacts,
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
