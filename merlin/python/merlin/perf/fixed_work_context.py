"""Project a frozen short workload through an actual source-command permutation.

This narrow adapter supports compute/preload hoisting across disjoint later operand loads. It
retains every selected load/configuration/compute from the anchor, including loads now issued after
the first compute. It never calls a shortened first-compute prefix an equivalent benchmark.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from copy import deepcopy
import hashlib
import json
from typing import Any, Mapping


def project_fixed_work_context(before_artifacts: Mapping[str, Any], after_artifacts: Mapping[str, Any],
                               anchor_context: Mapping[str, Any], *, target: str) -> tuple[dict, dict]:
    from merlin.targetgen.rocc import decode
    from .deps.rocc import INHERITS_DESTINATION

    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
    sha = lambda value: hashlib.sha256(canonical(value).encode()).hexdigest()
    artifacts, traces, operations = [], [], []
    for captured in (before_artifacts, after_artifacts):
        text = captured["lowered_text"]
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest != captured["candidate_lowered_sha256"]:
            raise ValueError("fixed-work projection source hash differs from retained emitted bytes")
        module = captured.get("parsed_lowered_module")
        if module is None:
            if len(text.encode()) > 2_000_000:
                raise ValueError("fixed-work projection exceeds host parsing byte policy")
            module = decode._parse_module(text)
        if module is None:
            raise ValueError("fixed-work projection source does not parse")
        decoded = decode.decode_module(module, target=target)["instructions"]
        if decoded != captured["decoded_trace"]["instructions"]:
            raise ValueError("fixed-work projection trace differs from retained source module")
        if not decoded or len(decoded) > 4096 or any(row["class"] == "UNKNOWN" for row in decoded):
            raise ValueError("fixed-work projection has absent/unknown/unbounded source commands")
        traces.append(decoded)
        operations.append([op for op in module.walk() if op.name == "llvm.inline_asm"])
        artifacts.append(digest)
    if artifacts[0] != anchor_context["artifact_sha256"] or artifacts[0] == artifacts[1]:
        raise ValueError("fixed-work anchor must bind the preceding, actually changed full-model artifact")
    before_cb, after_cb = (row["command_buffer"] for row in (before_artifacts, after_artifacts))
    if (before_cb["kernel_abi"] != after_cb["kernel_abi"] or before_cb["tensors"] != after_cb["tensors"]):
        raise ValueError("fixed-work projection changed the source ABI or tensor footprint")
    owner = lambda op: getattr(getattr(op.attributes.get("merlin.global_task"), "value", None), "data", None)

    def identity(row, op):
        if owner(op) is None:
            raise ValueError("fixed-work source command has no compiler task ownership")
        return canonical({"task": owner(op), **{name: row.get(name) for name in ("class", "funct", "rs1", "rs2")},
                          "asm": op.asm_string.data, "constraints": op.constraints.data,
                          "side_effects": op.has_side_effects is not None})

    keys = [[identity(row, op) for row, op in zip(trace, ops, strict=True)]
            for trace, ops in zip(traces, operations, strict=True)]
    if Counter(keys[0]) != Counter(keys[1]):
        raise ValueError("fixed-work projection changed commands, payloads, counts or task owners")
    compute_classes = set(INHERITS_DESTINATION) | set(INHERITS_DESTINATION.values())
    for predicate in (lambda row: row["class"] in compute_classes,
                      lambda row: row["class"] not in compute_classes):
        if [key for key, row in zip(keys[0], traces[0]) if predicate(row)] != [
                key for key, row in zip(keys[1], traces[1]) if predicate(row)]:
            raise ValueError("fixed-work projection changed the compute or non-compute stream order")
    occurrences = defaultdict(deque)
    for index, key in enumerate(keys[0]):
        occurrences[key].append(index)
    permutation = [occurrences[key].popleft() for key in keys[1]]
    position = {before: after for after, before in enumerate(permutation)}

    reads, weights = {}, set()
    for index, row in enumerate(traces[0]):
        payload = row.get("decoded", {})
        if row["class"] in INHERITS_DESTINATION.values():
            address = payload.get("weight_spad")
            if not isinstance(address, int) or address < 0:
                raise ValueError("staged weight address is unresolved")
            if address != decode.GARBAGE:
                count = decode._pack_fields(row["rs1"]["raw"])["rows"]
                weights = set(range(address, address + count))
                reads[index] = set(weights)
            else:
                reads[index] = set()
        elif row["class"] in INHERITS_DESTINATION:
            address = payload.get("a_spad")
            if not isinstance(address, int) or address < 0 or address == decode.GARBAGE or not weights:
                raise ValueError("compute activation address is unresolved")
            count = decode._pack_fields(row["rs1"]["raw"])["rows"]
            reads[index] = set(range(address, address + count)) | weights
    isa = decode.isa_constants(target)
    namespace = isa.get("ACC_I8")
    if not isinstance(namespace, int):
        raise ValueError("target accumulator/operand address namespace is unresolved")
    crossings = []
    for earlier in range(len(traces[0])):
        for later in range(earlier + 1, len(traces[0])):
            if position[earlier] < position[later]:
                continue
            load = traces[0][earlier].get("decoded", {})
            address, count = load.get("spad_addr"), load.get("rows")
            if (later not in reads or not isinstance(address, int) or address < 0 or address & namespace
                    or not isinstance(count, int) or count <= 0
                    or owner(operations[0][earlier]) != owner(operations[0][later])):
                raise ValueError("reordering is not same-task compute hoisting across an operand load")
            writes = set(range(address, address + count))
            if writes & reads[later]:
                raise ValueError("reordered load overlaps moved compute/preload operand rows")
            crossings.append([earlier, later])
    if not crossings:
        raise ValueError("fixed-work comparison has no actual supported scheduling change")
    anchor = list(anchor_context["instruction_indices"])
    if not 3 <= len(anchor) <= 32 or anchor_context.get("state_missing"):
        raise ValueError("fixed-work anchor is unbounded or has unresolved initialization")
    fields = ("class", "funct", "rs1", "rs2", "decoded")
    before_semantics = [{key: traces[0][index].get(key) for key in fields} for index in anchor]
    if before_semantics != anchor_context["instruction_semantics"]:
        raise ValueError("fixed-work anchor payload differs from the preceding emitted artifact")
    selected = sorted(position[index] for index in anchor)
    projected = deepcopy(dict(anchor_context))
    projected.update({"artifact_sha256": artifacts[1], "instruction_indices": selected,
                      "compute_instruction_indices": [position[index] for index in anchor[-2:]],
                      "instruction_semantics": [{key: traces[1][index].get(key) for key in fields} for index in selected],
                      "queued_competing_movement_indices": [position[index] for index in
                                                             anchor_context["queued_competing_movement_indices"]],
                      "initial_configuration_indices": {kind: position[index] for kind, index in
                                                          anchor_context["initial_configuration_indices"].items()}})
    projected["initial_configurations"] = {kind: traces[1][index] for kind, index in
                                           projected["initial_configuration_indices"].items()}
    projected["context_shape_sha256"] = sha({"instructions": projected["instruction_semantics"],
                                             "initial_configurations": projected["initial_configurations"]})
    projected["context_missing"] = [*projected["context_missing"],
                                     "future compute commands omitted symmetrically in both fixed-work projections"]
    work_contract = {"source_task_index": anchor_context["task_index"],
                     "source_op_indices": anchor_context["source_op_indices"],
                     "abi_sha256": sha(before_cb["kernel_abi"]), "tensor_footprint_sha256": sha(before_cb["tensors"]),
                     "timed_command_multiset": sorted(canonical(row) for row in before_semantics),
                     "timed_command_count": len(anchor),
                     "competing_load_count": len(anchor_context["queued_competing_movement_indices"])}
    proof = {"schema": "controlled_fixed_work_projection_v1", "status": "same_work_projection_verified",
             "scope": "controlled_fixed_work_slice", "before_artifact_sha256": artifacts[0],
             "after_artifact_sha256": artifacts[1], "after_to_before_permutation": permutation,
             "before_timed_indices": anchor, "after_timed_indices": selected,
             "same_task_command_multisets": True, "compute_order_preserved": True,
             "noncompute_order_preserved": True, "disjoint_operand_row_crossings": crossings,
             "work_contract": work_contract, "work_contract_sha256": sha(work_contract),
             "future_computes_omitted_symmetrically": True, "global_cost_validated": False,
             "full_model_numerical_equivalence": "NOT_ESTABLISHED"}
    return projected, proof
