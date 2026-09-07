"""Find short source-issued movement/compute contexts, without pretending they already ran.

Unlike an isolated primitive, these windows keep operand loads AND other queued loads inside the
proposed timer. The first supported family is a contiguous load/configuration prefix ending in one
initialized overwrite compute pair. Source order, dependency extents and competing commands are
preserved. This is a probe construction request, not physical overlap or cycle evidence.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .instruction_motif import initialized_compute_primitives


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def extract_queued_movement_context(
    trace: Mapping[str, Any], *, target: str, artifact_sha256: str, artifact_text: str,
    command_buffer: Mapping[str, Any] | None = None, parsed_module: Any = None,
    max_commands: int = 32, max_motifs: int = 4,
) -> dict[str, Any]:
    """Extract bounded context windows from the host's retained source/decoder pair.

    The host analyzer owns ``trace`` and ``parsed_module`` from decoding ``artifact_text``; callers
    must not substitute candidate-provided summaries. The actual text hash and canonical decoded
    trace digest accompany every result. Missing task ownership, external address placement, physical
    resources or entry-state evidence remain explicit; none is replaced with a synthetic rate.
    """
    from merlin.kernels.decode.rocc import funct_table_for
    from merlin.kernels.endpoints import endpoints_for
    from merlin.targetgen.rocc import decode

    actual_sha = hashlib.sha256(artifact_text.encode()).hexdigest()
    if actual_sha != artifact_sha256:
        raise ValueError("context extraction artifact hash differs from retained emitted bytes")
    if not 3 <= max_commands <= 64 or not 1 <= max_motifs <= 16:
        raise ValueError("context extraction requires bounded command and motif limits")
    rows = trace.get("instructions")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("context extraction requires a complete decoded instruction list")
    receipt: dict[str, Any] = {
        "schema": "queued_movement_context_candidates_v1", "status": "UNRESOLVED",
        "artifact_sha256": actual_sha, "decoded_trace_sha256": _digest(trace),
        "binding_method": "host-retained emitted bytes and their decoder output",
        "motifs": [], "unsupported": [], "considered_primitives": 0,
        "max_commands": max_commands, "max_motifs": max_motifs,
        "executed": False, "full_model_execution_allowed": False,
        "calibration_admissible": False, "in_context_cycles": None,
        "licence": "bounded source-context extraction only; no measured contention or physical overlap",
    }
    isa = decode.isa_constants(target)
    configs = set(isa.get("CONFIG_SUBTYPE", {}).values())
    table = funct_table_for(target)
    names = {int(code): name for code, name in (table.get("names") or {}).items()}
    endpoints = endpoints_for(target)
    asm = [op for op in parsed_module.walk() if op.name == "llvm.inline_asm"] if parsed_module else []
    if parsed_module is not None and len(asm) != len(rows):
        raise ValueError("retained module and decoded trace have different instruction counts")
    if parsed_module is not None:
        if decode.decode_module(parsed_module, target=target).get("instructions") != rows:
            raise ValueError("context trace differs from decoding the retained emitted module")
        receipt["binding_method"] = "verified emitted-byte hash and redecoded retained host module; no reparse"
    buffer = command_buffer or {}
    tasks = {task["task_index"]: task for task in
             buffer.get("params", {}).get("global_program_plan", {}).get("tasks", [])
             if isinstance(task, Mapping) and "task_index" in task}
    abi = buffer.get("kernel_abi", {}).get("args", [])
    tensors = buffer.get("tensors", {})

    def owner(index):
        if not asm:
            return None
        value = asm[index].attributes.get("merlin.global_task")
        return getattr(getattr(value, "value", None), "data", None)

    def config_before(index, subtype):
        return next((i for i in range(index - 1, -1, -1)
                     if rows[i].get("class") in configs
                     and rows[i].get("decoded", {}).get("subtype") == subtype), None)

    for primitive in initialized_compute_primitives(trace, target=target):
        receipt["considered_primitives"] += 1
        if len(receipt["motifs"]) >= max_motifs:
            break
        missing = list(primitive["missing"])
        inputs = primitive["initialization_provenance"]
        end = primitive["instruction_indices"][-1]
        start = min((item["producer_index"] for item in inputs), default=end)
        indices = list(range(start, end + 1))
        if not inputs or len(indices) > max_commands:
            missing.append("initialization prefix is absent or exceeds short command budget")
        # Do not reduce a larger compute sequence or a host/device crossing to its final pair.
        if any(rows[i].get("class") not in configs and "spad_addr" not in rows[i].get("decoded", {})
               for i in indices[:-2]):
            missing.append("prefix contains another compute, readout, completion or unknown state effect")
        load_indices = [i for i in indices[:-2] if "spad_addr" in rows[i].get("decoded", {})]
        consumed = {item["producer_index"] for item in inputs}
        competing = [i for i in load_indices if i not in consumed]
        if not competing:
            missing.append("no additional queued movement remains beyond the compute operand producers")
        if missing:
            if len(receipt["unsupported"]) < 8:
                receipt["unsupported"].append({"compute_index": end, "missing": sorted(set(missing))})
            continue

        state_missing = []
        context_missing = [
            "actual runtime operand base alignment, external memory mapping and cache state are unbound",
            "physical queue depth, issue backpressure and shared-port topology are not established",
            "following model commands are omitted; future traffic can affect completion of this prefix",
            "target-specific executable context setup/body/readback and numerical oracle are not constructed",
            "decoded entry completion has not been qualified against the target completion contract",
        ]
        entry = next((i for i in range(start - 1, -1, -1) if rows[i].get("class") == "FENCE"), None)
        command_empty = entry is not None and all(rows[i].get("class") in configs
                                                for i in range(entry + 1, start))
        if not command_empty:
            state_missing.append("source entry may contain queued commands preceding the chosen load prefix")
        initial_configs = {kind: config_before(start, kind) for kind in ("EX", "LD")}
        if any(index is None for index in initial_configs.values()):
            state_missing.append("execution or load configuration at the window entry is unresolved")

        task_owners = {owner(i) for i in indices}
        task_index = next(iter(task_owners)) if len(task_owners) == 1 else None
        task = tasks.get(task_index) if task_index is not None else None
        if not task or task_index < 0:
            state_missing.append("one compiler-owned task/source mapping is not available for the entire window")
        if asm and any(asm[i].parent_block() is not asm[start].parent_block() for i in indices):
            state_missing.append("source command window crosses control-flow blocks")
        elif asm:
            from .host_cfg_activity import _category
            block_ops = list(asm[start].parent_block().ops)
            first, last = block_ops.index(asm[start]), block_ops.index(asm[end])
            unexpected = sorted({op.name for op in block_ops[first:last + 1]
                                 if op.name != "llvm.inline_asm" and _category(op.name) not in
                                 {"constant", "address", "integer_arithmetic", "conversion", "comparison"}})
            if unexpected:
                state_missing.append(f"source window contains non-command host work: {unexpected}")

        transfers, dependencies, resources = [], [], []
        extents: dict[int, set[int]] = {}
        for index in load_indices:
            decoded = rows[index]["decoded"]
            address, count, cols = (decoded.get(key) for key in ("spad_addr", "rows", "cols"))
            if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
                   for value in (count, cols)) or not isinstance(address, int) or address < 0:
                state_missing.append(f"movement {index} lacks finite positive physical row extents")
                continue
            extents[index] = set(range(address, address + count))
            external = decoded.get("dram", {})
            argument = external.get("arg_index")
            tensor = (abi[argument].get("tensor") if isinstance(argument, int) and 0 <= argument < len(abi)
                      and isinstance(abi[argument], Mapping) else None)
            tensor_info = tensors.get(tensor, {})
            if external.get("kind") != "argbase" or not isinstance(external.get("offset"), int):
                state_missing.append(f"movement {index} has unresolved external pointer and byte offset")
            if not tensor_info.get("dtype"):
                state_missing.append(f"movement {index} has no ABI-bound operand dtype")
            config = config_before(index, "LD")
            config_payload = rows[config].get("decoded", {}) if config is not None else {}
            if not isinstance(config_payload.get("stride"), int) or config_payload["stride"] <= 0:
                state_missing.append(f"movement {index} has no decoded positive load stride")
            transfers.append({"instruction_index": index, "physical_rows": [address, address + count],
                              "rows": count, "cols": cols, "external_pointer": dict(external),
                              "tensor": tensor, "dtype": tensor_info.get("dtype"),
                              "load_configuration_index": config, "load_configuration": config_payload,
                              "compute_operand_producer": index in consumed})
            for earlier, slots in extents.items():
                if earlier < index and slots.intersection(extents[index]):
                    dependencies.append({"before": earlier, "after": index, "kind": "WAW",
                                         "state": "decoded operand-store rows"})
        for item in inputs:
            dependencies.append({"before": item["producer_index"], "after": end, "kind": "RAW",
                                 "operand": item["operand"],
                                 "physical_rows": [item["address"], item["address"] + item["rows"]]})
        for index in indices:
            identity = names.get(rows[index].get("funct"), "")
            bound = [{"endpoint": endpoint.name, "declared_engine": endpoint.engine,
                      "roles": list(endpoint.roles_of(identity)), "source": endpoint.source}
                     for endpoint in endpoints if endpoint.roles_of(identity)]
            resources.append({"instruction_index": index, "identity": identity, "endpoints": bound})
            if not bound:
                context_missing.append(f"instruction {index} has no target-derived endpoint role binding")
        used_rows = set().union(*(extents.get(index, set()) for index in consumed))
        disjoint = [index for index in competing if index in extents and not extents[index].intersection(used_rows)]
        semantics = [{key: rows[index].get(key) for key in ("class", "funct", "rs1", "rs2", "decoded")}
                     for index in indices]
        motif = {
            "schema": "queued_movement_context_v1", "artifact_sha256": actual_sha,
            "instruction_indices": indices, "instruction_semantics": semantics,
            "primitive_domain_digest": primitive["domain_digest"],
            "queued_competing_movement_indices": competing,
            "competing_movements_disjoint_from_operand_rows": disjoint,
            "initial_configuration_indices": initial_configs,
            "initial_configurations": {kind: rows[index] if index is not None else None
                                       for kind, index in initial_configs.items()},
            "source_entry_completion_index": entry, "command_empty_entry_observed": command_empty,
            "source_completion_semantics_proven": False,
            "task_index": task_index, "source_op_indices": task.get("source_op_indices") if task else None,
            "task_reads": task.get("reads") if task else None,
            "task_writes": task.get("writes") if task else None,
            "transfers": transfers, "dependencies": dependencies, "resource_bindings": resources,
            "dependency_scope": "explicit producer RAW and transfer WAW rows; not a complete reorder legality proof",
            "capacity_regime": "UNKNOWN: decoded row addresses need target-specific physical store binding",
            "state_missing": sorted(set(state_missing)), "context_missing": sorted(set(context_missing)),
            "proposed_timer": "all listed loads/config changes plus overwrite compute and final completion",
            "excluded_from_timer": "host input initialization, warm complete context, output readback and verification",
            "warmup_runs": 1, "measured_runs": 1, "maximum_wall_seconds": 600,
            "calibration_admissible": False, "in_context_cycles": None,
            "overlap_observed": None, "physical_parallelism_proven": False,
        }
        motif["context_shape_sha256"] = _digest({"instructions": semantics,
                                                  "initial_configurations": motif["initial_configurations"],
                                                  "target_isa": isa})
        receipt["motifs"].append(motif)
    if receipt["motifs"]:
        receipt["status"] = "STRUCTURAL_CONTEXT_CANDIDATES"
    return receipt
