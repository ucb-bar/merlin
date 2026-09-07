"""Translation-accounting checks for an emitted mixed-program global plan.

The host rederives source operation identities, SSA dependencies and tensor boundaries, and checks
the actual lowered IR's task ownership. This is a structural compilation proof. Numeric equivalence
and timing require separately admitted reduced witnesses; annotations alone prove neither.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from typing import Any


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _source_has_multiply_accumulate(op: Any) -> bool:
    """Necessary source-work evidence, not an accelerator-placement proof.

    Provenance families survive rewrites onto pointwise epilogues; they cannot
    classify computation. Match an actual reduction and yielded MAC recurrence.
    More elaborate/fused recurrences remain unsupported by this narrow check.
    """
    from xdsl.ir import Operation
    if op.name in ("linalg.matmul", "linalg.batch_matmul", "linalg.quantized_matmul"):
        return True
    if op.name != "linalg.generic" or len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        return False
    props = {**op.attributes, **op.properties}
    iterator_attrs = props.get("iterator_types")
    if iterator_attrs is None:
        return False
    iterators = [getattr(getattr(item, "data", None), "value", None) for item in iterator_attrs]
    if "reduction" not in iterators or any(kind not in {"parallel", "reduction"} for kind in iterators):
        return False
    body = op.regions[0].blocks[0]
    terminator = body.last_op
    if (len(body.args) != 3 or terminator is None or terminator.name != "linalg.yield"
            or len(terminator.operands) != 1):
        return False
    addition = terminator.operands[0].owner
    if (not isinstance(addition, Operation) or addition.name not in {"arith.addi", "arith.addf"}
            or len(addition.operands) != 2 or body.args[2] not in addition.operands):
        return False
    operands = list(addition.operands)
    multiply = operands[1 - operands.index(body.args[2])].owner
    if (not isinstance(multiply, Operation) or multiply.name not in {"arith.muli", "arith.mulf"}
            or len(multiply.operands) != 2):
        return False

    def origin(value):
        seen = set()
        while isinstance(value.owner, Operation) and value.owner.name in {
                "arith.extsi", "arith.extui", "arith.extf"}:
            if value in seen or len(value.owner.operands) != 1:
                return None
            seen.add(value)
            value = value.owner.operands[0]
        return value

    return {origin(value) for value in multiply.operands} == set(body.args[:2])


def verify_compiler_global_plan(*, source_text: str, lowered_text: str,
                                command_buffer: Mapping[str, Any], candidate_sha256: str,
                                command_buffer_sha256: str | None = None,
                                parsed_lowered_module: Any = None,
                                prepared_source_analysis: Any = None) -> dict[str, Any]:
    """Bind complete source/task/IR coverage to exact candidate and artifact identities.

    A package that does not yet emit the protocol remains UNKNOWN. Invalid or incomplete evidence
    is refused. The returned proof scope intentionally excludes target arithmetic correctness.

    ``parsed_lowered_module`` is a host-only integration hook: the caller must have parsed the
    exact ``lowered_text`` in this invocation, and must not mutate that module. Sharing it with
    instruction decoding avoids reparsing large artifacts. Candidate-supplied objects or a cached
    module without an exact byte identity must never be passed here. Verification still runs.
    """
    from merlin.frontends.linalg_mlir import make_context, parse_mlir_text
    from merlin.xdsl_dialects.lowering.dispatch_program import lower_model_to_dispatch_program
    from merlin.xdsl_dialects.lowering.global_plan_emission import dispatch_digest

    params = command_buffer.get("params")
    params = params if isinstance(params, Mapping) else {}
    receipt = params.get("global_program_plan")
    if not isinstance(receipt, Mapping):
        return {"status": "UNKNOWN", "reason": "candidate emitted no global_program_plan"}
    if receipt.get("schema") != "mixed_program_plan_v1":
        return {"status": "UNKNOWN", "reason": "unsupported compiler global-plan protocol"}
    problems = []
    source_digest = hashlib.sha256(source_text.encode()).hexdigest()
    if receipt.get("source_sha256") != source_digest:
        problems.append("global plan is not bound to the current full-model source")
    if command_buffer.get("declined"):
        problems.append("candidate declined whole-model compilation")
    if len(candidate_sha256) != 64 or any(c not in "0123456789abcdef" for c in candidate_sha256):
        problems.append("candidate identity is not a SHA-256 digest")
    if prepared_source_analysis is not None:
        if (getattr(prepared_source_analysis, "source_sha256", None) != source_digest
                or getattr(prepared_source_analysis, "parsed_module", None) is None
                or getattr(prepared_source_analysis, "graph", None) is None
                or dispatch_digest(prepared_source_analysis.graph)
                != getattr(prepared_source_analysis, "logical_dispatch_digest", None)):
            raise ValueError("prepared source analysis changed or does not match source bytes")
        source = prepared_source_analysis.parsed_module
        graph = prepared_source_analysis.graph
    else:
        source = parse_mlir_text(source_text)
        _, graph = lower_model_to_dispatch_program(source, prune=False)
    functions = [op for op in source.body.block.ops
                 if op.name == "func.func" and op.sym_name.data == graph.entry]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        return {"status": "refused", "problems": ["source must have one single-block model entry"]}
    block = functions[0].body.blocks[0]
    ops = [op for op in block.ops if op.name != "func.return"]
    returns = [op for op in block.ops if op.name == "func.return"]
    if len(returns) != 1 or len(ops) != len(graph.nodes):
        return {"status": "refused", "problems": ["source and outlined graph operation order disagree"]}
    if receipt.get("source_op_count") != len(ops):
        problems.append("global plan source operation count differs from the actual full graph")
    tasks = receipt.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return {"status": "refused", "problems": problems + ["global plan has no task list"]}
    owner: dict[int, int] = {}
    task_ids = []
    declared_task_kinds: dict[int, str] = {}
    ranges: list[tuple[int, int]] = []
    for row in tasks:
        if not isinstance(row, Mapping):
            problems.append("malformed task record")
            continue
        ident = row.get("task_index")
        if isinstance(ident, bool) or not isinstance(ident, int) or ident < 0:
            problems.append("task index is not a nonnegative integer")
            continue
        task_ids.append(ident)
        indices = row.get("source_op_indices")
        if not isinstance(indices, list) or not indices:
            problems.append(f"task {ident} has no source operations")
            continue
        valid_indices = [index for index in indices
                         if isinstance(index, int) and not isinstance(index, bool)
                         and 0 <= index < len(ops)]
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(ops):
                problems.append(f"task {ident} references an absent source operation")
            elif index in owner:
                problems.append(f"source operation {index} is covered more than once")
            else:
                owner[index] = ident
        kind = row.get("kind")
        if not isinstance(kind, str) or not kind.strip():
            problems.append(f"task {ident} must declare a nonempty execution-kind label")
        else:
            declared_task_kinds[ident] = kind
            if kind in {"contraction", "convolution"} and not any(
                    _source_has_multiply_accumulate(ops[index]) for index in valid_indices):
                problems.append(f"task {ident} claims contraction work without a supported source MAC reduction")
        # A host task may legitimately implement a contraction. The target's
        # capability/offload policy is separate from source/CFG coverage. Neither
        # source MACs nor candidate kind labels prove which hardware executed it.
        start, end = row.get("instruction_start"), row.get("instruction_end")
        if (any(isinstance(v, bool) or not isinstance(v, int) for v in (start, end))
                or start < 0 or end <= start):
            problems.append(f"task {ident} has an invalid scheduled instruction range")
        else:
            ranges.append((start, end))
    if task_ids != list(range(len(tasks))):
        problems.append("task indices must uniquely enumerate actual emission order")
    if set(owner) != set(range(len(ops))):
        problems.append(f"source operations are not fully covered: {sorted(set(range(len(ops))) - owner.keys())}")
    producers = {name: index for index, node in enumerate(graph.nodes) for name in node.outputs}
    for index, node in enumerate(graph.nodes):
        if index not in owner:
            continue
        for name in node.inputs:
            producer = producers.get(name)
            if producer in owner and owner[producer] > owner[index]:
                problems.append(f"task order violates source dependency {producer}->{index}")

    count = receipt.get("schedule_instruction_count")
    for name in ("prologue_instruction_range", "epilogue_instruction_range"):
        pair = receipt.get(name)
        if (not isinstance(pair, list) or len(pair) != 2
                or any(isinstance(v, bool) or not isinstance(v, int) for v in pair)
                or pair[0] < 0 or pair[1] < pair[0]):
            problems.append(f"invalid {name}")
        elif pair[1] > pair[0]:
            ranges.append(tuple(pair))
    ordered_ranges = sorted(ranges)
    if (isinstance(count, bool) or not isinstance(count, int) or count <= 0
            or not ordered_ranges or ordered_ranges[0][0] != 0
            or ordered_ranges[-1][1] != count
            or any(left[1] != right[0] for left, right in zip(ordered_ranges, ordered_ranges[1:]))):
        problems.append("task and wrapper ranges do not exactly partition scheduled instructions")

    tensors = command_buffer.get("tensors")
    tensors = tensors if isinstance(tensors, Mapping) else {}
    value_tensors: dict[Any, str] = {}
    encodings = params.get("storage_encodings", {})
    if not isinstance(encodings, Mapping):
        problems.append("storage encodings must map declared tensor names to explicit contracts")
        encodings = {}
    if any(not isinstance(name, str) or name not in tensors for name in encodings):
        problems.append("storage encoding names an absent materialized tensor")
    if "storage_encodings" in params and (not encodings or set(encodings) != set(tensors)):
        problems.append("explicit storage encoding ABI must cover every materialized tensor")
    checked_encodings: dict[str, dict[str, Any]] = {}

    def bind(value, tensor):
        from xdsl.dialects.builtin import TensorType
        if not isinstance(tensor, str):
            problems.append("materialized tensor binding must be a name")
            return
        spec = tensors.get(tensor)
        if not isinstance(spec, Mapping) or not isinstance(value.type, TensorType):
            problems.append(f"materialized source value has no declared tensor {tensor!r}")
            return
        logical_shape = list(value.type.get_shape())
        dtype = str(value.type.get_element_type())
        if tensor in encodings:
            from .storage_encoding import GroupedAxesStorage
            try:
                encoding = GroupedAxesStorage.from_dict(encodings[tensor])
                if (list(encoding.logical_shape) != logical_shape
                        or list(encoding.physical_shape) != spec.get("shape")
                        or encoding.dtype != dtype or encoding.dtype != spec.get("dtype")):
                    raise ValueError("encoding differs from actual source or declared physical type")
                checked_encodings[tensor] = {
                    "contract": encoding.to_dict(),
                    "logical_strides_elements": list(encoding.logical_strides_elements),
                    "proof_scope": "declared bijective logical grouping and bounded storage address map",
                    "caller_materialization": "requires artifact-bound pack/view evidence",
                    "emitted_consumer_addressing": "requires artifact-bound address evidence",
                }
            except ValueError as exc:
                problems.append(f"tensor {tensor!r} has invalid storage encoding: {exc}")
        elif logical_shape != spec.get("shape") or dtype != spec.get("dtype"):
            problems.append(f"tensor {tensor!r} changes its source shape or dtype")
        previous = value_tensors.get(value)
        if previous is not None and previous != tensor:
            problems.append("source SSA value has conflicting materialized bindings")
        value_tensors[value] = tensor

    entries = receipt.get("entry_bindings")
    outputs = receipt.get("output_bindings")
    if not isinstance(entries, list) or len(entries) != len(block.args):
        problems.append("entry bindings do not cover the full-model signature")
    else:
        for value, tensor in zip(block.args, entries, strict=True):
            bind(value, tensor)
    bindings = receipt.get("source_values")
    if not isinstance(bindings, list):
        problems.append("source value bindings must be a list")
        bindings = []
    for binding in bindings:
        if not isinstance(binding, Mapping):
            problems.append("malformed source value binding")
            continue
        index, result = binding.get("op_index"), binding.get("result_index")
        if (isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(ops)
                or isinstance(result, bool) or not isinstance(result, int)
                or not 0 <= result < len(ops[index].results)):
            problems.append("source value binding references an absent result")
        else:
            bind(ops[index].results[result], binding.get("tensor"))
    if not isinstance(outputs, list) or len(outputs) != len(returns[0].operands):
        problems.append("output bindings do not cover all model results")
    else:
        for value, tensor in zip(returns[0].operands, outputs, strict=True):
            if value_tensors.get(value) != tensor:
                problems.append("output bindings disagree with actual source return order")
    abi = command_buffer.get("kernel_abi") or {}
    if not isinstance(abi, Mapping):
        abi = {}
    if abi.get("kind") != "whole_program":
        problems.append("global program does not declare the whole_program kernel ABI")
    abi_args = abi.get("args")
    if (not isinstance(abi_args, list)
            or any(not isinstance(arg, Mapping) or not isinstance(arg.get("tensor"), str)
                   for arg in abi_args)):
        problems.append("whole-program ABI arguments must name tensors")
        abi_args = []
    abi_names = [arg["tensor"] for arg in abi_args]
    if len(set(abi_names)) != len(abi_names) or set(abi_names) != set(tensors):
        problems.append("whole-program ABI must cover each materialized tensor exactly once")
    if abi.get("outputs") != outputs:
        problems.append("whole-program ABI results disagree with source return bindings")
    # Source tensor origins are established above. Actual task reads/writes may retain values in
    # host SSA across tasks; require each declared physical crossing to have a real source origin.
    known_tensors = set(value_tensors.values())
    temporary_names: set[str] = set()
    temporaries = receipt.get("compiler_temporaries", [])
    if not isinstance(temporaries, list):
        problems.append("compiler temporaries must be a list")
        temporaries = []
    for temporary in temporaries:
        if not isinstance(temporary, Mapping):
            problems.append("malformed compiler temporary")
            continue
        name = temporary.get("tensor")
        index, result = temporary.get("source_op_index"), temporary.get("source_result_index")
        if (not isinstance(name, str) or name in known_tensors or name in temporary_names
                or isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(ops)
                or isinstance(result, bool) or not isinstance(result, int)
                or not 0 <= result < len(ops[index].results)):
            problems.append("compiler temporary lacks a unique tensor and actual source result")
            continue
        spec = tensors.get(name)
        value_type = ops[index].results[result].type
        from xdsl.dialects.builtin import TensorType
        if (not isinstance(spec, Mapping) or spec.get("role") != "intermediate"
                or not isinstance(value_type, TensorType)
                or spec.get("shape") != list(value_type.get_shape())
                or not isinstance(temporary.get("purpose"), str) or not temporary["purpose"]):
            problems.append("compiler temporary must declare purpose and preserve source result shape")
            continue
        # Dtype changes in compiler scratch storage are declarations here, not numeric proof.
        # Qualification must independently check the associated arithmetic/readout semantics.
        temporary_names.add(name)
    known_tensors.update(temporary_names)
    if set(tensors) != known_tensors:
        problems.append("materialized tensors lack complete source or compiler-temporary provenance")
    if set(encodings) != set(checked_encodings):
        problems.append("storage encodings lack valid source-bound contracts")
    initialized_temporaries: set[str] = set()
    for row in tasks:
        if isinstance(row, Mapping):
            for access in ("reads", "writes"):
                crossings = row.get(access)
                if not isinstance(crossings, list):
                    problems.append(f"task {row.get('task_index')} has malformed {access}")
                    continue
                for tensor in crossings:
                    if not isinstance(tensor, str) or tensor not in known_tensors:
                        problems.append(f"task {row.get('task_index')} has an unbound physical tensor crossing")
                    elif access == "reads" and tensor in temporary_names \
                            and tensor not in initialized_temporaries:
                        problems.append("compiler temporary is read before an owning task writes it")
                    elif access == "writes" and tensor in temporary_names:
                        initialized_temporaries.add(tensor)
    if temporary_names != initialized_temporaries:
        problems.append("declared compiler temporary has no owning writer")

    lowered = parsed_lowered_module
    if lowered is None:
        from xdsl.dialects.llvm import LLVM
        context = make_context()
        context.load_dialect(LLVM)
        lowered = parse_mlir_text(lowered_text, context)
    lowered.verify()
    emitted_counts: Counter[int] = Counter()
    shared_constants = 0
    cfg_evidence: dict[str, Any] = {}
    host_activity: dict[str, Any] = {"status": "UNKNOWN", "reason": "kernel CFG unavailable"}
    llvm_functions = [op for op in lowered.body.block.ops if op.name == "llvm.func"]
    if len(llvm_functions) != 1 or not llvm_functions[0].body.blocks:
        problems.append("lowered program must have one defined kernel function")
    elif len(llvm_functions[0].body.blocks[0].args) != len(abi_args):
        problems.append("lowered kernel arguments disagree with whole-program tensor ABI")
    if len(llvm_functions) == 1 and llvm_functions[0].body.blocks:
        from .host_cfg_index import prepare_host_cfg
        prepared_cfg = prepare_host_cfg(llvm_functions[0])
        from .task_cfg_evidence import analyze_task_cfg
        cfg_evidence = analyze_task_cfg(
            llvm_functions[0], task_ids, prepared_cfg=prepared_cfg)
        problems.extend(cfg_evidence["problems"])
        emitted_counts.update(cfg_evidence.get("emitted_operations_by_task", {}))
        shared_constants = cfg_evidence.get("shared_prologue_operations", {}).get("llvm.mlir.constant", 0)
        # A compact dynamic-work report keeps loop compression from being presented as work
        # deletion. Full CFG/block traces are not needed in an agent's first-page context.
        from .host_cfg_activity import analyze_host_cfg_activity
        try:
            activity = analyze_host_cfg_activity(
                llvm_functions[0], prepared_cfg=prepared_cfg)
            fields = ("schema", "status", "problems", "loop_count", "static_operations",
                      "dynamic_operations", "load_payload_bytes", "store_payload_bytes",
                      "static_allocation_payload_bytes", "cpu_cycles", "dram_bytes",
                      "cache_traffic_bytes", "scope", "limitations",
                      "top_allocations_by_static_payload", "top_buffers_by_scalar_memory_payload",
                      "memory_hotspot_identity_scope")
            host_activity = {name: activity[name] for name in fields}
            host_activity["artifact_sha256"] = hashlib.sha256(lowered_text.encode()).hexdigest()
            # Retain all already-computed rows for bounded probe selection. Top-N
            # is a presentation view, not the complete source-task accounting.
            host_activity["tasks"] = activity["tasks"]
            host_activity["task_activity_coverage"] = "complete"
            top_tasks = sorted(activity["tasks"], reverse=True, key=lambda row: (
                (row["load_payload_bytes"] or 0) + (row["store_payload_bytes"] or 0)))[:5]
            for row in top_tasks:
                ident = int(row["task"]) if row["task"] != "unowned" else -1
                matching = [task for task in tasks if task.get("task_index") == ident]
                row["source_regions"] = sorted({region for task in matching
                    for index in task["source_op_indices"]
                    if isinstance(index, int) and 0 <= index < len(ops)
                    if isinstance(region := getattr(ops[index].attributes.get("prov.region_id"), "data", None), str)})
            host_activity["top_tasks_by_scalar_memory_payload"] = top_tasks
        except Exception as exc:  # Cost evidence may be unavailable without invalidating a graph.
            host_activity = {"status": "UNKNOWN", "reason": f"host work analysis failed: {type(exc).__name__}: {exc}"}
    absent_tasks = set(task_ids) - emitted_counts.keys()
    if absent_tasks:
        problems.append(f"planned tasks emitted no owned operation: {sorted(absent_tasks)}")
    # Explicit representation changes need actual address/dataflow evidence, not
    # merely the candidate's plan quantities. Keep this distinct from complete
    # consumer numerics and timing. Inspect markers too: dropping the declaration
    # must not turn a submitted transition into an unexamined one.
    physical_transitions: dict[str, Any] = {"status": "not_declared",
        "scope": "no explicit physical transition declared or marked; not proof of absence"}
    if (receipt.get("physical_transitions") or
            any(any(name in op.attributes for name in ("merlin.global_transition",
                    "merlin.transition_source", "merlin.transition_buffer")) for op in lowered.walk()) or
            ("physical_transitions" in receipt and not isinstance(receipt["physical_transitions"], list))):
        from .physical_transition_evidence import verify_physical_transitions
        physical_transitions = verify_physical_transitions(
            source_text=source_text, lowered_text=lowered_text, command_buffer=command_buffer)
        if physical_transitions.get("status") == "refused":
            problems.append("declared physical transition fails actual address/dataflow verification")
    return {
        "schema": "compiler_global_plan_verification_v1",
        "status": "refused" if problems else "verified",
        "candidate_sha256": candidate_sha256,
        "source_sha256": source_digest,
        "logical_dispatch_digest": dispatch_digest(graph), "plan_digest": _digest(receipt),
        "candidate_lowered_sha256": hashlib.sha256(lowered_text.encode()).hexdigest(),
        "candidate_command_buffer_sha256": command_buffer_sha256 or _digest(command_buffer),
        "source_operations": len(ops), "tasks": len(tasks),
        "declared_task_kinds": declared_task_kinds,
        "execution_placement": "UNVERIFIED: requires target-bound emitted instruction evidence",
        "emitted_operations_by_task": dict(sorted(emitted_counts.items())),
        "shared_immutable_constants": shared_constants,
        "compiler_temporaries": sorted(temporary_names),
        "storage_encodings": checked_encodings,
        "physical_transition_evidence": physical_transitions,
        "control_flow": cfg_evidence,
        "host_activity": host_activity,
        "problems": problems,
        "proof_scope": "source coverage, graph dependencies, tensor ABI, scheduled and emitted ownership",
        "numeric_equivalence": "requires independent mechanism witnesses",
        "timing": "UNKNOWN",
    }
