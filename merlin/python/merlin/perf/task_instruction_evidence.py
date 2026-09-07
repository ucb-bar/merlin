"""Host-owned static instruction ownership; not descriptor or timing equivalence.

The caller supplies the parsed module it used to decode the exact retained LLVM
bytes. No candidate summaries, parsing, compiler invocation or execution occurs.
Roles and instruction classes are host-derived target data, never guessed here.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import json


def digest(document):
    # Normalize integer JSON keys before hashing evidence crossing worker IPC.
    value = json.loads(json.dumps(document, allow_nan=False))
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _pin(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def task_instruction_binding(*, source_text, lowered_text, command_buffer_text,
                             verified_plan, target_facts, host_policy_sha256):
    if not _pin(host_policy_sha256) or not isinstance(target_facts, Mapping) or not target_facts:
        raise ValueError("task instruction evidence requires explicit policy and target facts")
    return {"source_sha256": _sha(source_text), "lowered_sha256": _sha(lowered_text),
            "command_buffer_sha256": _sha(command_buffer_text),
            "compiler_sha256": verified_plan.get("candidate_sha256"),
            "logical_dispatch_digest": verified_plan.get("logical_dispatch_digest"),
            "plan_digest": verified_plan.get("plan_digest"),
            "target_facts_sha256": digest(target_facts),
            "host_verifier_policy_sha256": host_policy_sha256}


def target_instruction_facts(target):
    """Reuse the host decoder's ISA and target-derived descriptive role map."""
    from merlin.kernels.decode.rocc import funct_table_for
    from merlin.kernels.endpoints import endpoints_for
    from merlin.targetgen.rocc import decode
    names = funct_table_for(target).get("names") or {}
    endpoints = endpoints_for(target)
    return {"target": target, "isa": decode.isa_constants(target), "instruction_names": names,
        "roles_by_selector": {str(selector): sorted({role for endpoint in endpoints
            for role in endpoint.roles_of(name)}) for selector, name in names.items()},
        "role_sources": [endpoint.source for endpoint in endpoints]}


def summarize_task_instructions(*, source_text, lowered_text, command_buffer_text,
        command_buffer, verified_plan, parsed_module, decoded_trace,
        target_facts, host_policy_sha256, decode_module, short_execution_admission=None):
    """Join verified source ownership with every actual static inline-asm site.

    ``target_facts`` contains the host ISA plus optional ``roles_by_selector``.
    A role is descriptive only: no dynamic count, queue state, address semantics,
    loop expansion, numeric correctness or route correspondence is inferred.
    """
    binding = task_instruction_binding(source_text=source_text, lowered_text=lowered_text,
        command_buffer_text=command_buffer_text, verified_plan=verified_plan,
        target_facts=target_facts, host_policy_sha256=host_policy_sha256)
    plan = command_buffer.get("params", {}).get("global_program_plan")
    checks = {"source_sha256": binding["source_sha256"],
        "candidate_lowered_sha256": binding["lowered_sha256"],
        "candidate_command_buffer_sha256": binding["command_buffer_sha256"],
        "plan_digest": digest(plan)}
    implicit_sources = []
    admission_digest = None
    if short_execution_admission is not None:
        admission = short_execution_admission
        expected = {"schema": "short_initializer_execution_admission_v1",
            "status": "source_bound_numerical_probe", "source_sha256": binding["source_sha256"],
            "lowered_sha256": binding["lowered_sha256"], "command_buffer_sha256": digest(command_buffer),
            "candidate_sha256": binding["compiler_sha256"], "declared_proof_sha256": digest(verified_plan),
            "declared_plan_modified": False, "declared_plan_verified": False,
            "target_route_verified": False, "global_cost_calibration": False}
        closure = admission.get("initializer_closure", {}) if isinstance(admission, Mapping) else {}
        implicit_sources = closure.get("implicit_source_op_indices")
        if (not isinstance(admission, Mapping) or any(admission.get(k) != v for k, v in expected.items())
                or closure.get("status") != "proved" or closure.get("source_sha256") != binding["source_sha256"]
                or closure.get("command_buffer_sha256") != digest(command_buffer)
                or not isinstance(implicit_sources, list) or not implicit_sources
                or any(type(index) is not int or index < 0 for index in implicit_sources)
                or len(set(implicit_sources)) != len(implicit_sources)
                or verified_plan.get("status") != "refused"
                or verified_plan.get("control_flow", {}).get("status") != "verified"
                or verified_plan.get("problems") != ["source operations are not fully covered: " + str(implicit_sources)]):
            raise ValueError("short initializer summary requires exact separate host execution admission")
        admission_digest = digest(admission)
    if ((verified_plan.get("status") != "verified" and admission_digest is None) or not isinstance(plan, Mapping)
            or any(verified_plan.get(key) != value for key, value in checks.items())
            or any(not _pin(binding[key]) for key in ("compiler_sha256", "logical_dispatch_digest"))
            or digest(json.loads(command_buffer_text)) != digest(command_buffer)):
        raise ValueError("task instruction evidence requires exact verified source/LLVM/buffer proof")
    if parsed_module is None:
        raise ValueError("host-retained parsed LLVM unavailable; no probe-time reparse permitted")
    rows = decoded_trace.get("instructions")
    operations = list(parsed_module.walk())
    assembly = [op for op in operations if op.name == "llvm.inline_asm"]
    if (not isinstance(rows, list) or len(rows) != len(assembly)
            or decode_module(parsed_module).get("instructions") != rows):
        raise ValueError("decoded instructions differ from the host-retained LLVM module")
    tasks = {}
    source_owners = set()
    for task in plan.get("tasks", []):
        index, sources = task.get("task_index"), task.get("source_op_indices")
        if (type(index) is not int or index < 0 or index in tasks
                or not isinstance(sources, list) or not sources
                or any(type(s) is not int or s < 0 or s in source_owners for s in sources)
                or len(set(sources)) != len(sources)):
            raise ValueError("ambiguous task/source ownership")
        source_owners.update(sources)
        tasks[index] = {"task_index": index, "source_op_indices": sources,
            "declared_task_kind": task.get("kind"), "instruction_indices": [],
            "static_operation_counts": Counter(), "class_counts": Counter(),
            "role_counts": Counter(), "unknown_instruction_indices": [],
            "instructions_without_decoded_fields": [], "instructions_without_target_roles": []}

    def owner(op):
        value = getattr(getattr(op.attributes.get("merlin.global_task"), "value", None), "data", None)
        return value if type(value) is int else None

    wrappers = []
    roles = {str(key): value for key, value in target_facts.get("roles_by_selector", {}).items()}
    for op in operations:
        index = owner(op)
        if index in tasks:
            tasks[index]["static_operation_counts"][op.name] += 1
    for position, (op, instruction) in enumerate(zip(assembly, rows, strict=True)):
        if not isinstance(instruction, Mapping) or instruction.get("index") != position:
            raise ValueError("inconsistent decoder instruction position")
        index = owner(op)
        if index in (-1, -2):
            wrappers.append(position)
            continue
        if index not in tasks:
            raise ValueError("inline-asm has missing task ownership or inconsistent decoder position")
        task = tasks[index]
        klass = instruction.get("class")
        if not isinstance(klass, str) or not klass:
            raise ValueError("decoder instruction class is absent")
        task["instruction_indices"].append(position)
        task["class_counts"][klass] += 1
        if klass == "UNKNOWN":
            task["unknown_instruction_indices"].append(position)
        if not instruction.get("decoded"):
            task["instructions_without_decoded_fields"].append(position)
        declared_roles = roles.get(str(instruction.get("funct")), [])
        if not isinstance(declared_roles, list) or any(not isinstance(role, str) or not role for role in declared_roles):
            raise ValueError("target role facts are malformed")
        task["role_counts"].update(set(declared_roles))
        if not declared_roles:
            task["instructions_without_target_roles"].append(position)
    result_tasks = []
    for task in tasks.values():
        for field in ("static_operation_counts", "class_counts", "role_counts"):
            task[field] = dict(sorted(task[field].items()))
        task["owned_instruction_payload_sha256"] = digest([
            {key: value for key, value in rows[i].items() if key != "index"}
            for i in task["instruction_indices"]])
        task["classification_coverage"] = "partial" if task["unknown_instruction_indices"] else "complete"
        task["descriptor_semantics"] = "UNVERIFIED"
        result_tasks.append(task)
    return {"schema": "task_instruction_evidence_v1",
        "status": "short_admitted_static_ownership" if admission_digest else "static_ownership_verified",
        "binding": binding, "decoded_instructions_sha256": digest(rows),
        "declared_source_plan_status": verified_plan["status"],
        "short_execution_admission_sha256": admission_digest,
        "implicit_initializer_source_op_indices": implicit_sources,
        "tasks": result_tasks, "wrapper_instruction_indices": wrappers,
        "static_instruction_count": len(rows), "route_correspondence": "UNKNOWN",
        "numeric_equivalence": "UNPROVEN", "dynamic_instruction_counts": None,
        "timing_calibration_admissible": False, "global_cost_validated": False,
        "scope": "source-bound actual static instruction class/role presence only",
        "limitations": ["absence of decoded fields does not distinguish fieldless instructions from unmodeled descriptors",
            "payload differences include physical address and geometry differences, not necessarily route changes",
            "loop expansion, dynamic execution, operand/address semantics and surrounding contention are unproven"]}
