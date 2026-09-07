"""Compare cached full/short static task observations without reconstructing work.

Inputs must be host-retained summaries plus independently supplied expected
bindings and receipt hashes. This module neither parses LLVM nor chooses source
regions, compiler arms, opcode families, geometry, or a timing interpretation.
"""
from __future__ import annotations

from collections.abc import Mapping
from .task_instruction_evidence import digest


_BINDING_KEYS = ("source_sha256", "lowered_sha256", "command_buffer_sha256", "compiler_sha256",
    "logical_dispatch_digest", "plan_digest", "target_facts_sha256", "host_verifier_policy_sha256")


def _pin(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _selected(envelope, source_index, *, short_domain=False):
    if not isinstance(envelope, Mapping):
        raise ValueError("missing host summary envelope")
    summary, binding = envelope.get("summary"), envelope.get("expected_binding")
    if (not isinstance(summary, Mapping) or summary.get("schema") != "task_instruction_evidence_v1"
            or summary.get("status") not in {"static_ownership_verified", "short_admitted_static_ownership"}
            or digest(summary) != envelope.get("summary_sha256")
            or not isinstance(binding, Mapping) or summary.get("binding") != binding
            or any(not _pin(binding.get(key)) for key in _BINDING_KEYS)):
        raise ValueError("missing or stale host-owned task instruction evidence")
    if summary["status"] == "short_admitted_static_ownership":
        admission = envelope.get("short_execution_admission")
        if (not short_domain or not isinstance(admission, Mapping)
                or admission.get("schema") != "short_initializer_execution_admission_v1"
                or admission.get("status") != "source_bound_numerical_probe"
                or digest(admission) != summary.get("short_execution_admission_sha256")
                or summary.get("declared_source_plan_status") != "refused"
                or not summary.get("implicit_initializer_source_op_indices")
                or source_index in summary["implicit_initializer_source_op_indices"]
                or any(admission.get(key) != binding[key] for key in
                       ("source_sha256", "lowered_sha256", "command_buffer_sha256"))
                or admission.get("candidate_sha256") != binding["compiler_sha256"]):
            raise ValueError("short-only initializer summary lacks its separately pinned admission")
    tasks = summary.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("task observations are unavailable")
    selected = [task for task in tasks if source_index in task.get("source_op_indices", [])]
    if len(selected) != 1 or selected[0].get("source_op_indices") != [source_index]:
        raise ValueError("selected source lacks unique single-source task observation; fused attribution is unresolved")
    task = selected[0]
    indices = task.get("instruction_indices")
    unknown = task.get("unknown_instruction_indices")
    no_roles = task.get("instructions_without_target_roles")
    if (not isinstance(indices, list) or len(set(indices)) != len(indices)
            or any(type(index) is not int or index < 0 for index in indices)
            or not isinstance(unknown, list) or not set(unknown).issubset(indices)
            or not isinstance(no_roles, list) or not set(no_roles).issubset(indices)):
        raise ValueError("malformed static instruction coverage")
    for field in ("class_counts", "role_counts"):
        counts = task.get(field)
        if (not isinstance(counts, Mapping) or any(not isinstance(name, str) or not name
                or type(count) is not int or count <= 0 for name, count in counts.items())):
            raise ValueError("invalid target-derived class or role counts")
    if (sum(task["class_counts"].values()) != len(indices)
            or task["class_counts"].get("UNKNOWN", 0) != len(unknown)
            or not _pin(task.get("owned_instruction_payload_sha256"))):
        raise ValueError("static instruction summary is inconsistent")
    return {"task_index": task["task_index"], "source_op_indices": task["source_op_indices"],
        "declared_task_kind": task.get("declared_task_kind"),
        "source_ownership_scope": summary["status"],
        "implicit_initializer_source_op_indices": summary.get("implicit_initializer_source_op_indices", []),
        "known_classes": sorted(set(task["class_counts"])-{"UNKNOWN"}),
        "known_roles": sorted(task["role_counts"]),
        "class_counts": dict(task["class_counts"]), "role_counts": dict(task["role_counts"]),
        "static_instruction_count": len(indices), "unknown_instruction_count": len(unknown),
        "unbound_role_instruction_count": len(no_roles),
        "owned_instruction_payload_sha256": task["owned_instruction_payload_sha256"]}


def compare_task_route_presence(*, full, short, extraction, extraction_sha256, short_source_op_index):
    """Join four exact host records, returning observations rather than equivalence.

    ``full`` and ``short`` each map ``before``/``after`` to an envelope containing
    ``summary``, its host-recorded ``summary_sha256``, and ``expected_binding``
    obtained independently from the bound artifacts/current host policy. A
    candidate-supplied envelope is not an authority to invoke this host API.
    """
    result = {"schema": "task_route_presence_comparison_v1", "status": "UNKNOWN",
        "descriptor_semantic_equivalence": "UNKNOWN", "emitted_address_equivalence": "UNKNOWN",
        "numeric_equivalence": "UNPROVEN", "timing_calibration_admissible": False,
        "global_cost_validated": False, "full_model_executed": False,
        "scope": "four host-cached single-source task class/role presence observations",
        "limitations": ["instruction counts are static, not work, tile geometry, or latency",
            "matching class/role sets do not prove matching descriptor or address semantics",
            "undecoded instructions prevent claims of absent mechanisms",
            "fused multi-source tasks need a separate attribution proof"]}
    try:
        if (not isinstance(extraction, Mapping) or digest(extraction) != extraction_sha256
                or type(extraction.get("source_op_index")) is not int or extraction["source_op_index"] < 0
                or type(short_source_op_index) is not int or short_source_op_index < 0):
            raise ValueError("missing exact host extraction/source-index binding")
        source_pins = {"full": extraction.get("source_sha256"), "short": extraction.get("probe_source_sha256")}
        if any(not _pin(pin) for pin in source_pins.values()):
            raise ValueError("extraction lacks exact full and reduced source hashes")
        observations, bindings, receipts = {}, {}, {}
        for domain, records in (("full", full), ("short", short)):
            if not isinstance(records, Mapping) or set(records) != {"before", "after"}:
                raise ValueError("both explicit compiler arms are required in each source domain")
            observations[domain], bindings[domain], receipts[domain] = {}, {}, {}
            for arm, envelope in records.items():
                source_index = extraction["source_op_index"] if domain == "full" else short_source_op_index
                observations[domain][arm] = _selected(envelope, source_index, short_domain=domain == "short")
                bindings[domain][arm] = dict(envelope["expected_binding"])
                receipts[domain][arm] = envelope["summary_sha256"]
                if bindings[domain][arm]["source_sha256"] != source_pins[domain]:
                    raise ValueError("task observation belongs to a different source/extraction")
            if bindings[domain]["before"]["logical_dispatch_digest"] != bindings[domain]["after"]["logical_dispatch_digest"]:
                raise ValueError("compiler arms refer to different source graphs")
        for arm in ("before", "after"):
            if bindings["full"][arm]["compiler_sha256"] != bindings["short"][arm]["compiler_sha256"]:
                raise ValueError("reduced source was compiled by a different compiler arm")
        for key in ("target_facts_sha256", "host_verifier_policy_sha256"):
            if len({bindings[domain][arm][key] for domain in bindings for arm in bindings[domain]}) != 1:
                raise ValueError("target facts or host policy differ across the four observations")
        def presence(domain, field):
            return {arm: observations[domain][arm][field] for arm in ("before", "after")}
        classes = {domain: presence(domain, "known_classes") for domain in observations}
        roles = {domain: presence(domain, "known_roles") for domain in observations}
        def changes(values):
            return {domain: {"appeared": sorted(set(arms["after"])-set(arms["before"])),
                "disappeared_from_classified_sites": sorted(set(arms["before"])-set(arms["after"]))}
                for domain, arms in values.items()}
        changed = classes["full"]["before"] != classes["full"]["after"]
        same = classes["full"] == classes["short"]
        status = ("observed_known_class_presence_change_reproduced" if changed and same else
                  "no_known_class_presence_change" if same else "observed_known_class_presence_mismatch")
        result.update(status=status, bindings=bindings, summary_sha256=receipts,
            extraction_sha256=extraction_sha256, observations=observations,
            class_presence_changes=changes(classes), role_presence_changes=changes(roles),
            known_class_presence_equal_by_arm={arm: classes["full"][arm] == classes["short"][arm] for arm in ("before", "after")},
            known_role_presence_equal_by_arm={arm: roles["full"][arm] == roles["short"][arm] for arm in ("before", "after")},
            all_static_classes_decoded=all(not row["unknown_instruction_count"] for domain in observations.values() for row in domain.values()),
            selected_owned_payload_changed={domain: arms["before"]["owned_instruction_payload_sha256"] != arms["after"]["owned_instruction_payload_sha256"]
                for domain, arms in observations.items()})
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        result["reason"] = str(exc)
    return result
