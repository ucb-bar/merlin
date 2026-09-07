"""Compare emitted whole-model work without turning partial counters into cycle verdicts.

Static instruction counts, dynamic host IR work, and declared transfer payload are
separate dimensions. Their units must never be added or silently cancelled. This
brief directs the next compiler question; it does not select a timing winner.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _count(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if value >= 0 and math.isfinite(value) else None


def _metrics(analysis: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    diag = _mapping(analysis.get("diagnostics"))
    plan = _mapping(diag.get("verified_global_plan_emission"))
    host = _mapping(plan.get("host_activity"))
    host_known = host.get("status") == "derived"
    dynamic = _mapping(host.get("dynamic_operations"))
    artifact = _mapping(_mapping(diag.get("target_artifact_activity")).get("candidate"))
    issued = _mapping(artifact.get("issued")) if artifact.get("status") == "decoded" else {}
    arm = _mapping(_mapping(diag.get("arms")).get("candidate"))
    movement = _mapping(arm.get("movement"))
    placement = _mapping(_mapping(diag.get("model_contraction_placement")).get("candidate"))
    machine = _mapping(_mapping(diag.get("machine_artifact_activity")).get("candidate"))
    emitted_digest = _mapping(analysis.get("emission")).get("candidate_lowered_sha256")
    machine_known = (machine.get("status") == "compiled" and bool(emitted_digest)
                     and machine.get("source_sha256") == emitted_digest)
    sites = _mapping(machine.get("instruction_sites")) if machine_known else {}
    result: dict[str, dict[str, Any]] = {}

    def add(name: str, value: Any, scope: str, axis: str) -> None:
        result[name] = {"value": _count(value), "scope": scope, "cca_axis": axis}

    add("full_model_contraction_macs", placement.get("total_contraction_macs"),
        "captured contraction arithmetic, host and device", "compute.contraction_form")
    add("dispatches", plan.get("emitted_dispatches", plan.get("tasks")),
        "verified emitted tasks, not runtime cycles", "envelope.calls_in_loop")
    add("command_buffer_declared_movement_bytes",
        movement.get("exact_bytes") if movement.get("is_lower_bound") is False else None,
        "command-buffer declared payload, not all issued movement or DRAM",
        "communication.intermediate_materialized")
    for field, axis in (
        ("compute_instructions", "compute.contraction_form"),
        ("configuration_instructions", "dispatch.descriptor_reuse"),
        ("loop_descriptor_instructions", "dispatch.loop_offloaded"),
        ("movement_instructions", "communication.resident_across_calls"),
        ("synchronization_instructions", "communication.fences"),
    ):
        add("issued_" + field, issued.get(field),
            "classified emitted instruction sites; excludes unclassified sites and descriptor-expanded work", axis)
    for field in ("load_payload_bytes", "store_payload_bytes", "static_allocation_payload_bytes"):
        add("host_" + field, host.get(field) if host_known else None,
            "pre-optimization LLVM payload, not DRAM or machine instructions",
            "communication.intermediate_materialized")
    for field, axis in (("integer_arithmetic", "compute.contraction_form"),
                        ("floating_arithmetic", "compute.activation_vectorization"),
                        ("conversion", "layout.transpose_materialized")):
        add("host_dynamic_" + field, dynamic.get(field, 0) if host_known else None,
            "pre-optimization LLVM operation executions, not CPU cycles", axis)
    add("machine_object_bytes", machine.get("object_bytes") if machine_known else None,
        "compiled relocatable object size, including metadata; not runtime memory",
        "compute.contraction_form")
    for field in ("total", "vector", "vsetvl", "scalar_int", "scalar_float", "undecoded"):
        value = sites.get(field)
        if field == "total" and sites.get("schema") != "encoded_instruction_sites_v1":
            value = None  # Older totals exclude undecoded sites and are not commensurate.
        add("machine_instruction_sites_" + field, value,
            "post-LLVM static instruction sites, not dynamic executions or cycles",
            "compute.contraction_form")
    return result


def _bound_machine_object(analysis: Mapping[str, Any], *, arm: str = "candidate") -> str | None:
    machine = _mapping(_mapping(_mapping(analysis.get("diagnostics")).get(
        "machine_artifact_activity")).get(arm))
    source = _mapping(analysis.get("emission")).get(arm + "_lowered_sha256")
    digest = machine.get("object_sha256")
    if (machine.get("status") != "compiled" or not source
            or machine.get("source_sha256") != source
            or not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)):
        return None
    return digest


def compare_machine_objects(before: Mapping[str, Any], after: Mapping[str, Any], *,
                            before_arm: str = "candidate", after_arm: str = "candidate") -> dict[str, Any]:
    """Compare source-bound object bytes, not graph equivalence or execution cost.

    Arm selection is explicit: an unchanged last edit must not be confused with
    an unchanged optimization baseline. Missing arms never fall back to another.
    """
    left = _bound_machine_object(before, arm=before_arm)
    right = _bound_machine_object(after, arm=after_arm)
    return {
        "status": ("UNKNOWN" if left is None or right is None
                   else "identical" if left == right else "different"),
        "before_sha256": left, "after_sha256": right,
        "scope": "source-bound relocatable kernel object bytes only; not the linked full program",
    }


def _instruction_accounting(analysis: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _mapping(_mapping(_mapping(analysis.get("diagnostics")).get(
        "target_artifact_activity")).get("candidate"))
    decoded = artifact.get("status") == "decoded"
    issued = _mapping(artifact.get("issued")) if decoded else {}
    resolution = _mapping(artifact.get("encoding_resolution")) if decoded else {}
    unknown = resolution.get("unknown_instruction_indices")
    unknown_count = (len(set(unknown)) if isinstance(unknown, list)
                     and all(type(index) is int and index >= 0 for index in unknown) else None)
    return {"loop_descriptor_sites": _count(issued.get("loop_descriptor_instructions")),
            "classification_status": resolution.get("status", "UNKNOWN"),
            "unclassified_instruction_sites": unknown_count,
            "named_without_roles": resolution.get("named_without_role"),
            "dynamic_compute_work": "UNKNOWN", "physical_moved_bytes": None,
            "descriptor_expanded_work": "UNKNOWN",
            "scope": "static classified issue sites, not dynamic accelerator work or physical traffic"}


def compare_full_model_structure(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """Expose changed work and confounders for two verified emissions of one graph.

    Missing metrics are unresolved even when both sides lack them. A structural
    decrease can be worth pursuing without licensing a global performance claim.
    Numerical qualification is a separate requirement, including inherited waivers.
    """
    problems: list[str] = []
    graphs = []
    for label, analysis in (("before", before), ("after", after)):
        diag = _mapping(analysis.get("diagnostics"))
        graph = _mapping(diag.get("captured_logical_graph"))
        digest = graph.get("logical_dispatch_digest")
        if graph.get("status") != "verified" or not isinstance(digest, str) or not digest:
            problems.append(label + " complete graph is unverified")
        graphs.append(digest)
        if _mapping(diag.get("verified_global_plan_emission")).get("status") != "verified":
            problems.append(label + " plan emission is unverified")
    if graphs[0] != graphs[1]:
        problems.append("logical model graphs differ")
    result: dict[str, Any] = {
        "schema": "full_model_structural_delta_v1", "status": "refused" if problems else "compared",
        "problems": problems, "graph_digest": graphs[1], "metrics": {},
        "decreased": [], "increased": [], "unknown": [], "changed_cca_axes": [],
        "structural_relation": "unresolved", "cycle_selection": "UNMEASURED",
        "licence": "structural work accounting only; not numerical equivalence or a timing winner",
        "required_next_evidence": [],
        "machine_object_comparison": {"status": "UNKNOWN"},
    }
    if problems:
        return result
    accounting = {arm: _instruction_accounting(analysis)
                  for arm, analysis in (("before", before), ("after", after))}
    result["instruction_accounting"] = accounting
    if any(row["loop_descriptor_sites"] is not None and row["loop_descriptor_sites"] > 0
           for row in accounting.values()):
        result["required_next_evidence"].append(
            "loop descriptors can issue compute and movement internally: fewer standalone sites are "
            "not work deletion; verify the selected task's emitted route, descriptor semantics and "
            "matched short context before attributing a performance gain")
    if any(row["classification_status"] != "complete" for row in accounting.values()):
        result["required_next_evidence"].append(
            "instruction classification is incomplete or unknown: zero classified compute/movement "
            "sites is not proof that the accelerator performs no such work")
    result["machine_object_comparison"] = compare_machine_objects(before, after)
    identical_object = result["machine_object_comparison"]["status"] == "identical"
    if identical_object:
        result["required_next_evidence"].insert(0,
            "kernel object is byte-identical: pre-LLVM count reductions do not demonstrate "
            "emitted-kernel work deletion; inspect any separate ABI/runtime changes before a timing claim")
    left, right = _metrics(before), _metrics(after)
    for name, row in right.items():
        a, b = left[name]["value"], row["value"]
        delta = None if a is None or b is None else b - a
        result["metrics"][name] = {**row, "before": a, "after": b, "delta": delta}
        result["metrics"][name].pop("value")
        if delta is None:
            result["unknown"].append(name)
        elif delta != 0:
            result["decreased" if delta < 0 else "increased"].append(name)
            result["changed_cca_axes"].append(row["cca_axis"])
    result["changed_cca_axes"] = sorted(set(result["changed_cca_axes"]))
    down, up, unknown = result["decreased"], result["increased"], result["unknown"]
    if down and up:
        relation = "mixed_work_tradeoff"
    elif down:
        relation = "decrease_in_observed_metrics"
    elif up:
        relation = "increase_in_observed_metrics"
    else:
        relation = "no_observed_change"
    result["structural_relation"] = relation
    if unknown:
        result["required_next_evidence"].append(
            "resolve only missing work dimensions relevant to this edit; unknown work does not cancel")
    if up and down:
        result["required_next_evidence"].append(
            "price the changed competing mechanisms with matched short probes before choosing by cost")
    if up or down:
        result["required_next_evidence"].append(
            "qualify changed semantics; probe only uncertain changed costs, not unrelated compute")
    result["required_next_evidence"].append(
        "occupancy, contention and cache effects remain unknown without emitted context and cost evidence")
    return result
