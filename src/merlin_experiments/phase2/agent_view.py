"""Bounded agent-facing evidence views; no measurements or target discovery."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from . import contracts as P2_CONTRACTS
from .broker_evidence import _is_sha256


def controlled_context_capability(analysis: Mapping[str, Any], *, provider_installed: bool | None) -> dict[str, Any]:
    """Distinguish adapter installation from a supported motif in this exact emitted revision."""
    diag = analysis.get("diagnostics") or {}
    context = diag.get("queued_movement_context") or {}
    plan = diag.get("verified_global_plan_emission") or {}
    artifact = (analysis.get("emission") or {}).get("candidate_lowered_sha256")
    bound = (
        bool(artifact)
        and plan.get("status") == "verified"
        and plan.get("candidate_lowered_sha256") == artifact
        and context.get("schema") == "queued_movement_context_candidates_v1"
        and context.get("artifact_sha256") == artifact
    )
    motifs = context.get("motifs") if bound else None
    count = len(motifs) if isinstance(motifs, list) else None
    return {
        "provider_installed": provider_installed,
        "artifact_sha256": artifact,
        "current_supported_motif_count": count,
        "current_candidate_status": (
            "UNKNOWN"
            if count is None
            else "no_extracted_supported_motifs"
            if count == 0
            else "motifs_pending_admission"
        ),
        "available": bool(provider_installed and count),
        "admission_verified": False,
        "licence": "installation is not applicability; absent motifs do not prove absence of accelerator work",
    }


def storage_encoding_agent_summary(
    record: Mapping[str, Any], *, complete_evidence: str, arm: str = "candidate"
) -> dict[str, Any] | None:
    """Expose unresolved storage obligations without copying every tensor contract into a prompt."""
    if arm not in {"candidate", "baseline"}:
        raise ValueError("storage summary arm must be candidate or baseline")
    plan_key = "verified_global_plan_emission" if arm == "candidate" else "verified_baseline_global_plan_emission"
    analysis = record.get("analysis") or {}
    diagnostics = analysis.get("diagnostics") or {}
    plan = diagnostics.get(plan_key) or {}
    if not isinstance(plan, Mapping) or "storage_encodings" not in plan:
        return None
    contracts = plan["storage_encodings"]
    emission = analysis.get("emission") or {}
    graph = diagnostics.get("captured_logical_graph") or {}
    compiler_sha = (
        analysis.get("candidate_sha256") if arm == "candidate" else record.get("optimization_baseline_sha256")
    )
    expected = {
        "candidate_sha256": compiler_sha,
        "logical_dispatch_digest": graph.get("logical_dispatch_digest"),
        "candidate_lowered_sha256": emission.get(arm + "_lowered_sha256"),
        "candidate_command_buffer_sha256": emission.get(arm + "_command_buffer_sha256"),
    }
    bound = (
        plan.get("status") == "verified"
        and graph.get("status") == "verified"
        and (
            record.get("candidate_sha256")
            if arm == "candidate"
            else (record.get("optimization_baseline") or {}).get("sha256")
        )
        == compiler_sha
        and _is_sha256(plan.get("plan_digest"))
        and all(_is_sha256(value) and plan.get(key) == value for key, value in expected.items())
    )
    count = len(contracts) if isinstance(contracts, Mapping) else None
    summary: dict[str, Any] = {
        "schema": "agent_storage_encoding_summary_v1",
        "arm": arm,
        "status": "source_bound_contracts_verified"
        if bound and count
        else "no_checked_storage_contracts"
        if bound and count == 0
        else "UNKNOWN",
        "reported_contract_count": count,
        "source_plan_binding_verified": bound,
        "binding": {**expected, "plan_digest": plan.get("plan_digest")},
        "details": {
            "path": complete_evidence,
            "canonical_record_sha256": P2_CONTRACTS.document_sha256(record),
            "json_pointer": "/analysis/diagnostics/" + plan_key + "/storage_encodings",
            "encoding_map_sha256": P2_CONTRACTS.document_sha256(contracts),
        },
        "scope": "declared storage map only; caller materialization and actual consumer addresses are separate proofs",
        "full_model_numerics_qualified": False,
        "global_cost_validated": False,
    }
    transitions = plan.get("physical_transition_evidence")
    if isinstance(transitions, Mapping):
        summary["physical_transition_evidence"] = {
            "status": transitions.get("status", "UNKNOWN") if bound else "UNKNOWN",
            "source_plan_binding_verified": bound,
            "details": {
                "path": complete_evidence,
                "json_pointer": "/analysis/diagnostics/" + plan_key + "/physical_transition_evidence",
                "evidence_sha256": P2_CONTRACTS.document_sha256(transitions),
            },
            "scope": "explicit emitted copy/address proof only; not arbitrary consumer numerics or timing",
        }
    for key in ("caller_materialization", "emitted_consumer_addressing"):
        pending = 0
        histogram: dict[str, int] = {}
        for row in contracts.values() if isinstance(contracts, Mapping) else ():
            value = row.get(key) if isinstance(row, Mapping) else None
            label = value[:200] if isinstance(value, str) else "UNKNOWN"
            histogram[label] = histogram.get(label, 0) + 1
            pending += isinstance(value, str) and value.startswith("requires ")
        summary[key] = {
            "status": "UNRESOLVED" if bound and pending else "UNKNOWN",
            "reported_pending_count": pending if count is not None else None,
            "other_or_unknown_count": count - pending if count is not None else None,
            "reported_status_counts": dict(sorted(histogram.items())[:3]),
            "verified": False,
        }
    summary["obligation_examples"] = [
        {
            "tensor": name,
            "caller_materialization": row.get("caller_materialization", "UNKNOWN"),
            "emitted_consumer_addressing": row.get("emitted_consumer_addressing", "UNKNOWN"),
        }
        for name, row in (list(contracts.items())[:3] if isinstance(contracts, Mapping) else ())
        if isinstance(row, Mapping)
    ]
    return summary


def declared_instruction_prompt(brief: Mapping[str, Any]) -> str:
    """The prompt paragraph that puts the declared instruction set in front of the agent."""
    if brief.get("status") != "derived":
        return (
            "The target's DECLARED instruction set could not be derived in this run "
            f"({brief.get('reason')}). Treat instruction coverage as UNKNOWN; do not read the "
            "instructions the program already emits as the set the machine offers.\n"
        )
    listing = "; ".join(
        f"{entry['name']}" + (f" [{', '.join(entry['roles'])}]" if entry.get("roles") else "")
        for entry in brief.get("instructions") or ()
    )
    return (
        f"The target DECLARES {brief['declared_count']} accelerator instructions, derived from its "
        f"own RTL facts (custom opcode {brief.get('custom_opcode')}): {listing}. "
        "A declared instruction this program never emits is an OPPORTUNITY, not a defect: the "
        "compiler may be correct without it, and the largest structural levers hide in a "
        "capability the lowering never reaches for. isa_capability_utilization in the analysis "
        "states which of these the emitted program uses and which it has never emitted; "
        "capability_refusals states, per site, the clause the target's own selector refused a "
        "capability on. A guard sequence stops at its FIRST failing clause, so clearing the top "
        "clause reveals the next one rather than admitting those sites.\n"
    )


def unavailable_action_notice(unavailable: Mapping[str, str]) -> str:
    """The prompt paragraph naming every action this run cannot answer, and why.

    Written even when nothing is unavailable, because the two states must be distinguishable. A
    prompt that is silent about availability reads the same whether every provider is installed or
    none is, which is how an agent came to spend its one probe call on a refusal.
    """
    if not unavailable:
        return (
            "Every registered broker action has its host provider installed in this run; none "
            "will refuse for want of one.\n"
        )
    return (
        "These broker actions are REGISTERED and CANNOT be answered in this run, because the "
        "host did not install their provider: "
        + "; ".join(f"{name} ({reason})" for name, reason in sorted(unavailable.items()))
        + ". Calling one returns a refusal and no retry will make it available. Spend no budget "
        "on them, and treat every cost they would have measured as explicitly UNKNOWN rather "
        "than as unchanged.\n"
    )


def agent_analysis_view(
    record: Mapping[str, Any], *, complete_evidence: str, context_provider_installed: bool | None = None
) -> dict[str, Any]:
    """Keep priorities in the immediate response; the complete, unpruned graph stays available."""
    from merlin.perf.structural_delta import compare_machine_objects

    view = copy.deepcopy(dict(record))
    brief = view.setdefault("analysis", {}).setdefault("optimization_brief", {})
    brief.pop("package_inventory", None)
    if record.get("historical_reference") is not None:
        brief["historical_reference"] = copy.deepcopy(record["historical_reference"]["summary"])
    comparison_seed = record.get("optimization_baseline")
    if comparison_seed is not None:
        brief["optimization_comparison_seed"] = {
            key: comparison_seed[key]
            for key in ("selection", "sha256", "reason", "scope", "objective_numerical_qualification")
        }
    # Surface the existing host comparison before prioritizing raw IR byte reductions.
    # This is presentation only: source/object binding and equality belong to
    # compare_full_model_structure, not to a second classifier in the prompt adapter.
    structural = (record.get("static_comparison") or {}).get("structural_change") or {}
    objects = structural.get("machine_object_comparison") or {}
    object_status = objects.get("status") if structural.get("status") == "compared" else "UNKNOWN"
    messages = {
        "identical": (
            "IR-only change at the audited kernel-object boundary; byte-identical target object; "
            "no demonstrated machine-code saving. Do not prioritize pre-LLVM payload reductions "
            "as hardware traffic savings. Seek a change that survives code generation or separately "
            "measure the relevant compiler/caller effect."
        ),
        "different": (
            "The audited target object changed; this alone does not demonstrate less dynamic work "
            "or a performance benefit. Inspect the changed machine mechanism before pricing it."
        ),
        "UNKNOWN": (
            "No bound cross-revision kernel-object comparison is available. Pre-LLVM metric "
            "reductions do not establish machine-code savings."
        ),
    }
    if object_status not in messages:
        object_status = "UNKNOWN"
    brief["machine_code_evidence_priority"] = {
        "status": object_status,
        "comparison_arm": "preceding_analyzed_revision",
        "previous_iteration": (record.get("static_comparison") or {}).get("previous_iteration"),
        "message": (
            "Relative to the preceding analyzed revision only: "
            + messages[object_status]
            + " This does not erase earlier changes relative to the optimization baseline."
        ),
        "comparison": copy.deepcopy(objects),
        "evidence_pointer": "/static_comparison/structural_change/machine_object_comparison",
        "scope": "relocatable kernel object only; caller, setup, linked ELF and timing are separate",
        "global_performance_benefit": "UNKNOWN",
        "numerical_equivalence": "UNKNOWN",
    }
    analysis = record.get("analysis") or {}
    brief["optimization_baseline_machine_code_comparison"] = {
        **compare_machine_objects(analysis, analysis, before_arm="baseline"),
        "comparison_arm": "optimization_baseline",
        "message": "Cumulative object-byte comparison with the compiled optimization baseline, not the last edit. "
        "Different bytes do not establish less dynamic work or a speedup.",
        "evidence_pointers": ["/analysis/emission", "/analysis/diagnostics/machine_artifact_activity"],
        "global_performance_benefit": "UNKNOWN",
        "numerical_equivalence": "UNKNOWN",
    }
    graph = view.get("analysis", {}).get("diagnostics", {}).get("captured_logical_graph", {})
    for key in ("dispatch_program", "nodes", "buffers"):
        graph.pop(key, None)
    graph["complete_unpruned_evidence"] = complete_evidence
    storage = storage_encoding_agent_summary(record, complete_evidence=complete_evidence)
    if storage is not None:
        plan = view["analysis"]["diagnostics"]["verified_global_plan_emission"]
        plan.pop("storage_encodings", None)
        plan["storage_encoding_summary"] = storage
        brief["storage_encoding_obligations"] = copy.deepcopy(storage)
    baseline_storage = storage_encoding_agent_summary(record, complete_evidence=complete_evidence, arm="baseline")
    if baseline_storage is not None:
        baseline_plan = view["analysis"]["diagnostics"]["verified_baseline_global_plan_emission"]
        baseline_plan.pop("storage_encodings", None)
        baseline_plan["storage_encoding_summary"] = baseline_storage
    view["controlled_context_capability"] = controlled_context_capability(
        record.get("analysis") or {}, provider_installed=context_provider_installed
    )
    # Measurements belong in the immediate decision context, not only in an optional raw attachment.
    if record.get("decision_feedback"):
        feedback = record["decision_feedback"]
        view["measurement_driven_next_step"] = feedback["next_step"]
        brief["scoped_mechanism_coverage"] = {
            "scope": feedback["scope"],
            "status": feedback["status"],
            "occupancy_and_latency_hiding": feedback["observation"],
            "source_task_index": feedback["source_task_index"],
            "source_op_indices": feedback["source_op_indices"],
            "same_declared_movement_work": feedback["same_declared_movement_work"],
            "physical_movement_bytes": None,
            "full_model_occupancy": "UNKNOWN",
            "full_model_contention": "UNKNOWN",
            "full_model_cycle_ordering": "UNMEASURED",
            "buffer_capacity_contract": feedback["buffer_capacity_contract"],
            "pipeline_projection_admitted": False,
        }
    portfolio = view.get("portfolio") or {}
    for index, member in enumerate(portfolio.get("members") or ()):
        member_analysis = member.get("analysis")
        if not isinstance(member_analysis, Mapping):
            continue
        member_brief = member_analysis.get("optimization_brief") or {}
        ranked = []
        for action in (member_brief.get("ranked_actions") or ())[:4]:
            ranked.append(
                {
                    key: copy.deepcopy(action.get(key))
                    for key in ("rank", "kind", "status", "detail", "evidence", "required_effects")
                }
            )
            ranked[-1]["edit_surfaces"] = [
                {key: copy.deepcopy(surface.get(key)) for key in ("id", "path", "symbol", "scope", "effects")}
                for surface in (action.get("edit_surfaces") or ())
            ]
        diagnostics = member_analysis.get("diagnostics") or {}
        member["analysis"] = {
            "schema": "portfolio_member_agent_summary_v1",
            "candidate_sha256": member_analysis.get("candidate_sha256"),
            "failure": copy.deepcopy(member_analysis.get("failure")),
            "optimization_brief": {
                "objective": copy.deepcopy(member_brief.get("objective")),
                "optimization_order": copy.deepcopy(member_brief.get("optimization_order")),
                "ranked_actions": ranked,
                "gap_coverage": copy.deepcopy(member_brief.get("gap_coverage")),
                "global_planner_wiring": copy.deepcopy(member_brief.get("global_planner_wiring")),
                "mechanism_coverage": copy.deepcopy(member_brief.get("mechanism_coverage")),
            },
            "global_signals": {
                key: copy.deepcopy(diagnostics.get(key))
                for key in (
                    "lower_bound",
                    "barriers",
                    "ordering_signals",
                    "queued_movement_context",
                    # The machine's declared capabilities, the clause each refused site failed, and the
                    # cycle floor -- kept for SECONDARY members too, which are otherwise pruned to
                    # accounting the program produced about itself.
                    "structural_levels",
                    "isa_capability_utilization",
                    "capability_refusals",
                    "cost_plane",
                )
            },
            "complete_unpruned_evidence": {
                "path": complete_evidence,
                "json_pointer": f"/portfolio/members/{index}/analysis",
            },
        }
    return view


def portfolio_action_digest(
    record: Mapping[str, Any], *, complete_evidence: str, edit_contract: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Resolve every portfolio member to one compact, action-oriented host view.

    The iteration record intentionally stores the primary analysis by JSON reference while
    secondary members are inline.  That is efficient archival structure but a poor navigation
    surface for an authoring agent.  Resolve the reference here and expose only bounded totals and
    edit surfaces that exactly occur in the host-frozen authority.
    """
    portfolio = record.get("portfolio") or {}
    members = portfolio.get("members") or ()
    authorized = {
        (row.get("surface_id"), row.get("path"), row.get("symbol"))
        for row in ((edit_contract or {}).get("existing_symbols") or ())
        if isinstance(row, Mapping)
    }
    rows = []
    shared_optimization_order = None
    for index, member in enumerate(members):
        analysis = record.get("analysis") if index == 0 else member.get("analysis")
        if not isinstance(analysis, Mapping):
            analysis = {}
        member_brief = analysis.get("optimization_brief") or {}
        if shared_optimization_order is None:
            shared_optimization_order = copy.deepcopy(member_brief.get("optimization_order"))
        diagnostics = analysis.get("diagnostics") or {}
        arm = (diagnostics.get("arms") or {}).get("candidate") or {}
        representation = arm.get("representation_activity") or {}
        movement = arm.get("movement") or {}
        plan = diagnostics.get("verified_global_plan_emission") or {}
        host = plan.get("host_activity") or {}
        placement = (diagnostics.get("model_contraction_placement") or {}).get("candidate") or {}
        task_kinds: dict[str, int] = {}
        for kind in (plan.get("declared_task_kinds") or {}).values():
            task_kinds[str(kind)] = task_kinds.get(str(kind), 0) + 1
        actions = []
        for action in (member_brief.get("ranked_actions") or ())[:4]:
            surfaces = []
            for surface in action.get("edit_surfaces") or ():
                key = (surface.get("id"), surface.get("path"), surface.get("symbol"))
                if key not in authorized:
                    continue
                surfaces.append(
                    {
                        key_name: copy.deepcopy(surface.get(key_name))
                        for key_name in ("id", "path", "symbol", "scope", "effects")
                    }
                )
                surfaces[-1]["authority"] = "exact_host_frozen_existing_symbol"
            actions.append(
                {
                    key_name: copy.deepcopy(action.get(key_name))
                    for key_name in ("rank", "kind", "status", "detail", "evidence", "required_effects")
                }
            )
            actions[-1]["authorized_edit_surfaces"] = surfaces
        rows.append(
            {
                "identity": copy.deepcopy(member.get("identity")),
                "readiness": copy.deepcopy(member.get("readiness")),
                "totals": {
                    "logical_graph_status": (diagnostics.get("captured_logical_graph") or {}).get("status"),
                    "logical_dispatches": (diagnostics.get("captured_logical_graph") or {}).get("dispatches"),
                    "accelerator_macs": arm.get("macs"),
                    "accelerator_work_exact": arm.get("exact"),
                    "movement_known_bytes": movement.get("known_bytes"),
                    "movement_known_bytes_in": movement.get("known_bytes_in"),
                    "movement_known_bytes_out": movement.get("known_bytes_out"),
                    "command_counts": copy.deepcopy(representation.get("command_counts")),
                    "task_kind_counts": task_kinds,
                    "lane_counts": copy.deepcopy((representation.get("placement") or {}).get("lane_counts")),
                    "lane_transitions": (representation.get("placement") or {}).get("adjacent_lane_transitions"),
                    "contraction_count": placement.get("contraction_count"),
                    "contraction_macs_by_lane": copy.deepcopy(placement.get("macs_by_lane")),
                    "host_dynamic_operations": copy.deepcopy(host.get("dynamic_operations")),
                    "host_load_payload_bytes": host.get("load_payload_bytes"),
                    "host_store_payload_bytes": host.get("store_payload_bytes"),
                    "host_static_allocation_payload_bytes": host.get("static_allocation_payload_bytes"),
                },
                # WHAT THE MACHINE OFFERS, WHY IT WAS REFUSED, AND WHAT THE PROGRAM CANNOT COST LESS
                # THAN. The digest is the view the prompt tells the agent to start from, and until this
                # it carried only accounting the program produced about itself: no declared instruction
                # set, no refusal clause, and no cycle floor under the dispatch delta.
                "isa_capability_utilization": copy.deepcopy(diagnostics.get("isa_capability_utilization")),
                "capability_refusals": copy.deepcopy(diagnostics.get("capability_refusals")),
                "cost_plane": copy.deepcopy(diagnostics.get("cost_plane")),
                "top_ranked_actions": actions,
                "optimization_order": copy.deepcopy(member_brief.get("optimization_order")),
                "complete_unpruned_evidence": {
                    "path": complete_evidence,
                    "json_pointer": "/analysis" if index == 0 else f"/portfolio/members/{index}/analysis",
                },
            }
        )
    return {
        "schema": "portfolio_action_digest_v1",
        "portfolio_sha256": portfolio.get("portfolio_sha256"),
        "candidate_sha256": record.get("candidate_sha256"),
        "members": rows,
        "optimization_order": shared_optimization_order,
        "fast_accuracy_bounded_evaluation": copy.deepcopy(record.get("fast_evaluation")),
        "selection": "per-model Pareto evidence; totals are never summed across models",
        "timing_status": "UNMEASURED_FULL_MODEL",
    }
