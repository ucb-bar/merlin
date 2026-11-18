"""Expose optimization evidence and real compiler edit surfaces to an experiment agent.

The harness, not the agent, joins a whole-model bottleneck to the compiler source that is allowed to
change it.  A package may declare semantic ``optimization_surfaces`` in its manifest.  Every declared
file and symbol is checked against the package's actual Python AST and command component graph before
it is shown.  When the declaration is absent, a structural source index is still provided, but the
semantic join remains explicitly UNKNOWN rather than guessed from a filename.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from merlin.analysis.optimization.evaluation.model_report import OptimizationOpportunity, WholeModelReport
from merlin.analysis.optimization.guidance.vocabulary import FINDING_AXES as _FINDING_AXES
from merlin.analysis.optimization.guidance.vocabulary import GAP_AXES as _GAP_AXES
from merlin.analysis.optimization.guidance.vocabulary import (
    MACRO_OPTIMIZATION_LADDER as _MACRO_OPTIMIZATION_LADDER,
)
from merlin.analysis.optimization.guidance.vocabulary import OPPORTUNITY_AXES as _OPPORTUNITY_AXES
from merlin.analysis.optimization.guidance.vocabulary import OPPORTUNITY_EFFECTS as _OPPORTUNITY_EFFECTS
from merlin.analysis.optimization.guidance.vocabulary import (
    QUANTIZED_REGION_EDIT_ROLES as _QUANTIZED_REGION_EDIT_ROLES,
)
from merlin.analysis.optimization.guidance.vocabulary import STRUCTURAL_EFFECTS as _STRUCTURAL_EFFECTS
from merlin.analysis.optimization.guidance.vocabulary import (
    _declared_surface_vocabulary,
    macro_optimization_order,
)

# ---- agent_guidance ----------------------------------------------------------------------------


def guidance_for_emission_analysis(
    analysis: Mapping[str, Any], inventory: PackageOptimizationInventory
) -> dict[str, Any]:
    """Turn one host-owned whole-model emission comparison into exact edit instructions.

    This is deliberately a structural brief, not a performance verdict.  It ranks regressions and
    visible inefficiencies in the emitted complete-model program, then joins each one to manifest
    declarations that resolve to real Python AST symbols.  A package without such declarations gets
    an explicit manifest edit instruction and the complete source index; semantic ownership is never
    guessed from a filename or function name.
    """

    from merlin.analysis.optimization.guidance.agent_brief import (
        PackageOptimizationInventory,
        _dominant_signatures,
        _host_hotspot_share,
        _host_tasks_by_cost,
        _resource_cost,
        build_compiler_edit_contract,
    )

    arms = analysis.get("arms")
    arms = arms if isinstance(arms, Mapping) else {}
    baseline = arms.get("baseline")
    baseline = baseline if isinstance(baseline, Mapping) else {}
    candidate = arms.get("candidate")
    candidate = candidate if isinstance(candidate, Mapping) else {}
    findings: list[dict[str, Any]] = []

    def add(
        kind: str,
        detail: str,
        effects: Sequence[str],
        *,
        evidence: Mapping[str, Any] | None = None,
        priority: int,
        cca_axes: Sequence[str] | None = None,
        magnitude_share: float | None = None,
        magnitude_basis: str | None = None,
    ) -> None:
        findings.append(
            {
                "kind": kind,
                "detail": detail,
                "required_effects": sorted(set(str(effect) for effect in effects)),
                "required_cca_axes": sorted(set(cca_axes) if cca_axes is not None else _FINDING_AXES.get(kind, ())),
                "evidence": dict(evidence or {}),
                "priority": priority,
                # HOW BIG, when the analysis can say. Findings used to be ordered by priority class and
                # then ALPHABETICALLY BY KIND, so among equals the first letter of a finding's name
                # decided what an agent worked on first. Measured on a real brief: the host hotspot
                # holding 45.7% of the host lane -- 549,695 of 1,202,745 dynamic operations, on a lane
                # that is >=93% of the measured window -- ranked SIXTH, below a synchronization finding
                # whose whole evidence was `{"baseline": 3, "candidate": 23}`. Twenty units outranked
                # half a million because "a" precedes "h".
                "magnitude_share": (
                    round(float(magnitude_share), 6)
                    if isinstance(magnitude_share, (int, float)) and not isinstance(magnitude_share, bool)
                    else None
                ),
                # Shares from different denominators are NOT comparable, so the basis travels with the
                # number and the ordering only ever compares within one kind's own class.
                "magnitude_basis": magnitude_basis,
            }
        )

    if candidate.get("status") == "declined":
        decline = candidate.get("declined")
        decline = decline if isinstance(decline, Mapping) else {}
        add(
            "whole_model_lowering_declined",
            "the candidate emitted no executable whole-model command stream",
            ("placement", "fusion", "movement", "residency", "issue", "latency_hiding"),
            evidence={"op": decline.get("op"), "reason": decline.get("reason"), "shape": decline.get("shape")},
            priority=-1,
        )

    candidate_representation = candidate.get("representation_activity")
    candidate_representation = candidate_representation if isinstance(candidate_representation, Mapping) else {}
    candidate_placement = candidate_representation.get("placement")
    candidate_placement = candidate_placement if isinstance(candidate_placement, Mapping) else {}
    lane_transitions = candidate_placement.get("adjacent_lane_transitions")
    if isinstance(lane_transitions, int) and not isinstance(lane_transitions, bool) and lane_transitions > 0:
        add(
            "inter_lane_boundary_pressure",
            "the complete model crosses compiler-declared lane boundaries that require explicit pricing",
            ("placement", "movement", "residency", "fusion", "latency_hiding"),
            evidence={
                "adjacent_lane_transitions": lane_transitions,
                "lane_counts": candidate_placement.get("lane_counts"),
            },
            priority=1,
        )

    weighted = analysis.get("model_contraction_placement")
    weighted = weighted if isinstance(weighted, Mapping) else {}
    weighted_candidate = weighted.get("candidate")
    weighted_candidate = weighted_candidate if isinstance(weighted_candidate, Mapping) else {}
    by_lane = weighted_candidate.get("mac_fraction_by_lane")
    if isinstance(by_lane, Mapping) and len(by_lane) > 1:
        add(
            "heterogeneous_model_placement",
            "captured contraction MAC work is split across multiple declared lanes",
            ("placement", "fusion", "movement"),
            evidence={
                "mac_fraction_by_lane": dict(by_lane),
                "total_contraction_macs": weighted_candidate.get("total_contraction_macs"),
                "unresolved_placement_macs": weighted_candidate.get("unresolved_placement_macs"),
            },
            priority=1,
        )
    memory_regime = weighted_candidate.get("memory_regime")
    memory_regime = memory_regime if isinstance(memory_regime, Mapping) else {}
    regime_counts = memory_regime.get("region_counts")
    if isinstance(regime_counts, Mapping) and regime_counts:
        add(
            "memory_regime_pressure",
            "captured contractions occupy different on-chip capacity regimes; overlap legality and "
            "reload policy must follow the regime rather than one universal schedule",
            ("residency", "movement", "tiling", "latency_hiding"),
            evidence={
                "region_counts": dict(regime_counts),
                "mac_fraction_by_regime": memory_regime.get("mac_fraction_by_regime"),
                "double_buffer_eligible_regime": memory_regime.get("double_buffer_eligible_regime"),
            },
            priority=1,
        )

    # A capability the hardware declares and the program never reaches for. This is the finding
    # that names a STRUCTURAL lever rather than a cheaper spelling of the current one: measured on
    # a whole-model emission, 8 of 25 declared functs were used, and the 17 unused included the
    # entire device-side convolution sequencer -- so every convolution's patch generation ran as
    # host scalar code. No other signal in this brief could have said that: the command buffer was
    # well formed, the gate passed, and the counters looked busy.
    utilization = analysis.get("isa_capability_utilization")
    if isinstance(utilization, Mapping) and utilization.get("unused"):
        declared = utilization.get("declared_count")
        used = utilization.get("used_count")
        add(
            "declared_capability_unused",
            "the target declares instructions this program never emits; each is a capability the "
            "compiler does not reach for, and the largest structural levers hide there",
            ("placement", "issue", "movement", "latency_hiding", "tiling"),
            evidence={
                "declared_count": declared,
                "used_count": used,
                "used": utilization.get("used"),
                "unused": utilization.get("unused"),
                "undeclared_emitted": utilization.get("undeclared_emitted"),
                "licence": utilization.get("licence"),
            },
            priority=0,
            magnitude_share=(
                round(1.0 - used / declared, 6)
                if isinstance(declared, int) and declared and isinstance(used, int)
                else None
            ),
            magnitude_basis="fraction of the target's declared instructions never emitted",
        )

    # WHY a declared capability is absent, not merely THAT it is. The compiler's own selector
    # names the deciding clause for every refused site and callers routinely discard it
    # (`eligible, _reason = select(...)`), which is the only machine-readable statement of what
    # would have to change. Ranked above the unused-capability finding because it is strictly more
    # actionable: one says "you never emit this", the other says "you were refused, here is the
    # clause". The cascade caveat travels with it -- clearing the top clause reveals the next.
    refusals = analysis.get("capability_refusals")
    if isinstance(refusals, Sequence) and not isinstance(refusals, (str, bytes)):
        for entry in refusals:
            if not isinstance(entry, Mapping) or not entry.get("clauses"):
                continue
            refused = entry.get("sites_refused") or 0
            total = entry.get("sites_total") or 0
            add(
                "declared_capability_refused",
                "the compiler's own selector refused a capability the target declares, and named "
                "the deciding clause; each clause is a stated precondition to change",
                ("placement", "issue", "movement", "latency_hiding", "tiling"),
                evidence={
                    "capability": entry.get("capability"),
                    "sites_refused": refused,
                    "sites_total": total,
                    "clauses": entry.get("clauses"),
                    "first_refusal_only": entry.get("first_refusal_only"),
                    "caveat": entry.get("caveat"),
                },
                priority=0,
                magnitude_share=(round(refused / total, 6) if total else None),
                magnitude_basis="fraction of candidate sites refused this capability",
            )

    verified_plan = analysis.get("verified_global_plan_emission")
    verified_plan = verified_plan if isinstance(verified_plan, Mapping) else {}
    host_activity = verified_plan.get("host_activity")
    host_activity = host_activity if isinstance(host_activity, Mapping) else {}
    host_tasks = host_activity.get("top_tasks_by_scalar_memory_payload")
    host_allocations = host_activity.get("top_allocations_by_static_payload")
    host_buffers = host_activity.get("top_buffers_by_scalar_memory_payload")
    if verified_plan.get("status") == "verified" and any(
        isinstance(rows, list) and rows for rows in (host_tasks, host_allocations, host_buffers)
    ):
        add(
            "host_memory_hotspot",
            "source-derived allocations and host loops expose materialization and scalar-work hotspots; "
            "smaller emitted IR alone does not delete this dynamic work",
            ("fusion", "movement", "placement", "layout"),
            evidence={
                "status": host_activity.get("status", "UNKNOWN"),
                "top_tasks": host_tasks[:5] if isinstance(host_tasks, list) else [],
                "top_allocations": host_allocations[:3] if isinstance(host_allocations, list) else [],
                "top_buffers": host_buffers[:3] if isinstance(host_buffers, list) else [],
                "artifact_sha256": host_activity.get("artifact_sha256"),
                "allocation_identity_scope": host_activity.get("memory_hotspot_identity_scope", "UNKNOWN"),
                "load_payload_bytes": host_activity.get("load_payload_bytes"),
                "store_payload_bytes": host_activity.get("store_payload_bytes"),
                "scope": "pre-optimization LLVM scalar payload, not DRAM or CPU cycles; "
                "static allocation payload is not stack-frame size or live-memory peak",
                "cost_ranked_tasks": _host_tasks_by_cost(host_activity),
                # The emitter shapes that dominate, as OPERATION SEQUENCES. This is the
                # instrument that turned "67% integer arithmetic" into three named
                # primitives -- a float max emulated as `fsub ashr and and xor or`, a
                # round-to-even emulated as ten integer ops, an im2col row loop peeling
                # coordinates with `udiv urem` -- each of which became a lever worth
                # 10-45% of the host lane. Without the sequence an author is told a
                # family and a task and must guess the primitive.
                "dominant_block_signatures": _dominant_signatures(host_activity),
            },
            # Class 0, alongside the regressions, and carrying a MEASURED share so it orders ahead
            # of them: this is the only finding in the brief that says how much of the dominant
            # lane it accounts for, and the host lane is >=93% of the measured window on the one
            # run with a counter partition. It sat at priority 1 -- behind every regression -- so
            # the agent's ranked work list opened with deltas of tens of units while half the host
            # lane went unmentioned until rank six.
            priority=0,
            magnitude_share=_host_hotspot_share(host_activity),
            magnitude_basis="largest host task's share of host-lane dynamic operations",
        )

    baseline_macs, candidate_macs = baseline.get("macs"), candidate.get("macs")
    if (
        isinstance(baseline_macs, (int, float))
        and not isinstance(baseline_macs, bool)
        and isinstance(candidate_macs, (int, float))
        and not isinstance(candidate_macs, bool)
        and candidate_macs > baseline_macs
    ):
        add(
            "arithmetic_expansion",
            "candidate emits more arithmetic for the same frozen model",
            ("fusion", "tiling"),
            evidence={"baseline_macs": baseline_macs, "candidate_macs": candidate_macs},
            priority=0,
        )

    baseline_movement = baseline.get("movement")
    baseline_movement = baseline_movement if isinstance(baseline_movement, Mapping) else {}
    candidate_movement = candidate.get("movement")
    candidate_movement = candidate_movement if isinstance(candidate_movement, Mapping) else {}
    base_bytes, cand_bytes = (baseline_movement.get("known_bytes"), candidate_movement.get("known_bytes"))
    if (
        isinstance(base_bytes, (int, float))
        and not isinstance(base_bytes, bool)
        and isinstance(cand_bytes, (int, float))
        and not isinstance(cand_bytes, bool)
        and cand_bytes > base_bytes
    ):
        add(
            "movement_regression",
            "candidate declares more command-buffer movement than baseline",
            ("movement", "residency", "layout", "encoding"),
            evidence={"baseline_known_bytes": base_bytes, "candidate_known_bytes": cand_bytes},
            priority=0,
        )
    if candidate_movement.get("is_lower_bound") is True:
        add(
            "unpriced_movement",
            "some candidate movement is not represented by an exact byte count",
            ("movement", "residency", "encoding"),
            evidence={"refusals": list(candidate_movement.get("refusals") or ())},
            priority=1,
        )

    barriers = analysis.get("barriers")
    barriers = barriers if isinstance(barriers, Mapping) else {}
    if barriers.get("status") == "counted":
        removed = barriers.get("removed")
        if isinstance(removed, int) and not isinstance(removed, bool) and removed < 0:
            add(
                "synchronization_regression",
                "candidate added completion points to the complete-model command stream",
                ("synchronization", "issue", "latency_hiding"),
                evidence={
                    "baseline_barriers": barriers.get("baseline_barriers"),
                    "candidate_barriers": barriers.get("candidate_barriers"),
                },
                priority=0,
            )

    levels = analysis.get("structural_levels")
    levels = levels if isinstance(levels, Mapping) else {}
    candidate_levels = levels.get("candidate")
    candidate_levels = candidate_levels if isinstance(candidate_levels, Mapping) else {}
    for raw in candidate_levels.get("findings") or ():
        if not isinstance(raw, Mapping):
            continue
        kind = str(raw.get("kind") or "structural_inefficiency")
        effects = _STRUCTURAL_EFFECTS.get(kind, frozenset({"movement", "fusion"}))
        add(
            kind,
            str(raw.get("detail") or "candidate has a structural inefficiency"),
            effects,
            evidence={key: raw[key] for key in ("level", "value", "at_command") if key in raw},
            priority=1,
        )

    representation = candidate.get("representation_activity")
    representation = representation if isinstance(representation, Mapping) else {}
    emitted = representation.get("emitted_encoding_transitions")
    emitted = emitted if isinstance(emitted, Mapping) else {}
    if emitted.get("status") != "measured":
        add(
            "unverified_encoding_transitions",
            "declared encodings are visible, but executed conversions have not been counted",
            ("encoding", "layout", "fusion"),
            evidence={
                "declared_directives": emitted.get("declared_directives"),
                "status": emitted.get("status", "UNKNOWN"),
            },
            priority=2,
        )
    occupancy = representation.get("occupancy")
    occupancy = occupancy if isinstance(occupancy, Mapping) else {}
    if occupancy.get("status") != "measured":
        add(
            "unknown_occupancy_and_overlap",
            "compute occupancy and movement/compute overlap require an event adapter or warm counters",
            ("latency_hiding", "issue", "tiling", "movement", "residency"),
            evidence={"status": occupancy.get("status", "UNKNOWN")},
            priority=2,
        )

    artifact = analysis.get("target_artifact_activity")
    artifact = artifact if isinstance(artifact, Mapping) else {}
    artifact_baseline = artifact.get("baseline")
    artifact_baseline = artifact_baseline if isinstance(artifact_baseline, Mapping) else {}
    artifact_candidate = artifact.get("candidate")
    artifact_candidate = artifact_candidate if isinstance(artifact_candidate, Mapping) else {}
    base_issued = artifact_baseline.get("issued")
    base_issued = base_issued if isinstance(base_issued, Mapping) else {}
    cand_issued = artifact_candidate.get("issued")
    cand_issued = cand_issued if isinstance(cand_issued, Mapping) else {}

    def _numeric_delta(field: str) -> tuple[int | float, int | float] | None:
        left, right = base_issued.get(field), cand_issued.get(field)
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
        ):
            return left, right
        return None

    delta = _numeric_delta("movement_instructions")
    if delta is not None and delta[1] > delta[0]:
        add(
            "issued_movement_regression",
            "candidate issues more movement instructions in the lowered target artifact",
            ("movement", "residency", "layout", "encoding"),
            evidence={"baseline": delta[0], "candidate": delta[1]},
            priority=0,
        )
    delta = _numeric_delta("configuration_instructions")
    if delta is not None and delta[1] > delta[0]:
        add(
            "dispatch_configuration_regression",
            "candidate reconfigures the target endpoint more often",
            ("issue", "latency_hiding", "synchronization"),
            evidence={"baseline": delta[0], "candidate": delta[1]},
            priority=0,
        )
    delta = _numeric_delta("synchronization_instructions")
    if delta is not None and delta[1] > delta[0]:
        add(
            "artifact_synchronization_regression",
            "candidate issues more synchronization in the lowered target artifact",
            ("synchronization", "issue", "latency_hiding"),
            evidence={"baseline": delta[0], "candidate": delta[1]},
            priority=0,
        )
    base_loop = base_issued.get("loop_descriptor_instructions")
    cand_loop = cand_issued.get("loop_descriptor_instructions")
    if (
        isinstance(base_loop, int)
        and not isinstance(base_loop, bool)
        and isinstance(cand_loop, int)
        and not isinstance(cand_loop, bool)
        and base_loop > 0
        and cand_loop == 0
    ):
        add(
            "loop_offload_regression",
            "candidate removed the target's emitted loop-descriptor path",
            ("issue", "tiling", "latency_hiding"),
            evidence={"baseline": base_loop, "candidate": cand_loop},
            priority=0,
        )

    # EVERY SCORED TERM AT ONCE, AND WHAT THE EDIT TRADED. Each finding above judges ONE term in
    # isolation and calls any rise a regression. That rule cannot express the edit this loop most
    # needs to recognise: one that moves work BETWEEN resources. Measured on one model, moving a
    # quantization epilogue onto the accelerator's store path removed 146.2M host operations and
    # raised accelerator dispatches 11.6x; decomposing convolutions into accumulating matmuls
    # removed 553.3M host operations and raised movement 24%. Both are plausibly large wins and
    # both score as losses term by term. `cost_terms.compare_terms` keeps the terms separate and
    # prices a composite ONLY from rates whose provenance is recorded -- so an unpriced term makes
    # the composite UNPRICED rather than silently weighing it zero, which is the bug that hides a
    # resource the edit actually spent.
    resource_before = {
        "contraction_macs": baseline_macs,
        "declared_movement_bytes": base_bytes,
        "completion_points": barriers.get("baseline_barriers"),
        "issued_movement_instructions": base_issued.get("movement_instructions"),
        "issued_configuration_instructions": base_issued.get("configuration_instructions"),
        "issued_synchronization_instructions": base_issued.get("synchronization_instructions"),
        "issued_loop_descriptor_instructions": base_issued.get("loop_descriptor_instructions"),
    }
    resource_after = {
        "contraction_macs": candidate_macs,
        "declared_movement_bytes": cand_bytes,
        "completion_points": barriers.get("candidate_barriers"),
        "issued_movement_instructions": cand_issued.get("movement_instructions"),
        "issued_configuration_instructions": cand_issued.get("configuration_instructions"),
        "issued_synchronization_instructions": cand_issued.get("synchronization_instructions"),
        "issued_loop_descriptor_instructions": cand_issued.get("loop_descriptor_instructions"),
    }
    resource_cost = _resource_cost(
        resource_before,
        resource_after,
        analysis.get("resource_rates"),
        analysis.get("design"),
    )
    if resource_cost["terms_increased"]:
        add(
            "resource_trade",
            "this candidate reduced one scored resource and raised another; that is a BET, not a "
            "regression, and it cannot be judged without an exchange rate between the two",
            ("placement", "fusion", "movement", "issue", "latency_hiding"),
            evidence={
                "terms_increased": resource_cost["terms_increased"],
                "composite_status": resource_cost["composite_status"],
                "composite_cycle_delta": resource_cost["composite_cycle_delta"],
                "unpriced_moved_terms": resource_cost["unpriced_moved_terms"],
                "terms": resource_cost["terms"],
                "reading": resource_cost["reading"],
            },
            # Class 0 beside the per-term regressions it reinterprets: read after them, an author is
            # told to undo an edit the composite may show was a win.
            priority=0,
        )

    encoding = artifact_candidate.get("encoding_resolution")
    encoding = encoding if isinstance(encoding, Mapping) else {}
    if artifact_candidate and encoding.get("status") != "complete":
        add(
            "unresolved_target_encoding",
            "the emitted artifact contains instructions without a complete semantic-role decode",
            ("encoding", "layout"),
            evidence={
                "status": encoding.get("status", "UNKNOWN"),
                "unknown_instruction_indices": list(encoding.get("unknown_instruction_indices") or ()),
                "named_without_role": list(encoding.get("named_without_role") or ()),
            },
            priority=1,
        )

    for raw in artifact_candidate.get("artifact_opportunities") or ():
        if not isinstance(raw, Mapping) or not isinstance(raw.get("axis"), str):
            continue
        axis = str(raw["axis"])
        prefix = axis.partition(".")[0]
        effects = {
            "compute": ("fusion", "tiling"),
            "memory": ("movement", "residency", "layout"),
            "layout": ("layout", "encoding", "movement"),
            "dispatch": ("issue", "latency_hiding", "synchronization"),
            "communication": ("movement", "residency", "latency_hiding"),
            "simt": ("synchronization", "latency_hiding"),
            "envelope": ("fusion", "issue"),
        }.get(prefix, ("issue",))
        add(
            "artifact_cca_opportunity",
            str(raw.get("observation") or "emitted role shape exposes an optimization"),
            effects,
            evidence={
                key: raw[key]
                for key in ("axis", "change", "seam", "forkable_now", "status", "confidence")
                if key in raw
            },
            priority=1,
            cca_axes=(axis,),
        )

    trace = analysis.get("trace_conformance")
    trace = trace if isinstance(trace, Mapping) else {}
    introduced = trace.get("introduced_candidate_findings")
    if isinstance(introduced, Sequence) and not isinstance(introduced, (str, bytes)) and introduced:
        add(
            "trace_regression",
            "candidate introduced decoded trace findings absent from the frozen Phase-1 baseline",
            ("residency", "movement", "synchronization", "encoding", "issue"),
            evidence={"introduced_findings": list(introduced)},
            priority=0,
        )

    def _order(row: Mapping[str, Any]) -> tuple[int, int, float, str]:
        """Priority class, then SIZE, then name -- the name only as a last resort tie-break.

        A finding whose size the analysis could measure is ordered ahead of one it could not,
        because "we know this is 45.7% of the lane" is strictly better guidance than "this exists".
        """
        share = row.get("magnitude_share")
        sized = isinstance(share, (int, float)) and not isinstance(share, bool)
        return (int(row["priority"]), 0 if sized else 1, -float(share) if sized else 0.0, str(row["kind"]))

    ranked: list[dict[str, Any]] = []
    for rank, finding in enumerate(sorted(findings, key=_order), start=1):
        required = set(finding["required_effects"])
        required_axes = set(finding["required_cca_axes"])
        axis_matches = [surface for surface in inventory.surfaces if required_axes.intersection(surface.cca_axes)]
        effect_matches = [surface for surface in inventory.surfaces if required.intersection(surface.effects)]
        matches = axis_matches or effect_matches
        row = dict(finding)
        row.pop("priority", None)
        row.update(
            {
                "rank": rank,
                "status": "actionable" if matches else "unmapped",
                "edit_surfaces": [surface.to_dict() for surface in matches],
                "mapping_basis": (
                    "exact CCA axis" if axis_matches else "broad effect compatibility" if effect_matches else None
                ),
                "next_step": (
                    "edit one listed path/symbol, rerun inspect-optimization-surfaces, then rerun "
                    "analyze-whole-model; use a short mechanism-equivalent probe only for uncertain costs"
                    if matches
                    else "add an optimization_surfaces entry to manifest.yaml for the real owning AST "
                    "symbol; the host will verify path and symbol ownership, then analyze whether the "
                    "declared emitted delta actually appears"
                ),
            }
        )
        ranked.append(row)

    movement_status = (
        "unavailable: lowering declined"
        if candidate.get("status") == "declined"
        else "lower_bound"
        if candidate_movement.get("is_lower_bound") is True
        else "declared_exact"
        if candidate_movement
        else "UNKNOWN"
    )
    trace_status = str(trace.get("status") or "UNKNOWN")
    artifact_status = str(artifact_candidate.get("status") or "UNKNOWN")
    mechanism_coverage = {
        "arithmetic_demand": "observed" if candidate_macs is not None else "UNKNOWN",
        "whole_model_lowering": str(candidate.get("status") or "UNKNOWN"),
        "declared_movement_volume": movement_status,
        "issued_movement": artifact_status,
        "host_scalar_work": str(host_activity.get("status") or "UNKNOWN"),
        "target_encoding": str(encoding.get("status") or "UNKNOWN"),
        "dispatch_shape": artifact_status,
        "synchronization": artifact_status,
        "residency_reload_defect": trace_status,
        "occupancy": "UNKNOWN for full model; bound reduced profiles are reported separately as scoped evidence",
        "contention": "UNKNOWN for full model; executed reduced counters do not establish model-wide contention",
        "cycle_ordering": "UNMEASURED: cheap signals are not validated schedule rankers",
    }

    placement = representation.get("placement")
    placement = placement if isinstance(placement, Mapping) else {}
    candidate_cca = artifact_candidate.get("program_cca")
    candidate_cca = candidate_cca if isinstance(candidate_cca, Mapping) else {}
    candidate_dispatch = candidate_cca.get("dispatch")
    candidate_dispatch = candidate_dispatch if isinstance(candidate_dispatch, Mapping) else {}
    overlap = candidate_dispatch.get("dma_overlap")

    def needs_evidence(status: str) -> bool:
        normalized = status.strip().casefold()
        return (
            "unknown" in normalized
            or "unmeasured" in normalized
            or normalized.startswith(
                (
                    "refused",
                    "declined",
                    "unavailable",
                    "failed",
                    "error",
                    "invalid",
                    "unsupported",
                    "missing",
                    "unresolved",
                    "not_",
                    "not ",
                )
            )
        )

    level_status = str(candidate_levels.get("status") or "UNKNOWN")
    structural_status = (
        level_status
        if candidate_levels.get("status") and needs_evidence(level_status)
        else "screened"
        if isinstance(candidate_levels.get("findings"), Sequence)
        and not isinstance(candidate_levels.get("findings"), (str, bytes))
        else "UNKNOWN"
    )
    observability = {
        "whole_model_placement_and_coverage": str(placement.get("status") or "UNKNOWN"),
        "arithmetic_lowering": mechanism_coverage["arithmetic_demand"],
        "encoding_and_layout": mechanism_coverage["target_encoding"],
        "movement_and_materialization": mechanism_coverage["issued_movement"],
        "residency_across_operations": mechanism_coverage["residency_reload_defect"],
        "fusion_and_host_boundaries": structural_status,
        "dispatch_and_loop_offload": mechanism_coverage["dispatch_shape"],
        "latency_hiding_and_double_buffering": (
            "structurally_observed"
            if isinstance(overlap, bool)
            else "UNKNOWN: emitted DMA/wait evidence and scoped warm profiles do not alone prove full-model latency hiding"
        ),
        "synchronization": mechanism_coverage["synchronization"],
        "capacity_and_contention": (
            "capacity_regime_derived; dynamic_contention=UNKNOWN"
            if memory_regime.get("status") == "derived"
            else "UNKNOWN: target capacity/resource evidence is not joined"
        ),
        "exact_quantized_epilogues_and_residuals": "UNKNOWN: operation counts do not establish staged rounding equivalence or second-operand offload",
        "arena_lifetimes_and_reuse": "UNKNOWN: static allocation totals do not prove aliasing, live ranges or asynchronous completion",
        "entry_abi_and_runtime_overhead": "UNKNOWN: task counts do not establish entry pointer binding or runtime overhead",
    }
    cheap_validation = {
        "whole_model_placement_and_coverage": "re-emit fixed full-model capture and compare placement",
        "arithmetic_lowering": "re-emit and compare exact command/artifact work",
        "encoding_and_layout": "decode emitted target artifact and count unresolved transitions",
        "movement_and_materialization": "compare declared bytes and issued movement instructions",
        "residency_across_operations": "run exact decoded reload/liveness checks",
        "fusion_and_host_boundaries": "compare materializations, host boundaries, and producer uses",
        "dispatch_and_loop_offload": "compare role-tagged config and loop-descriptor issue",
        "latency_hiding_and_double_buffering": "static DMA/wait shape, then one reduced warm counter run",
        "synchronization": "compare command completion points and emitted sync instructions",
        "capacity_and_contention": "requires a target resource adapter plus reduced pressure witness",
        "exact_quantized_epilogues_and_residuals": "clone the actual typed epilogue/residual chain; preserve each rounding stage and compare a short independent-reference witness",
        "arena_lifetimes_and_reuse": "prove buffer aliases, last uses and device completion on the full plan; compare planned live bytes and check a short reuse hazard witness",
        "entry_abi_and_runtime_overhead": "verify emitted entry signature and base/offset bindings against the full CFG; qualify a short address-binding mechanism",
    }
    gap_coverage: list[dict[str, Any]] = []
    for gap, axes in _GAP_AXES.items():
        matches = [surface for surface in inventory.surfaces if axes.intersection(surface.cca_axes)]
        observed = observability[gap]
        unresolved = needs_evidence(observed)
        gap_coverage.append(
            {
                "gap": gap,
                "evidence_status": observed,
                "cca_axes": sorted(axes),
                "edit_surfaces": [surface.to_dict() for surface in matches],
                "edit_status": "mapped" if matches else "missing_verified_surface",
                "cheap_validation": cheap_validation[gap],
                "coverage_status": (
                    "needs_evidence" if unresolved else "needs_edit_surface" if not matches else "ready"
                ),
            }
        )

    plan_identity_fields = (
        "candidate_sha256",
        "source_sha256",
        "logical_dispatch_digest",
        "plan_digest",
        "candidate_lowered_sha256",
        "candidate_command_buffer_sha256",
    )
    plan_identities = {key: verified_plan.get(key) for key in plan_identity_fields}
    bound_plan = verified_plan.get("status") == "verified" and all(
        isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)
        for value in plan_identities.values()
    )
    owned_counts = verified_plan.get("emitted_operations_by_task")
    temporaries = verified_plan.get("compiler_temporaries")
    plan_cfg = verified_plan.get("control_flow")
    plan_cfg = plan_cfg if isinstance(plan_cfg, Mapping) else {}
    compiler_plan = {
        "status": "verified"
        if bound_plan
        else (
            "UNKNOWN: verified plan lacks complete artifact identities"
            if verified_plan.get("status") == "verified"
            else str(verified_plan.get("status") or "UNKNOWN")
        ),
        "identities": plan_identities,
        "source_operations": verified_plan.get("source_operations"),
        "tasks": verified_plan.get("tasks"),
        "owned_emitted_operation_count": sum(owned_counts.values())
        if bound_plan
        and isinstance(owned_counts, Mapping)
        and all(type(value) is int and value >= 0 for value in owned_counts.values())
        else None,
        "compiler_temporary_count": len(temporaries) if bound_plan and isinstance(temporaries, list) else None,
        "control_flow": {
            key: plan_cfg.get(key, "UNKNOWN")
            for key in ("status", "reachable_blocks", "return_blocks", "proof_scope", "loop_trip_count_equivalence")
        },
        "proof_scope": verified_plan.get("proof_scope", "UNKNOWN"),
        "numerical_equivalence": verified_plan.get("numeric_equivalence", "UNPROVEN"),
        "costs": "UNKNOWN",
        "optimality": "UNPROVEN",
    }

    return {
        "schema": "agent_emission_optimization_brief_v1",
        "objective": "minimize warm complete-model compute cycles",
        "optimization_order": macro_optimization_order(),
        "timing_status": "UNMEASURED",
        "ranked_actions": ranked,
        "package_inventory": inventory.to_dict(),
        "compiler_edit_contract_template": build_compiler_edit_contract(inventory),
        "mechanism_coverage": mechanism_coverage,
        # EVERY RESOURCE THIS EDIT MOVED, PRICED ONLY WHERE A RATE WAS MEASURED. Rides beside the
        # ranked actions rather than inside one, because it is the table a reader needs to judge
        # any of them: the ranked findings each name a single term, and the question "was the rise
        # in that term paid for by the fall in another" is answerable from here alone.
        "resource_cost": resource_cost,
        "gap_coverage": gap_coverage,
        "global_planner_wiring": {
            "status": "compiler_owned_plan_verified" if bound_plan else "UNKNOWN",
            "compiler_owned_plan": compiler_plan,
            "shared_exact_cover_search": {
                "status": "UNKNOWN",
                "proof": None,
                "scope": "a verified compiler-owned plan does not establish use of the shared exact-cover solver",
            },
            "required": (
                "optimize and re-verify the existing compiler-owned full-model plan; use focused reduced "
                "witnesses for changed semantics and uncertain costs. If adopting shared exact-cover search, "
                "bind actual solver selection through a target planning adapter and emitted-plan receipt"
                if bound_plan
                else "provide a source/artifact-bound compiler-owned plan with complete task, ABI and CFG coverage; "
                "shared TargetPlanningAdapter integration is a separate optional search route"
            ),
            "available_shared_blocks": [
                "GlobalPlan exact-cover search",
                "explicit ValueRepresentation transitions",
                "ActivityTimeline dependency/resource scheduling",
                "PipelineProjection from reduced stage calibration",
                "verified GlobalPlanEmission accounting",
                "OutlinedGlobalPlanEmitter with expanded full-model IR equivalence proof",
            ],
            "captured_logical_graph": {
                key: value
                for key, value in (analysis.get("captured_logical_graph") or {}).items()
                if key != "dispatch_program"
            },
            "warning": (
                "shared planner classes existing in Merlin does not prove this compiler package "
                "selects or emits their plan"
            ),
        },
        "edit_protocol": [
            "select declared surface ids and state a source/plan-bound hypothesis, expected delta, semantic obligations and stopping condition",
            "edit only the host-frozen permitted package paths and AST symbols; a changed candidate manifest cannot expand permission",
            "add helpers only within explicit host-approved extension directories; otherwise report the missing surface",
            "refresh the optimization-surface inventory after the edit",
            "re-emit the fixed complete-model witness and verify the intended structural delta",
            "check changed semantics using a relative proof or a focused independent correctness witness",
            "only when a selection depends on uncertain cost, run one warm and one measured "
            "short mechanism-equivalent probe; never execute a full layer or model",
            "reject absent structural deltas or incorrect edits; retain uncalibrated structural "
            "improvements as unmeasured, never as demonstrated cycle speedups",
        ],
    }
