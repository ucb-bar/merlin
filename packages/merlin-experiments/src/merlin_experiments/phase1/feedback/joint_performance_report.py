"""Answer-free projections of original joint timing observations, never grants."""

import math

from ...phase2.contracts import document_sha256


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def project(report, observation, structural, *, selection_sha256, reason):
    """Keep unknown, provisional and qualified meanings separate.

    This initial observer issues no qualified status. Held source-group error,
    ordering, physical boundary and attainable-envelope qualification remain
    separate operator work; input dictionaries cannot grant that authority.
    """
    value = dict(
        schema="merlin.phase1.joint_performance.v1", status="UNKNOWN", reason=reason,
        selection_sha256=selection_sha256, structural_roofline=structural,
        estimated_whole_soc_cycles=None, structural_floor_to_estimate_ratio=None,
        qualified_attainable_fraction=None, performance_target_met=False,
        promotion_authority=False, timing_qualified=False,
        model_sha256=None, comparison_domain_sha256=None, scope_sha256=None,
        feature_counts={}, unresolved_events=[], modeled_resource_activity=None,
        complete_event_accounting=False,
        scope="complete kernel-call window; host, device and completion inside that window",
        scope_excludes=["input staging", "output publication", "metric formatting", "parent readback"],
    )
    failure_stage = observation.get("failure_stage")
    if failure_stage in {
        "runtime_validation", "policy_validation", "producer_validation", "namespace_setup",
        "scope_disassembly", "scope_binding", "functional_execution", "annotated_execution",
        "result_verification",
    }:
        wall = observation.get("wall_seconds")
        value["observation_diagnostics"] = {
            "failure_stage": failure_stage,
            "deadline_exhausted": observation.get("deadline_exhausted") is True,
            "host_wall_seconds": wall if type(wall) in (int, float) and math.isfinite(wall) and wall >= 0 else None,
            "scope": "inclusive host observation/setup wall time; not target execution cycles",
            "authority": "failure localization only; no numerical or timing qualification",
        }
    if report is None or observation.get("status") != "observed":
        return value
    windows = report.get("counter_windows")
    cycles = report.get("estimated_cycles")
    binding = observation.get("scope_binding") or {}
    valid_scope = (
        isinstance(windows, list) and len(windows) == 1
        and windows[0].get("estimated_cycles") == cycles
        and all(windows[0].get(key) == binding.get(key) for key in ("begin_pc", "end_pc"))
        and report.get("scope_sha256") == binding.get("scope_sha256")
    )
    if not valid_scope:
        value["reason"] = "native estimate lost its original selected counter window"
        return value
    counts = report.get("feature_counts") or {}
    # Counts are descriptive, not cycles; no tensor values or native paths leave
    # this adapter. Names come from the selected observer vocabulary.
    value["feature_counts"] = {k: v for k, v in counts.items()
                               if isinstance(k, str) and len(k) <= 256 and type(v) is int and v >= 0}
    value.update(
        model_sha256=report.get("model_sha256"), scope_sha256=report.get("scope_sha256"),
        report_sha256=(observation.get("report") or {}).get("sha256"),
        comparison_domain_sha256=document_sha256(report.get("domain")),
        unresolved_events=[str(v)[:256] for v in report.get("unresolved", [])[:64]],
    )
    complete = (observation.get("complete_counts") is True
                and report.get("timing_complete") is True and report.get("unresolved") == [])
    value["complete_event_accounting"] = complete
    # Existing Spike resource totals describe whole_trace, not the chosen call.
    # Publish the distinction explicitly; do not form percentages or add them.
    breakdown = report.get("cycle_breakdown") or {}
    if breakdown.get("scope") == "whole_trace":
        value["modeled_resource_activity"] = dict(
            scope="whole_trace", additive=False, usable_as_kernel_decomposition=False,
            queue_scope="generic resource waiting queues; not accelerator reservation-station occupancy",
            unavailable=["reservation_station_highwater", "admission_wait_cycles",
                         "dependency_wait_cycles", "ready_stall_cycles"],
            resources=[{k: r.get(k) for k in ("resource_id", "reservations", "busy_unit_cycles",
                                              "unknown_events", "peak_queue_entries")}
                       for r in breakdown.get("resources", [])[:128]],
        )
    if _positive(cycles) and complete:
        floor = (structural or {}).get("structural_compute_floor_cycles")
        if _positive(floor) and cycles < floor:
            value.update(status="REFUTED", reason="estimate is below structural compute floor")
        else:
            value.update(status="PROVISIONAL", estimated_whole_soc_cycles=cycles,
                         structural_floor_to_estimate_ratio=floor / cycles if _positive(floor) else None)
    return value


def compare(baseline, candidate):
    """Compare fixed-evaluator provisional estimates, never choose an incumbent."""
    keys = ("capsule_sha256", "selection_sha256", "model_sha256", "comparison_domain_sha256", "scope_sha256")
    if (any(baseline.get(k) != candidate.get(k) or not baseline.get(k) for k in keys)
            or any(row.get("status") != "PROVISIONAL" or not _positive(row.get("estimated_whole_soc_cycles"))
                   for row in (baseline, candidate))):
        return dict(status="UNKNOWN", candidate_to_baseline_ratio=None,
                    reason="same evaluator, complete scope and provisional estimates required")
    return dict(status="PROVISIONAL", candidate_to_baseline_ratio=(
        candidate["estimated_whole_soc_cycles"] / baseline["estimated_whole_soc_cycles"]),
        timing_qualified=False, promotion_authority=False,
        matched_software_library_target_met=None)
