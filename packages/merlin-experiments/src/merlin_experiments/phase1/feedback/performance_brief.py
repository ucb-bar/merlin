"""A short performance brief inside every self-check response, where the author actually reads.

The ranked headroom list and the architecture-selected mechanisms lived in the verdict and in a
workspace file. Measured on one attempt: the author ran 56 self-checks and read their cycle and
floor fields 20 times, but opened the verdict's ranked list once, never opened the mechanism
suggestions and never ran the per-command debugger, and measured cycles did not move. This brief
repeats, for the test cases of the request itself, what those other places already say: cycles with
their evidence kind, the roofline bound and gap, array utilization, the busiest modeled resource,
up to four catalogue mechanisms matching that bottleneck and, for public roster cases whose command
buffer was copied to the artifact folder, the exact debugger command.

The objective is fewer complete elapsed cycles for the same results. Primary
workload roles lead exploration; small controls retain diagnostic records. Useful
compute-ceiling attainment is not measured physical PE occupancy, and modeled
resource reservations alone do not establish bottlenecks or overlap.

Each entry's ``fence_overlap`` (:mod:`.fence_overlap`) counts the full-device fences of the case's
command stream, output tiles per fence and the static overlap ratio; ``fence_bound`` lists the passing
cases fenced after nearly every output tile, with or without cycle evidence, and the catalogue
mechanisms that address it. ``tuning_knob_candidates`` (:mod:`.knob_lint`, added by the broker) names
policy switches in ``submission/`` worth declaring as tuning knobs. A ranked case with real headroom and
a matching library mechanism also gets ``library_pass`` in its ``optimization_next`` (:mod:`.library_pass`).
Every advised case whose declaration derives an operation family staged in ``OPTIMIZATION_BY_OPERATION.json``
gets ``operation`` there too: the family's entry id, its focus and the panel command
(:mod:`..operation_optimizations`).

Everything is advisory, bounded and repeats only numbers the response already carries or reads from
the emitted artifacts it names; it changes no status or count and never fails a check.
"""

from __future__ import annotations

from pathlib import Path

from . import capacity_use, fence_overlap, mechanism_hints, model_calibration, reference_baselines, shape_classes
from .performance_headroom import _bound, _dominant_resource, _positive

SCHEMA = "merlin.selfcheck_performance_brief.v1"
#: Entries kept per response; the counts cover the rest.
LIMIT = 12
#: Entries in the utilization ranking.
UTILIZATION_LIMIT = 6
RANKED_LIST = ("Primary workload ranking and separate control diagnostics: qa/latest_completed_grade.json -> "
               "feedback_brief.performance_priority.development_headroom (qa/verdict.json while current).")


def _passes(row: dict) -> bool:
    tiers = row.get("tiers")
    if isinstance(tiers, dict) and "L2" in tiers:
        tier = tiers["L2"]
        return (tier.get("status") if isinstance(tier, dict) else tier) == "pass"
    return row.get("pass") is True or row.get("numerical_execution_passed") is True


def _evidence(row: dict):
    """``(cycles, roofline, label)`` from the row's own measurement, or None."""
    performance = row.get("performance") if isinstance(row.get("performance"), dict) else {}
    roofline = performance.get("structural_roofline") or {}
    if row.get("barrier_tier") == "L3" and row.get("pass") is True and _positive(row.get("barrier_cycles")):
        return row["barrier_cycles"], roofline, "measured: gSIM cycles of this request"
    if (
        performance.get("status") in ("PROVISIONAL", "QUALIFIED")
        and performance.get("complete_event_accounting") is True
        and _positive(performance.get("estimated_whole_soc_cycles"))
    ):
        return (
            performance["estimated_whole_soc_cycles"],
            roofline,
            f"modeled: Spike timing model ({performance['status'].lower()})",
        )
    return None


def _inspect(name: str, artifact_root, workspace: Path) -> str | None:
    """The debugger command for a copied command buffer, as a workspace-relative path; else None."""
    if not name or "/" in name:
        return None
    root = Path(artifact_root or "selfcheck_out")
    buffer = (root if root.is_absolute() else workspace / root) / name / "command_buffer.json"
    try:
        relative = buffer.resolve().relative_to(workspace.resolve())
    except (OSError, ValueError):
        return None
    if not buffer.is_file():
        return None
    return f"python isa_tools.py debug {relative.as_posix()} --capsule {name}"


def build(rows, *, workspace=None, artifact_root=None, debuggable: bool = True) -> dict | None:
    """The brief for one response's rows (``per_capsule`` or ``cases``); None when nothing passed."""
    workspace = Path(workspace) if workspace is not None else Path.cwd()
    passing = [row for row in rows or () if isinstance(row, dict) and _passes(row)]
    if not passing:
        return None
    selection = mechanism_hints.load_selection(workspace)
    calibrated = model_calibration.calibration(workspace)
    entries, missing, fences = [], [], {}
    for row in passing:
        name = str(row.get("capsule") or row.get("name") or "")
        fence = fence_overlap.for_entry(
            name, workspace, row.get("performance"), artifact_root=artifact_root, selection=selection
        )
        if fence is not None:
            fences[name] = fence
        found = _evidence(row)
        bound, limiter = _bound(found[1]) if found else (None, None)
        if found is None or bound is None or found[0] < bound:
            missing.append(name)
            continue
        cycles, roofline, label = found
        entry = {
            "test_case": name,
            "cycles": cycles,
            "evidence": label,
            "roofline_bound_cycles": round(bound, 1),
            "bound_by": limiter,
            "gap_ratio": round(cycles / bound, 2),
            "headroom_cycles": round(cycles - bound, 1),
        }
        useful, peak = roofline.get("useful_macs"), roofline.get("structural_peak_macs_per_cycle")
        if _positive(useful) and _positive(peak):
            entry["array_utilization_pct"] = round(100.0 * useful / (cycles * peak), 2)
        dominant = _dominant_resource(row.get("performance") or {})
        if dominant is not None:
            entry["dominant_modeled_resource"] = dominant
        if fence is not None:
            entry["fence_overlap"] = fence
        reading = _reading(entry)
        if reading:
            entry["reading"] = reading
        suggested = mechanism_hints.hints(selection, dominant=dominant, bound_by=limiter)
        if suggested:
            entry["mechanism_basis"] = suggested[0].get("why")
            entry["suggested_mechanisms"] = [
                {key: hint.get(key) for key in ("id", "api", "facet", "workload_check")} for hint in suggested
            ]
        if label.startswith("modeled"):
            entry.update(
                model_calibration.apply(
                    cycles,
                    calibrated,
                    bound_class=_bound_class(limiter),
                    family=model_calibration.semantic_family(workspace, name),
                )
                or {}
            )
        command = _inspect(name, artifact_root, workspace) if debuggable else None
        if command:
            entry["inspect"] = command
        entries.append(entry)
    from ..workload_roles import PRIMARY, workspace_roles
    roles = workspace_roles(workspace)
    diagnostics = []
    if roles is not None:
        diagnostics = [entry for entry in entries if roles.get(entry['test_case']) not in PRIMARY]
        entries = [entry for entry in entries if roles.get(entry['test_case']) in PRIMARY]
    entries.sort(key=lambda e: (-e["headroom_cycles"], -e["gap_ratio"], e["test_case"]))
    by_name = {str(row.get("capsule") or row.get("name") or ""): row for row in passing}
    for entry in entries[:LIMIT] + diagnostics[:LIMIT]:
        entry["evidence_kind"] = shape_classes.evidence_kind(entry["evidence"])
        use = capacity_use.for_case(entry["test_case"], by_name.get(entry["test_case"]) or {}, workspace,
                                    artifact_root=artifact_root)
        if use is not None:
            entry["capacity_use"] = use
    baselines = reference_baselines.load(workspace)
    reference_baselines.annotate(entries, reference_baselines.index(baselines))
    compared = reference_baselines.summary(entries, baselines)
    brief = {
        "schema": SCHEMA, "advisory_only": True,
        "note": ("Performance of the passing test cases in this request, largest headroom first. The objective "
                 "is fewer cycles for the same results. Array utilization here is useful compute-ceiling "
                 "attainment, not physical PE occupancy. Resource values describe model reservations, "
                 "not a proven bottleneck. Suggested mechanisms match resource diagnostics, mechanisms "
                 "naming its unit first (OPTIMIZATION_SUGGESTIONS.json); compare variants on the same cases "
                 "by cycles before keeping one. " + RANKED_LIST),
        "counts": {"passing": len(passing), "with_evidence": len(entries), "without_evidence": len(missing)},
        "largest_headroom": entries[:LIMIT],
        "diagnostic_controls": diagnostics[:LIMIT],
        "diagnostic_control_count": len(diagnostics),
    }
    utilized = sorted(
        (e for e in entries if "array_utilization_pct" in e),
        key=lambda e: (e["array_utilization_pct"], -e["headroom_cycles"], e["test_case"]),
    )
    if utilized:
        brief["lowest_utilization"] = [
            {
                key: e[key]
                for key in ("test_case", "array_utilization_pct", "cycles", "evidence", "evidence_kind",
                            "dominant_modeled_resource")
                if key in e
            }
            for e in utilized[:UTILIZATION_LIMIT]
        ]
    from .shape_panel import evidence as cycles_of

    observed = [(name, found[0], found[2], row.get("performance"))
                for name, row in by_name.items() if (found := cycles_of(row))[0]]
    line = shape_classes.vs_best_line(observed, workspace)
    if line:
        brief["vs_best_per_class"] = line
    fence_bound = fence_overlap.summary(fences, selection)
    if fence_bound is not None:
        brief["fence_bound"] = fence_bound
    if compared is not None:
        brief["reference_baselines"] = {**compared, "line": reference_baselines.headline_line(compared)}
    if calibrated is not None:
        brief["model_calibration"] = _summary(calibrated)
    if missing:
        brief["without_evidence"] = sorted(missing)[:10]
        structural = [name for name in missing if roles is None or roles.get(name) in PRIMARY]
        if structural:
            brief["structural_exploration"] = {
                "cases": sorted(set(structural))[:LIMIT],
                "basis": "candidate selection for investigation; no cycle or speedup claim",
                "action": ("Continue optimization while timing coverage is repaired. Use legal pass/schedule "
                           "alternatives and inspect their emitted products: operand traffic and reloads, "
                           "packing frequency, storage lifetimes, tile coverage, completion boundaries and "
                           "CPU work. Retain alternatives with different resource tradeoffs; check affected "
                           "Spike numerics and independent shapes/tails. See explore-schedules and "
                           "cycle-model-and-gsim. UNKNOWN stays outside cycle rankings; fewer instructions "
                           "alone are not a speedup. Large cases do not acquire gSIM permission."),
                "performance_qualification": False,
            }
        brief["unknown_timing"] = unknown_timing(missing, roles)
    if not debuggable:
        brief["inspect_note"] = (
            "isa_tools.py debug accepts public roster test cases; run a core self-check of a "
            "related roster case to inspect its command buffer per command."
        )
    # Keep a small actionable record beside the per-case cycles. Authors often
    # filter the response down to those rows and never see the separate brief.
    advice = {}
    for entry in entries[:LIMIT]:
        advice[entry["test_case"]] = {
            "basis": "complete cycle observation; resource hints are hypotheses",
            "action": "Compare one legal alternative on this shape class and neighbors; inspect emitted effects.",
            "mechanisms": entry.get("suggested_mechanisms", [])[:4],
            "skill": "shape-schedule-selection",
            "signals": {key: entry[key] for key in (
                "cycles", "evidence", "evidence_kind", "array_utilization_pct",
                "roofline_bound_cycles", "gap_ratio", "dominant_modeled_resource",
                "capacity_use", "reading") if key in entry},
        }
    for name in sorted(missing):
        if len(advice) >= LIMIT or roles is not None and roles.get(name) not in PRIMARY:
            continue
        advice[name] = {
            "basis": "timing unavailable; no performance ranking",
            "action": ("Keep the incumbent and explore a legal alternative using traffic, reuse and completion "
                       "structure; check emitted effects and numerics. Repair timing before claiming a speedup."),
            "mechanisms": (fences.get(name) or {}).get("suggested_mechanisms", [])[:4],
            "skill": "explore-schedules",
        }
        # Storage/reuse evidence comes from facts and emitted artifacts and does
        # not need a complete timing estimate. Keep the same bounded row budget.
        use = capacity_use.for_case(name, by_name.get(name) or {}, workspace,
                                    artifact_root=artifact_root)
        signals = {"timing_status": "unavailable", "array_utilization_pct": None,
                   "basis": "structural evidence; no elapsed-cycle utilization"}
        if use is not None:
            signals["capacity_use"] = use
        if fences.get(name) is not None:
            signals["fence_overlap"] = fences[name]
        advice[name]["signals"] = signals
    _link_operations(advice, workspace)
    if advice:
        brief["per_case_advice"] = advice
    return brief


def _link_operations(advice: dict, workspace: Path) -> None:
    """Name each advised case's entry in the staged per-operation principles (``optimization_next.operation``)."""
    from .. import operation_optimizations
    from . import case_declarations

    selection = operation_optimizations.load_selection(workspace) if advice else None
    if selection is None:
        return
    for name, target in advice.items():
        found = operation_optimizations.link(case_declarations.load(name, workspace), selection)
        if found is not None:
            target["operation"] = found


#: The self-check request that measures one named case on gSIM (one capsule per request).
GSIM_COMMAND = "python agent_selfcheck.py --capsules {name} --sim gsim --wait-budget 5"


def unknown_timing(missing, roles) -> dict:
    """Say what UNKNOWN timing means and how to obtain cycles for one case."""
    from ..workload_roles import PRIMARY

    names = sorted(set(missing))
    performance = [n for n in names if roles is not None and roles.get(n) in PRIMARY]
    direct = [n for n in names if n not in performance]
    out = {
        "meaning": ("UNKNOWN timing is missing evidence, not a measured loss or a refusal of the variant. A "
                    "variant whose timing is UNKNOWN is still a candidate; obtain cycles before deciding."),
        "how_to_measure": ("Request gSIM cycles for one named case per request: `"
                           + GSIM_COMMAND.format(name="NAME") + "`, then collect it with `--attach ID`. "
                           "This admits required correctness cases under their own gSIM budget. Performance-"
                           "cohort (scalability) cases are measured by the operator's idle gSIM, which takes "
                           "the smallest case without timing evidence in each family first; to compare a "
                           "variant sooner, measure a smaller related correctness case of the same kind."),
    }
    if direct:
        out["commands"] = [GSIM_COMMAND.format(name=n) for n in direct[:LIMIT]]
    if performance:
        out["awaiting_idle_gsim"] = performance[:LIMIT]
    return out


def _reading(entry: dict) -> str | None:
    """Relate useful-work attainment to modeled reservations without occupancy claims."""
    utilization, dominant = entry.get("array_utilization_pct"), entry.get("dominant_modeled_resource")
    busy = None
    if isinstance(dominant, dict) and _positive(dominant.get("modeled_activity")):
        busy = (f"{dominant.get('name')} has "
                f"{round(100 * dominant['modeled_activity'], 1)}% modeled reservation activity")
    fence = entry.get("fence_overlap") or {}
    if utilization is None and busy is None:
        return fence.get("reading") if fence.get("fence_per_tile") else None
    parts = [f"useful compute-ceiling attainment is {utilization}% over {entry['cycles']} cycles"
             if utilization is not None
             else f"{entry['cycles']} cycles"]
    if busy is not None:
        parts.append(f"{busy}; inspect dependencies and stalls before attributing a bottleneck")
    else:
        parts.append(f"structural {entry.get('bound_by')} lower bound is {entry['roofline_bound_cycles']} cycles")
    if fence.get("fence_per_tile") and fence.get("reading"):
        parts.append(fence["reading"])
    return "; ".join(parts)


def _bound_class(limiter: str | None) -> str | None:
    return None if limiter is None else ("compute" if limiter == "compute" else "memory requests")


def _summary(calibrated: dict) -> dict:
    """The calibration levels that have a ratio, compactly, with the rule for using them."""
    levels = {level: {name: fitted for name, fitted in groups.items() if "median_measured_over_modeled" in fitted}
              for level, groups in (calibrated.get("levels") or {}).items()}
    return {"pairs": calibrated.get("pairs"), "basis": calibrated.get("basis"),
            "levels": {level: groups for level, groups in levels.items() if groups}, "use": calibrated.get("use"),
            "accuracy_card": calibrated.get("accuracy_card"), "qualification_issued": False}


def attach(document: dict, rows, **kwargs) -> dict:
    """Add ``performance_brief`` to a response in place; any failure leaves the response unchanged."""
    try:
        brief = build(rows, **kwargs)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return document
    if brief is not None:
        document["performance_brief"] = brief
        from . import library_pass

        # Adds the matching library pass and its paste-ready A/B to each ranked case's advice.
        library_pass.annotate(rows, brief, workspace=kwargs.get("workspace"), suite=library_pass.suite_of(document))
        advice = brief.get("per_case_advice", {})
        for key in ("per_capsule", "cases"):
            for row in document.get(key) or ():
                if not isinstance(row, dict):
                    continue
                name = str(row.get("capsule") or row.get("name") or "")
                if name in advice:
                    row["optimization_next"] = advice[name]
    return document
