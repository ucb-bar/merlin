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
(:mod:`..operation_optimizations`). A case whose declaration or IR shows a chain a composition route
covers gets ``composition`` (and a model or program case ``source_composition_request``) from
:mod:`.composition_next`, on its row when the brief gives it no advice. The response's ``next_steps``
(:mod:`.next_steps`) holds one ranked choice and its apply command for each of the top cases by
headroom or bundle; the self-check client prints those lines after the reply. ``family_convergence``
(:mod:`.family_rollup`) bundles every passing case by operation family and shape class (one
representative to optimize, its siblings the held-out check) and states each family's correct / total,
so a passing case's performance step never waits for the rest of its family.

Everything is advisory, bounded and repeats only numbers the response already carries or reads from
the emitted artifacts it names; it changes no status or count and never fails a check.
"""

from __future__ import annotations

import math
from pathlib import Path
from statistics import median

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


def _evidence(row: dict, *, model_first: bool = False):
    """``(cycles, roofline, label)`` from the row's own measurement, else its fast estimate, or None.

    ``model_first``: when one response mixes gSIM-measured and modeled rows, a row with a complete
    Spike model estimate is ranked on it, so the ranking compares model cycles with model cycles.
    """
    performance = row.get("performance") if isinstance(row.get("performance"), dict) else {}
    roofline = performance.get("structural_roofline") or {}
    measured = row.get("barrier_tier") == "L3" and row.get("pass") is True and _positive(row.get("barrier_cycles"))
    if measured and not (model_first and _modeled(performance)):
        return row["barrier_cycles"], roofline, "measured: gSIM cycles of this request"
    if _modeled(performance):
        return (
            performance["estimated_whole_soc_cycles"],
            roofline,
            f"modeled: Spike timing model ({performance['status'].lower()})",
        )
    from . import fast_estimate

    if fast_estimate.cycles_of(performance) is not None:
        # No complete estimate: the calibrated census estimate, under its own label, so the case ranks.
        return fast_estimate.cycles_of(performance), roofline, fast_estimate.label(performance)
    return None


def _modeled(performance: dict) -> bool:
    return (performance.get("status") in ("PROVISIONAL", "QUALIFIED")
            and performance.get("complete_event_accounting") is True
            and _positive(performance.get("estimated_whole_soc_cycles")))


def _mixed_sources(rows) -> bool:
    """Whether some rows would rank on gSIM cycles and others only on model cycles."""
    kinds = {(_evidence(row) or (0, 0, ""))[2].partition(":")[0] for row in rows}
    return {"measured", "modeled"} <= kinds


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


def _schedule_activity(performance):
    """Keep existing source-service evidence beside cycles, with the same ROI.

    Service and credit-residency unions explain overlap; generic reservations
    and static command interleaving are different observations.
    """
    scoped = performance.get("scoped_resource_feedback") or {}
    activity = scoped.get("source_actor_activity") or {}
    cycles = performance.get("estimated_whole_soc_cycles")
    if (activity.get("schema") != "merlin.scoped_source_actor_feedback.v1"
            or activity.get("status") != "modeled" or not _positive(cycles)
            or activity.get("estimated_cycles_same_window") != cycles
            or any(not performance.get(key) or activity.get(key) != performance[key]
                   or scoped.get(key) != performance[key]
                   for key in ("model_sha256", "elf_sha256", "scope_sha256"))):
        return None
    def fraction(value):
        return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
    names = {"execute.feed", "reader.slot_residency", "writer.state", "writer.slot_residency"}
    actors = {row["name"]: row["active_fraction"] for row in activity.get("actors", [])
              if row.get("name") in names and row.get("availability") == "modeled"
              and fraction(row.get("active_fraction"))}
    overlaps = [{key: row[key] for key in ("left", "right", "fraction_of_window")}
                for row in activity.get("overlaps", [])
                if row.get("left") in actors and row.get("right") in actors
                and row.get("availability") == "modeled" and fraction(row.get("fraction_of_window"))][:4]
    if not actors:
        return None
    return {"basis": "same-ROI modeled source service and finite-credit residency; not physical PE occupancy",
            "active_fraction": actors, "overlap_fraction": overlaps, "additive": False,
            "next_action": ("Inspect legal storage/completion boundaries when active phases do not overlap; "
                            "compare complete cycles after an actual shared-pass alternative. "
                            "Zero overlap alone establishes neither legality nor profitability.")}


def _role_summary(passing, entries, roles):
    """Response-local diagnostics, never a pooled utilization or a speedup claim."""
    from ..workload_roles import ROLES

    observed = {entry["test_case"]: entry for entry in entries}
    groups = {}
    for row in passing:
        name = str(row.get("capsule") or row.get("name") or "")
        role = (roles or {}).get(name, "unclassified")
        if role not in ROLES:
            role = "unclassified"
        entry = observed.get(name)
        kind = shape_classes.evidence_kind(entry["evidence"]) if entry else "unavailable"
        group = groups.setdefault((role, kind), [])
        group.append(entry)
    result = []
    for (role, kind), group in sorted(groups.items()):
        item = {"workload_role": role, "evidence_kind": kind, "passing_cases": len(group)}
        for field in ("array_utilization_pct", "gap_ratio"):
            values = [entry[field] for entry in group if entry is not None and field in entry]
            if values:
                item[f"median_{field}"] = round(median(values), 2)
                item[f"{field}_cases"] = len(values)
        result.append(item)
    return {
        "scope": "passing cases in this response; not cumulative campaign coverage",
        "basis": ("Separate workload roles and evidence kinds. Useful compute-ceiling attainment, "
                  "not physical PE occupancy. Medians describe this sample, not speedup or model accuracy. "
                  "Timing without usable roofline evidence is unavailable here."),
        "groups": result,
    }


def build(rows, *, workspace=None, artifact_root=None, debuggable: bool = True, context=None) -> dict | None:
    """The brief for one response's rows (``per_capsule`` or ``cases``); None when nothing passed."""
    workspace = Path(workspace) if workspace is not None else Path.cwd()
    from .result_details import load_row

    rows = [load_row(row, workspace=workspace) for row in rows or ()]
    passing = [row for row in rows or () if isinstance(row, dict) and _passes(row)]
    if not passing:
        return None
    selection = (context.selection if context is not None else mechanism_hints.with_source_evidence(
        mechanism_hints.load_selection(workspace), workspace))
    from ..workload_roles import PRIMARY, workspace_roles

    roles = context.roles if context is not None else (workspace_roles(workspace) or {})
    calibrated = model_calibration.calibration(workspace)
    entries, missing, fences = [], [], {}
    model_first = _mixed_sources(passing)
    for row in passing:
        name = str(row.get("capsule") or row.get("name") or "")
        case_selection = mechanism_hints.for_case(selection, name, workspace)
        fence = fence_overlap.for_entry(
            name, workspace, row.get("performance"), artifact_root=artifact_root, selection=case_selection, roles=roles
        )
        if fence is not None:
            fences[name] = fence
        found = _evidence(row, model_first=model_first)
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
        if model_first and label.startswith("modeled") and (_evidence(row) or (0, 0, ""))[2].startswith("measured"):
            entry["gsim_cycles"] = row["barrier_cycles"]  # ranked on model cycles beside model-only rows
        useful, peak = roofline.get("useful_macs"), roofline.get("structural_peak_macs_per_cycle")
        if _positive(useful) and _positive(peak):
            entry["array_utilization_pct"] = round(100.0 * useful / (cycles * peak), 2)
        dominant = _dominant_resource(row.get("performance") or {})
        if dominant is not None:
            entry["dominant_modeled_resource"] = dominant
        if label.startswith("modeled"):
            activity = _schedule_activity(row.get("performance") or {})
            if activity is not None:
                entry["schedule_activity"] = activity
        if fence is not None:
            entry["fence_overlap"] = fence
        reading = _reading(entry)
        if reading:
            entry["reading"] = reading
        suggested = (mechanism_hints.hints(case_selection, dominant=dominant, bound_by=limiter)
                     if roles.get(name) in PRIMARY else [])
        if suggested:
            entry["mechanism_basis"] = suggested[0].get("why")
            entry["suggested_mechanisms"] = [
                {key: hint.get(key) for key in ("id", "api", "facet", "workload_check", "opportunity_status",
                                              "source_call_sites", "prerequisites", "next_action")}
                for hint in suggested
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
    role_summary = _role_summary(passing, entries, roles)
    diagnostics = [entry for entry in entries if roles.get(entry['test_case']) not in PRIMARY]
    entries = [entry for entry in entries if roles.get(entry['test_case']) in PRIMARY]
    for entry in diagnostics:
        entry['workload_role'] = roles.get(entry['test_case'], 'unclassified')
        entry['optimization_status'] = ('classification_needed' if entry['workload_role'] == 'unclassified'
                                        else 'diagnostic_only')
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
        "note": ("Performance of primary-role passing test cases in this request, largest headroom first. "
                 "Controls retain diagnostics without routine tuning; missing roles require classification. "
                 "The objective "
                 "is fewer cycles for the same results. Array utilization here is useful compute-ceiling "
                 "attainment, not physical PE occupancy. Resource values describe model reservations, "
                 "not a proven bottleneck. Suggested mechanisms match resource diagnostics, mechanisms "
                 "naming its unit first (OPTIMIZATION_SUGGESTIONS.json); compare variants on the same cases "
                 "by cycles before keeping one. " + RANKED_LIST),
        "counts": {"passing": len(passing), "with_evidence": len(entries) + len(diagnostics),
                   "without_evidence": len(missing)},
        "largest_headroom": entries[:LIMIT],
        "diagnostic_controls": diagnostics[:LIMIT],
        "diagnostic_control_count": len(diagnostics),
        "workload_role_summary": role_summary,
    }
    classified = PRIMARY | {'correctness', 'calibration', 'diagnostic_control'}
    unclassified = sorted(name for name in by_name if roles.get(name) not in classified)
    if unclassified:
        brief['classification_needed'] = {
            'count': len(unclassified), 'cases': unclassified[:LIMIT],
            'action': ('Select source-bound workload roles in the public role manifest before performance exploration; '
                       'retain correctness checks.')}
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
                for name, row in by_name.items() if roles.get(name) in PRIMARY and (found := cycles_of(row))[0]]
    line = shape_classes.vs_best_line(observed, workspace)
    if line:
        brief["vs_best_per_class"] = line
    fence_bound = fence_overlap.summary(
        {name: block for name, block in fences.items() if roles.get(name) in PRIMARY}, selection)
    if fence_bound is not None:
        brief["fence_bound"] = fence_bound
    if compared is not None:
        brief["reference_baselines"] = {**compared, "line": reference_baselines.headline_line(compared)}
    if calibrated is not None:
        brief["model_calibration"] = _summary(calibrated)
    if missing:
        brief["without_evidence"] = sorted(missing)[:10]
        structural = [name for name in missing if roles.get(name) in PRIMARY]
        if structural:
            brief["structural_exploration"] = {
                "cases": sorted(set(structural))[:LIMIT],
                "basis": "candidate selection for investigation; no cycle or speedup claim",
                "action": ("Continue optimization while timing coverage is repaired. Use legal pass/schedule "
                           "alternatives and inspect their emitted products: operand traffic and reloads, "
                           "packing frequency, storage lifetimes, tile coverage, completion boundaries and "
                           "CPU work. Retain alternatives with different resource tradeoffs; check affected "
                           "Spike numerics and independent shapes/tails. See explore-schedules and "
                           "cycle-model-and-gsim. Each passing case's calibrated fast estimate (evidence kind "
                           "fast_estimate) ranks it until a complete estimate arrives: decide on it and revert "
                           "if a later measurement disagrees. Large cases do not acquire gSIM permission."),
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
                "capacity_use", "reading", "schedule_activity") if key in entry},
        }
    for name in sorted(missing):
        if len(advice) >= LIMIT or roles.get(name) not in PRIMARY:
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
    direct = [n for n in names if (roles or {}).get(n) in ("correctness", "calibration", "diagnostic_control")]
    out = {
        "meaning": ("UNKNOWN timing is missing evidence, not a measured loss or a refusal of the variant. A "
                    "variant whose timing is UNKNOWN is still a candidate; decide on its calibrated fast "
                    "estimate (performance.fast_estimate, with its error band) and revert if a complete "
                    "estimate or measurement later disagrees."),
        "how_to_measure": ("Performance is compared on the Spike cycle model by default (self-checks, "
                           "--sweep, --panel, variants). Where the model reports UNKNOWN, request gSIM "
                           "cycles for one named case per request: `"
                           + GSIM_COMMAND.format(name="NAME") + "`, then collect it with `--attach ID`. "
                           "This admits required correctness cases under their own gSIM budget. To compare "
                           "a variant, obtain a supported complete-scope observation of the primary workload. "
                           "Control measurements "
                           "serve correctness/calibration diagnostics, not routine performance tuning or a "
                           "substitute performance result for a larger workload."),
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
    from . import next_steps

    try:
        context = next_steps.Context(kwargs.get('workspace'), document, rows)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        context = None
    try:
        brief = build(rows, context=context, **kwargs)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        brief = None
    from . import composition_next

    if brief is not None:
        document["performance_brief"] = brief
        from . import library_pass

        # Adds the matching library pass and its paste-ready A/B to each ranked case's advice.
        library_pass.annotate(rows, brief, workspace=kwargs.get("workspace"), suite=library_pass.suite_of(document),
                              context=context)
    # Composition routes for chain cases and source_composition.json options for model and program cases,
    # beside the library pass; cases without ranked advice get them on their own row.
    composition_next.annotate(document, rows, brief, workspace=kwargs.get("workspace"), context=context)
    from . import family_rollup

    # Preserve compact case hydration and per-family held-out readiness.
    families = family_rollup.attach(brief, rows, workspace=kwargs.get("workspace"))
    from . import reference_map

    # Matching operator-mapped reference regions for passing performance targets only.
    reference_map.attach(brief, families, workspace=kwargs.get("workspace"))
    next_steps.attach(document, rows, brief, workspace=kwargs.get("workspace"), families=families, context=context)
    if brief is not None:
        advice = brief.get("per_case_advice", {})
        for key in ("per_capsule", "cases"):
            for row in document.get(key) or ():
                if not isinstance(row, dict):
                    continue
                name = str(row.get("capsule") or row.get("name") or "")
                if name in advice:
                    row["optimization_next"] = advice[name]
    return lean(document)


#: The brief's note once the response is lean: where the advice that left the response lives.
LEAN_NOTE = ("Primary-role passing cases, largest headroom first; advisory. Per-case advice: each case's detail "
             "file (cases[].detail.path -> optimization_next). Ranked list: " + RANKED_LIST.partition(": ")[2])


def lean(document: dict) -> dict:
    """Drop the response blocks that only repeat what the author already gets elsewhere.

    Measured on the four fresh7 attempts (15 sessions, 386 self-checks, ~70 KB median reply): no author
    command read ``per_case_advice``, ``family_convergence`` or ``next_steps.steps``, and the scripts that
    read replies filtered ``performance_brief``/``next_steps`` out to see their cases. Together they were
    ~80% of a reply. Each stays reachable: the per-case advice is copied onto the case row (its detail file's
    ``optimization_next``), the family roll-up's line is in ``next_steps.lines`` and its bundles in
    ``qa/idle_tuning.json``, and each step's choice and command are its printed line."""
    brief = document.get("performance_brief")
    if isinstance(brief, dict):
        from .family_rollup import KEY as FAMILIES

        brief.pop("per_case_advice", None)
        brief.pop(FAMILIES, None)
        brief["note"] = LEAN_NOTE
    steps = document.get("next_steps")
    if isinstance(steps, dict):
        steps.pop("steps", None)
    return document
