"""Rank every passing test case by performance headroom, from the evidence the workspace already holds.

The verdict's performance priority ranked only the selected scalability cohort, and only rows whose
Spike timing model returned complete accounting: in one run 222 of 252 rows carried no timing at all and
the brief listed nothing actionable, so performance waited until correctness was finished. This block
covers every test case that passes L2, in any cohort, using the best bound evidence available:

* measured: the cycles of the test case's latest gSIM certificate, read from the simulator response row
  whose execution digest is the one ``qa/tier_state.json`` certified, with that row's own structural
  roofline (compute floor and ideal request service);
* measured, for a case without a certificate tier (the scalability cases): the operator's idle-time gSIM
  measurement (:mod:`.idle_gsim`) of the executable the verdict row's estimate describes, else its newest
  measurement flagged ``measured_on_earlier_source``; advisory, like everything here;
* modeled: the Spike timing model's estimate and structural floor attached to the verdict row, when the
  model accounted for every event;
* estimated: the calibrated fast census estimate (:mod:`.fast_estimate`) when no complete estimate exists;
* otherwise the case is counted as lacking evidence and named.

Headroom is cycles minus the roofline bound. Ranked entries carry ``fence_overlap`` (fences, output
tiles per fence and the static overlap ratio of the case's copied command stream and timing census,
:mod:`.fence_overlap`), and ``fence_bound`` lists every passing case fenced after nearly every output
tile. Everything here is advisory, carries no promotion authority and repeats only numbers the author's
own checks already return.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from . import capacity_use, fast_estimate, reference_baselines, shape_classes

SCHEMA = "merlin.development_performance_headroom.v1"
#: Ranked entries kept in the brief; the counts cover the rest.
LIMIT = 15
#: Entries in the utilization ranking.
UTILIZATION_LIMIT = 8
_MAX_RECORD_BYTES = 64 << 20


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _load(path: Path):
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_RECORD_BYTES:
            return None
        from .result_details import hydrate

        return hydrate(json.loads(path.read_text()), document_path=path)
    except (OSError, ValueError):
        return None


def _passes_l2(row) -> bool:
    tier = (row.get("tiers") or {}).get("L2")
    return (tier.get("status") if isinstance(tier, dict) else tier) == "pass"


def _measured_index(workspace: Path) -> dict:
    """``{execution_digest: (cycles, structural_roofline)}`` from passing gSIM response rows."""
    index = {}
    channel = workspace / ".qa_channel"
    if not channel.is_dir():
        return index
    for path in sorted(channel.glob("simresp_*.json")):
        document = _load(path)
        if not isinstance(document, dict) or document.get("sim") != "gsim":
            continue
        for row in document.get("per_capsule") or ():
            digest, cycles = row.get("execution_digest"), row.get("barrier_cycles")
            if (
                isinstance(digest, str)
                and _positive(cycles)
                and row.get("numerical_execution_passed") is not False
                and row.get("pass") is not False
            ):
                roofline = (row.get("performance") or {}).get("structural_roofline") or {}
                index[digest] = (cycles, roofline)
    return index


#: Evidence label of an operator idle-time gSIM measurement of a case without a certificate tier.
ADVISORY = "measured: gSIM measurement (advisory)"


def _advisory_index(workspace: Path) -> dict:
    """``{case: [measurement]}`` newest first, from the idle gSIM lane's responses (:mod:`.idle_gsim`).

    A measurement is ``{t, execution_digest, elf_sha256, cycles, roofline}`` of a row whose numerics passed
    on both engines; such a case has no certificate tier, so these are its only measured cycles.
    """
    from ..joint_gsim_measurement import REQUEST_PREFIX

    found: dict = {}
    channel = workspace / ".qa_channel"
    if not channel.is_dir():
        return found
    for path in channel.glob(f"resp_{REQUEST_PREFIX}*.json"):
        document = _load(path)
        if not isinstance(document, dict) or document.get("sim") != "gsim" or document.get("variant"):
            continue
        for row in document.get("per_capsule") or ():
            if not isinstance(row, dict):
                continue
            cycles = row.get("barrier_cycles")
            if row.get("pass") is not True or not _positive(cycles) or not row.get("barrier_tier"):
                continue
            performance = row.get("performance") if isinstance(row.get("performance"), dict) else {}
            found.setdefault(str(row.get("capsule")), []).append(
                {
                    "t": document.get("graded_at_unix_ns") or 0,
                    "execution_digest": row.get("execution_digest"),
                    "elf_sha256": performance.get("elf_sha256"),
                    "cycles": cycles,
                    "roofline": performance.get("structural_roofline") or {},
                }
            )
    for rows in found.values():
        rows.sort(key=lambda r: r["t"], reverse=True)
    return found


def advisory_current(rows, performance: dict):
    """The measurement of the executable ``performance`` describes, else the newest, and whether it is stale."""
    if not rows:
        return None, False
    current = (performance or {}).get("elf_sha256")
    match = next((r for r in rows if current and r["elf_sha256"] == current), None)
    return (match, False) if match is not None else (rows[0], bool(current))


def _bound(roofline: dict) -> tuple[float | None, str | None]:
    """The largest positive structural bound: compute floor or ideal read/write request service."""
    terms = {"compute": roofline.get("structural_compute_floor_cycles")}
    for side, service in (roofline.get("logical_request_ideal_service") or {}).items():
        if isinstance(service, dict):
            terms[f"{side} requests"] = service.get("ideal_requested_service_cycles")
    terms = {name: value for name, value in terms.items() if _positive(value)}
    if not terms:
        return None, None
    name = max(terms, key=terms.get)
    return terms[name], name


def _dominant_resource(performance: dict):
    resources = (performance.get("scoped_resource_feedback") or {}).get("resources") or ()
    ranked = [r for r in resources if isinstance(r, dict) and _positive(r.get("capacity_normalized_activity"))]
    if not ranked:
        return None
    top = max(ranked, key=lambda r: r["capacity_normalized_activity"])
    return {"name": top.get("name"), "modeled_activity": round(top["capacity_normalized_activity"], 3)}


def _entry(name, row, tier_state, measured, workspace, estimates=None, newest_digest=None, advisory=None):
    performance = row.get("performance") or {}
    certificate = (tier_state.get(name) or {}).get("L3") or {}
    observed = measured.get(certificate.get("execution_digest")) if certificate.get("status") == "pass" else None
    lane, lane_stale = advisory_current((advisory or {}).get(name), performance) if observed is None else (None, False)
    if observed is None and lane is not None:
        # Measured beats modeled, as for a certificate; an earlier executable is flagged, never hidden.
        cycles, roofline, evidence, stale = lane["cycles"], lane["roofline"], ADVISORY, lane_stale
    elif observed is not None:
        cycles, roofline = observed
        evidence = "measured: latest gSIM certificate"
        latest_l2 = (tier_state.get(name) or {}).get("L2") or {}
        # A ledger slot that passed, or joined an earlier certificate at an unchanged device boundary,
        # for the CURRENT loop-tier executable means these cycles describe the current program.
        current = latest_l2.get("execution_digest")
        slots = ((tier_state.get(name) or {}).get("<certs>") or {}).get("L3") or {}
        if current and any(
            isinstance(v, dict) and v.get("execution_digest") == current and v.get("status") in ("pass", "joined")
            for v in slots.values()
        ):
            stale = False
        else:
            # Per-command component digests say whether the code this test case exercises changed since
            # the certificate; the whole-submission digest moves with every unrelated edit.
            before, after = certificate.get("components"), latest_l2.get("components")
            if isinstance(before, dict) and isinstance(after, dict) and before and after:
                stale = before != after
            else:
                stale = bool(latest_l2.get("digest")) and latest_l2.get("digest") != certificate.get("digest")
    elif (
        performance.get("status") in ("PROVISIONAL", "QUALIFIED")
        and performance.get("complete_event_accounting") is True
        and _positive(performance.get("estimated_whole_soc_cycles"))
    ):
        cycles, roofline = performance["estimated_whole_soc_cycles"], performance.get("structural_roofline") or {}
        evidence, stale = f"modeled: Spike timing model ({performance['status'].lower()})", False
    elif estimates and estimates.get(name):
        # The newest submission self-check that timed this case (an author request or the operator's
        # idle-time measurement); older than the newest checked source when another digest followed.
        _, digest, performance = estimates[name][0]
        cycles, roofline = performance["estimated_whole_soc_cycles"], performance.get("structural_roofline") or {}
        evidence = "modeled: Spike timing model (self-check)"
        stale = newest_digest is not None and digest != newest_digest
    elif fast_estimate.cycles_of(performance) is not None:
        # No complete estimate yet: the calibrated census estimate ranks the case under its own label.
        cycles, roofline = fast_estimate.cycles_of(performance), performance.get("structural_roofline") or {}
        evidence, stale = fast_estimate.label(performance), False
    else:
        return None
    bound, limiter = _bound(roofline)
    if bound is None or cycles < bound:
        return None
    entry = {
        "test_case": name,
        "cycles": cycles,
        "roofline_bound_cycles": round(bound, 1),
        "bound_by": limiter,
        "gap_ratio": round(cycles / bound, 2),
        "headroom_cycles": round(cycles - bound, 1),
        "evidence": evidence,
    }
    useful, peak = roofline.get("useful_macs"), roofline.get("structural_peak_macs_per_cycle")
    if _positive(useful) and _positive(peak):
        entry["array_utilization_pct"] = round(100.0 * useful / (cycles * peak), 2)
    if stale:
        entry["measured_on_earlier_source"] = True
    dominant = _dominant_resource(performance)
    if dominant is not None:
        entry["dominant_modeled_resource"] = dominant
    if (workspace / "selfcheck_out" / name / "command_buffer.json").is_file():
        entry["inspect"] = f"python isa_tools.py debug selfcheck_out/{name}/command_buffer.json --capsule {name}"
    return entry


def collect(workspace, verdict) -> dict:
    """The evidence both the headroom block and the headline read, loaded once per verdict."""
    workspace = Path(workspace)
    tier_state = _load(workspace / "qa" / "tier_state.json") or {}
    from .idle_performance import estimates as channel_estimates

    channel = workspace / ".qa_channel"
    timed = channel_estimates(channel, with_status=True) if channel.is_dir() else {}
    return {
        "tier_state": tier_state if isinstance(tier_state, dict) else {},
        "measured": _measured_index(workspace),
        "estimates_with_status": timed,
        "estimates": {name: [row[:3] for row in rows] for name, rows in timed.items()},
        "advisory_measured": _advisory_index(workspace),
        "reference_baselines": reference_baselines.load(workspace),
    }


def build(workspace, verdict, *, limit=LIMIT, evidence=None, observe=None) -> dict:
    """The headroom block for ``verdict`` (see the module docstring).

    ``observe(passing rows, tier_state, measured index, estimates, advisory measurements)`` sees the
    evidence the ranking is built from, before any truncation (the per-case ledger of :mod:`.perf_history`
    records it).
    """
    workspace = Path(workspace)
    from .result_details import hydrate

    verdict = hydrate(verdict, workspace=workspace)
    evidence = collect(workspace, verdict) if evidence is None else evidence
    tier_state, measured, estimates = evidence["tier_state"], evidence["measured"], evidence["estimates"]
    advisory = evidence.get("advisory_measured") or {}
    newest = max((rows[0] for rows in estimates.values() if rows), key=lambda r: r[0], default=None)
    newest_digest = newest[1] if newest is not None else None
    ranked, missing = [], []
    passing = [row for row in verdict.get("per_capsule") or () if isinstance(row, dict) and _passes_l2(row)]
    if observe is not None:
        observe(passing, tier_state if isinstance(tier_state, dict) else {}, measured, estimates, advisory)
    from .joint_test_scope import excluded_cases

    # Cases declared ``performance_scope: excluded`` keep their per-case history but never rank.
    excluded = excluded_cases(verdict.get("joint_test_selection"))
    for row in passing:
        name = row.get("capsule", row.get("name"))
        if name in excluded:
            continue
        entry = _entry(name, row, tier_state, measured, workspace, estimates, newest_digest, advisory)
        if entry is None:
            missing.append(name)
        else:
            ranked.append(entry)
    # Raw observations remain in their ledgers; headline rankings require public primary roles.
    from ..workload_roles import PRIMARY, workspace_roles

    roles = workspace_roles(workspace) or {}
    diagnostics = [entry for entry in ranked if roles.get(entry["test_case"]) not in PRIMARY]
    ranked = [entry for entry in ranked if roles.get(entry["test_case"]) in PRIMARY]
    evidence["ranked"] = ranked
    unclassified = sorted(str(row.get('capsule', row.get('name'))) for row in passing
                          if roles.get(row.get('capsule', row.get('name'))) not in
                          PRIMARY | {'correctness', 'calibration', 'diagnostic_control'})
    ranked.sort(key=lambda e: (-e["headroom_cycles"], -e["gap_ratio"], str(e["test_case"])))
    # Operator-measured reference compilers on the same cases: cycles and same-engine ratios, advisory.
    baselines = evidence.get("reference_baselines")
    reference_baselines.annotate(evidence["ranked"], reference_baselines.index(baselines))
    from .mechanism_hints import hints, load_selection

    selection = load_selection(workspace)
    from . import model_calibration
    from .performance_brief import _bound_class, _summary

    calibrated = model_calibration.calibration(workspace)
    from . import fence_overlap

    rows = {str(row.get("capsule", row.get("name"))): row for row in passing}
    fences = {}
    for name, row in rows.items():
        if name not in excluded and roles.get(name) in PRIMARY:
            fence = fence_overlap.for_entry(name, workspace, row.get("performance"), selection=selection, roles=roles)
            if fence is not None:
                fences[name] = fence
    for entry in ranked[:limit]:
        suggested = hints(selection, dominant=entry.get("dominant_modeled_resource"), bound_by=entry.get("bound_by"))
        if suggested:
            entry["suggested_mechanisms"] = suggested
        if str(entry["test_case"]) in fences:
            entry["fence_overlap"] = fences[str(entry["test_case"])]
        entry["evidence_kind"] = shape_classes.evidence_kind(entry["evidence"])
        use = capacity_use.for_case(str(entry["test_case"]), rows.get(str(entry["test_case"])) or {}, workspace)
        if use is not None:
            entry["capacity_use"] = use
        if entry["evidence"].startswith("modeled"):
            entry.update(
                model_calibration.apply(
                    entry["cycles"],
                    calibrated,
                    bound_class=_bound_class(entry.get("bound_by")),
                    family=model_calibration.semantic_family(workspace, str(entry["test_case"])),
                )
                or {}
            )
    utilization = sorted(e["array_utilization_pct"] for e in ranked if "array_utilization_pct" in e)
    return {
        "schema": SCHEMA,
        "advisory_only": True,
        "promotion_authority": False,
        "scope": "passing primary workload roles, except performance_scope excluded",
        "classification_needed": {"count": len(unclassified), "cases": unclassified[:limit]},
        "diagnostic_controls": diagnostics[:limit],
        "diagnostic_control_count": len(diagnostics),
        "policy": (
            "Optimize representative workloads; small controls retain correctness and diagnostic feedback. "
            "Performance is part of every change, not a final phase: work the largest headroom first, "
            "inspect where its cycles go, change the schedule or lowering, and A/B the variant on the "
            "same test cases. Correctness gates are unchanged."
        ),
        "counts": {
            "passing_l2": len(passing),
            "ranked": len(ranked),
            "measured": sum(e["evidence"].startswith("measured") for e in ranked),
            "modeled": sum(e["evidence"].startswith("modeled") for e in ranked),
            "fast_estimate": sum(e["evidence"].startswith("estimated") for e in ranked),
            "without_evidence": len(missing),
            "performance_scope_excluded": len(excluded & {row.get("capsule", row.get("name")) for row in passing}),
        },
        "median_array_utilization_pct": utilization[len(utilization) // 2] if utilization else None,
        "mechanism_hints": (
            "catalogue mechanisms this target's architecture supports, matched to each "
            "entry's busiest modeled resource (mechanisms naming its storage unit first, then "
            "by facet); candidates to try and A/B by cycles, not a pass order"
            if selection
            else "unavailable: the architecture selection was not staged"
        ),
        "model_calibration": _summary(calibrated) if calibrated is not None else None,
        "largest_headroom": ranked[:limit],
        # Useful compute-ceiling attainment is diagnostic, not physical engine occupancy.
        "lowest_utilization": [
            {
                key: e[key]
                for key in ("test_case", "array_utilization_pct", "cycles", "evidence", "evidence_kind",
                            "dominant_modeled_resource")
                if key in e
            }
            for e in sorted(
                (e for e in ranked if "array_utilization_pct" in e),
                key=lambda e: (e["array_utilization_pct"], -e["headroom_cycles"], str(e["test_case"])),
            )[:UTILIZATION_LIMIT]
        ],
        "without_evidence": sorted(str(name) for name in missing)[:20],
        "fence_bound": fence_overlap.summary(fences, selection),
        **({"reference_baselines": compared} if (compared := reference_baselines.summary(ranked, baselines)) else {}),
    }


def attach(workspace, verdict: dict, *, observe=None) -> dict:
    """Add the block under ``feedback_brief.performance_priority`` and the headline first; never fails a
    verdict (see :mod:`.performance_headline`)."""
    try:
        evidence = collect(workspace, verdict)
        block = build(workspace, verdict, evidence=evidence, observe=observe)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return verdict
    brief = verdict.get("feedback_brief")
    if isinstance(brief, dict):
        priority = brief.get("performance_priority")
        if not isinstance(priority, dict):
            priority = brief["performance_priority"] = {"advisory_only": True, "promotion_authority": False}
        priority["development_headroom"] = block
    from . import performance_headline

    try:
        headline = performance_headline.build(
            workspace,
            verdict,
            ranked=evidence.get("ranked") or [],
            block=block,
            tier_state=evidence["tier_state"],
            measured=evidence["measured"],
            estimates=evidence["estimates_with_status"],
            advisory=evidence.get("advisory_measured"),
        )
    except Exception as error:  # noqa: BLE001 -- a summary must never fail the grade it summarizes
        print(f"[grade] performance headline unavailable: {type(error).__name__}", flush=True)
        return verdict
    placed = performance_headline.place(verdict, headline)
    verdict.clear()
    verdict.update(placed)
    return verdict
