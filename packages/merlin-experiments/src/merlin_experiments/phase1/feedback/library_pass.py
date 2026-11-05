"""The library pass to try next for a slow test case, inside the per-case ``optimization_next`` it reads.

Measured on one run (16 author sessions, ~1,600 tool calls): the author read each self-check case's own
block 41 times but filtered the response-level ``performance_brief`` out of 4 of its 5 reads, so the
brief's catalogue mechanisms never reached it; it touched the compact dense pass 25 times and the
panel pipelines, readout overlap, segmented pipeline, rolling window and the schedule dialect not once.
The brief copies each ranked case's advice onto its row as ``optimization_next``; this module adds
``optimization_next["library_pass"]`` for each case with real headroom:

* ``line`` -- one line: evidence kind, cycles, gap over the bound, compute-peak attainment |
  bottleneck class | library mechanism(s) with module:symbol and whether ``submission/`` imports them |
  a paste-ready A/B (apply through an authored adapter, then self-check changed emission) | the
  ``ab_compare.py`` command | the skill and pass-library reference to read;
* the same pieces as fields (``bottleneck``, ``mechanisms``, ``ab``, ``skill``, ``reference``).

The bottleneck class comes from what the brief already computed for the case, in this order: a stream
fenced after nearly every output tile (:mod:`.fence_overlap`); operand bytes read well beyond what the
local capacity allows (:mod:`.capacity_use`); the busiest modeled resource; the limiting roofline bound.
The mechanisms are :mod:`.mechanism_hints`' for that class (untried applicable alternatives first,
source calls distinguished from imports, no private performance ranking), each marked by whether the
author's own files under ``submission/`` import its module
(``ast``, the scheduling-api skill's inventory reader) and by the workload prerequisites
``OPTIMIZATION_SUGGESTIONS.json`` has not checked. The A/B starts from the first named mechanism without
such a prerequisite (else the first named).

Labels: evidence by kind (:func:`.shape_classes.evidence_kind`; measured, calibrated and provisional
cycle estimates remain distinct, and a Spike count is not cycles); attainment is useful MACs /
(cycles x structural peak), compute-peak attainment, not physical PE occupancy; a request roofline is
derived from the simulated memory path's bytes per cycle and is labelled gSIM-domain with that rate. No
private reference or FireSim number is read here.

Real headroom: the bound is below the cycles by more than :data:`HEADROOM_SHARE` of them and by more than
the model's own spread when the brief carries one. Nothing is written for a case without a known
bottleneck or a library mechanism for it. Advisory only: nothing is ranked, gated or applied, and a
failure here leaves the response as it was.
"""

from __future__ import annotations

import importlib.util
import shlex
import sys
from pathlib import Path

from . import mechanism_hints

KEY = "library_pass"
#: Headroom (cycles minus bound) below this share of the cycles is not worth a library pass line.
HEADROOM_SHARE = 0.2
#: Mechanisms named per case.
PER_LINE = 2
#: Words a catalogue entry uses for keeping operands local instead of re-reading them.
RESTREAM_WORDS = ("resident", "retain", "retention", "reuse", "panel")
EVIDENCE = {
    "gsim_measured": "gSIM-measured",
    "calibrated_model": "calibrated timing model",
    "provisional_model": "provisional cycle-model estimate (not measured hardware cycles)",
    "spike_count": "Spike count (not cycles)",
}
_FACET_CLASS = {
    mechanism_hints._STORAGE: "local storage",
    mechanism_hints._TRANSFER: "memory transfers",
    mechanism_hints._ISSUE: "instruction issue",
    mechanism_hints._EXECUTE: "accelerator execution",
    mechanism_hints._HOST: "host CPU work",
}
SCALABILITY_SCHEMA = "merlin.phase1.public_scalability.v1"
PROGRAMS_SCHEMA = "merlin.phase1.public_program_checks.v1"
SKILL = "ab-library-pass"
SKILL_ROOTS = (Path(".agents") / "skills", Path(".claude") / "skills")
_INVENTORY = ("author_skills", "scheduling-api", "scripts", "library_inventory.py")
_LOADED: list = []


def inventory_script():
    """The author skill's stdlib-only inventory script, loaded once by path (its ``ast`` import reader)."""
    if not _LOADED:
        path = Path(__file__).resolve().parents[1].joinpath(*_INVENTORY)
        spec = importlib.util.spec_from_file_location("merlin_experiments._library_inventory", path)
        module = importlib.util.module_from_spec(spec)
        before = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = before
        _LOADED.append(module)
    return _LOADED[0]


def suite_of(document) -> str | None:
    """``core``, ``scalability`` or ``programs`` from a response's schema; None for a variant response."""
    if not isinstance(document, dict) or document.get("variant"):
        return None
    schema = document.get("schema")
    if schema == SCALABILITY_SCHEMA:
        return "scalability"
    if schema == PROGRAMS_SCHEMA:
        return "programs"
    return "core"


def has_headroom(entry: dict) -> bool:
    cycles, headroom = entry.get("cycles"), entry.get("headroom_cycles")
    if not isinstance(cycles, int | float) or not isinstance(headroom, int | float) or cycles <= 0:
        return False
    spread = (entry.get("model_confidence") or {}).get("spread")
    floor = max(HEADROOM_SHARE, spread if isinstance(spread, int | float) and spread > 0 else 0.0)
    return headroom / cycles > floor


def _pct(value) -> str:
    return f"{round(100 * value)}%" if isinstance(value, int | float) else "?"


def bottleneck(entry: dict, selection, *, deprioritize=()) -> tuple[str, list[dict]] | None:
    """``(class text, mechanisms)`` for one brief entry, or None when the bottleneck is unknown."""
    fence = entry.get("fence_overlap") if isinstance(entry.get("fence_overlap"), dict) else {}
    if fence.get("fence_per_tile"):
        found, _ = mechanism_hints.fence_hints(
            selection, why="a full-device fence after nearly every output tile serializes tiles",
            deprioritize=deprioritize,
        )
        text = f"fence-bound ({fence.get('fences')} fences, {fence.get('tiles_per_fence')} output tiles per fence)"
        return text, found
    reads = (
        ((entry.get("capacity_use") or {}).get("operand_reads") or {})
        if isinstance(entry.get("capacity_use"), dict)
        else {}
    )
    ratio, allowed = reads.get("ratio"), reads.get("capacity_bound_ratio")
    if isinstance(ratio, int | float) and isinstance(allowed, int | float) and ratio > allowed * (1 + HEADROOM_SHARE):
        text = f"operand re-streaming (reads {ratio}x the compulsory bytes; local capacity allows ~{allowed}x)"
        return text, mechanism_hints.hints(selection, bound_by="read requests", words=RESTREAM_WORDS,
                                          why=text, deprioritize=deprioritize)
    dominant = entry.get("dominant_modeled_resource")
    facets = mechanism_hints.resource_facets((dominant or {}).get("name")) if isinstance(dominant, dict) else None
    if facets is not None:
        unit = mechanism_hints.storage_unit(dominant["name"])
        text = (
            f"{_FACET_CLASS.get(facets, 'resource')} (busiest modeled resource {dominant['name']}, "
            f"{_pct(dominant.get('modeled_activity'))} modeled activity" + (f", {unit}" if unit else "") + ")"
        )
        return text, mechanism_hints.hints(selection, dominant=dominant, bound_by=entry.get("bound_by"),
                                          deprioritize=deprioritize)
    bound = entry.get("bound_by")
    if mechanism_hints.bound_facets(bound) is not None:
        return (f"{bound} roofline bound (no resource diagnostic)", mechanism_hints.hints(
            selection, bound_by=bound, deprioritize=deprioritize))
    return None


def _variant(mechanism: str) -> str:
    name = "lib-" + "".join(c if c.isalnum() or c in ".-" else "-" for c in mechanism)
    return name[:64].rstrip("-.")


def ab_command(case: str, variant: str, suite: str, *, mechanism: str, workspace: Path) -> str:
    """Apply through an authored adapter before self-checking a changed variant.

    The preparation tool refuses missing hooks, unchanged products and a
    disconnected manifest emitter; copying a submission alone is not an A/B.
    """
    route = {"scalability": "--suite scalability", "programs": "--suite programs --sim spike"}.get(suite, "--sim spike")
    script = f"{skill_root(workspace)}/{SKILL}/scripts/apply_library_pass.py"
    prepare = shlex.join([
        "python", script, "--submission", "submission", "--variant", f"variants/{variant}",
        "--adapter", "tools/library_pass_adapter.py", "--mechanism", mechanism,
        "--case", case, "--receipt", f"ab/{variant}-application.json",
    ])
    return (
        f"{prepare} && python agent_selfcheck.py --submission {shlex.quote(f'variants/{variant}')} "
        f"--capsules {shlex.quote(case)} {route} --wait-budget 5"
    )


def skill_root(workspace: Path) -> str:
    for root in SKILL_ROOTS:
        if (Path(workspace) / root / SKILL).is_dir():
            return root.as_posix()
    return SKILL_ROOTS[0].as_posix()


def bound_text(bound_by, roofline) -> str:
    """The bound's name; a request bound says it is the simulated memory path's (gSIM-domain) rate."""
    if not isinstance(bound_by, str) or "requests" not in bound_by:
        return f"the {bound_by} bound"
    side = bound_by.split()[0]
    service = ((roofline or {}).get("logical_request_ideal_service") or {}).get(side) or {}
    rate = service.get("client_peak_bytes_per_cycle") if isinstance(service, dict) else None
    shown = f", {rate:g} B/cycle" if isinstance(rate, int | float) and rate > 0 else ""
    return f"the {bound_by} bound (gSIM-domain memory path{shown})"


def block(
    entry: dict, *, selection, suite: str, imports: dict, workspace: Path, roofline=None, roles=None
) -> dict | None:
    """``optimization_next["library_pass"]`` for one brief entry (module docstring), or None."""
    from ..workload_roles import PRIMARY, workspace_roles

    roles = workspace_roles(workspace) if roles is None else roles
    case = entry.get("test_case")
    if (roles or {}).get(case) not in PRIMARY:
        return None
    if (not isinstance(case, str) or not case or suite not in ("core", "scalability", "programs")
            or not has_headroom(entry)):
        return None
    if selection and 'source_calls' not in selection:
        selection = mechanism_hints.with_source_evidence(selection, workspace)
    selection = mechanism_hints.for_case(selection, case, workspace)
    script = inventory_script()
    found = bottleneck(entry, selection)
    if found is None or not found[1]:
        return None
    text, mechanisms = found
    workload = {
        row.get("id"): sorted(
            k for k, v in (row.get("requirements") or {}).items() if k.startswith("workload.") and v != "supported"
        )
        for row in (selection or {}).get("all_mechanisms") or ()
        if isinstance(row, dict)
    }
    shown = mechanisms[:PER_LINE]
    # The A/B starts from the first shown mechanism with no unchecked workload prerequisite, so it does not
    # begin with a pass whose workload the case may not have; the order shown stays the hints' own.
    first = next((hint for hint in shown if hint['opportunity_status'] in ('candidate', 'missing_adapter')
                  and not workload.get(hint.get("id"))), shown[0])
    named, fields = [], []
    for hint in shown:
        module, _, symbol = str(hint.get("api") or "").partition(":")
        imported = bool(script.users([module], [symbol] if symbol else [], imports)) if module else False
        unchecked = workload.get(hint.get("id")) or []
        named.append(
            f"{hint.get('id')} ({hint.get('api')}; "
            + (f"applies if {', '.join(unchecked)}; " if unchecked else "")
            + f"{hint['opportunity_status']}; "
            + f"{'imported by submission/' if imported else 'not imported by submission/'})"
        )
        fields.append(
            {
                "id": hint.get("id"),
                "api": hint.get("api"),
                "facet": hint.get("facet"),
                "imported_by_submission": imported,
                "application_status": "unknown; neither imports nor source calls establish runtime execution",
                **{key: hint[key] for key in ('opportunity_status', 'source_call_sites', 'observation_basis',
                                              'prerequisites', 'next_action', 'recipe')},
                "applies_if": unchecked,
            }
        )
    variant = _variant(str(first.get("id")))
    kind = entry.get("evidence_kind")
    evidence = EVIDENCE.get(kind, entry.get("evidence") or "unlabelled evidence")
    attainment = entry.get("array_utilization_pct")
    measured = f"{evidence}: {entry['cycles']} cycles, x{entry.get('gap_ratio')} over " + bound_text(
        entry.get("bound_by"), roofline
    )
    if attainment is not None:
        measured += f", {attainment}% compute-peak attainment"
    check = ab_command(case, variant, suite, mechanism=str(first.get("id")), workspace=workspace)
    compare = f"python {skill_root(workspace)}/{SKILL}/scripts/ab_compare.py variants/{variant}"
    reference = f"pass-library reference/{first.get('facet')}.md"
    unconfirmed = []
    if (entry.get("fence_overlap") or {}).get("fence_per_tile"):
        _, unconfirmed = mechanism_hints.fence_hints(selection, why=text)
    return {
        "line": (
            f"{measured} | bottleneck: {text} | library: {', '.join(named)} | "
            f"next: {first['next_action']} | A/B after prerequisites: {check} | then: {compare} | "
            f"read: {SKILL}, {reference}"
            + (" | unresolved effect/capability contracts: "
               + ", ".join(str(row["id"]) for row in unconfirmed[:2]) if unconfirmed else "")
        ),
        "evidence_kind": kind,
        "bottleneck": text,
        "mechanisms": fields,
        "related_capability_unconfirmed": unconfirmed,
        "unconfirmed_action": "establish the named effect/capability contracts before applying those mechanisms",
        "ab": {"variant": f"variants/{variant}", "mechanism": first.get("id"), "check": check, "compare": compare,
               "adapter": "tools/library_pass_adapter.py", "application_status": "not_run",
               "requires": "authored adapter, executed registered pipeline and changed manifest-emitted product",
               "missing_adapter_action": "construct the generic prepare/source hooks described in ab-library-pass; "
                                         "do not grade an unchanged copy"},
        "skill": SKILL,
        "reference": reference,
        "basis": "advisory: the bottleneck is a diagnostic hypothesis; keep the pass when its cycle estimate "
        "falls with numerics unchanged in the same provider/ROI. Estimates (the complete timing model, else the "
        "calibrated fast estimate) are the basis for the decision: prefer the change the estimate favors and "
        "revert if a later measurement disagrees. Final required gSIM gates are unchanged.",
    }


def annotate(rows, brief, *, workspace=None, suite: str | None, context=None) -> int:
    """Add :data:`KEY` to the ``per_case_advice`` (copied onto each row as ``optimization_next``) of each
    ranked case that gets one; returns how many.

    Never raises: an unreadable selection or tree leaves the brief as it was."""
    advice = brief.get("per_case_advice") if isinstance(brief, dict) else None
    if suite is None or not isinstance(advice, dict):
        return 0
    try:
        workspace = Path(workspace) if workspace is not None else Path.cwd()
        selection = (context.selection if context is not None else mechanism_hints.with_source_evidence(
            mechanism_hints.load_selection(workspace), workspace))
        if not selection:
            return 0
        imports = selection.get('source_imports', {})
        by_name = {str(r.get("capsule") or r.get("name") or ""): r for r in rows or () if isinstance(r, dict)}
        written = 0
        for entry in brief.get("largest_headroom") or ():
            name = str(entry.get("test_case"))
            target = advice.get(name)
            if not isinstance(target, dict):
                continue
            row = by_name.get(name) or {}
            roofline = (row.get("performance") or {}).get("structural_roofline") if isinstance(row, dict) else None
            found = block(
                entry, selection=selection, suite=suite, imports=imports, workspace=workspace, roofline=roofline,
                roles=context.roles if context is not None else None
            )
            if found:
                target[KEY] = found
                written += 1
        return written
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ImportError):
        return 0


__all__ = [
    "HEADROOM_SHARE",
    "KEY",
    "ab_command",
    "annotate",
    "block",
    "bottleneck",
    "bound_text",
    "has_headroom",
    "suite_of",
]
