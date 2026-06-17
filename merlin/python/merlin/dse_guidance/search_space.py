"""DSE search-space template — the bridge from Merlin to a future DSE engine.

Merlin does not run DSE and does not rank designs. Its final hand-off is a *search-space template*:
for every HW/SW abstraction Merlin knows about (:data:`contract.ABSTRACTION_MAP`), whether the
recovered workload contract implies a need for it, why, and which knobs a DSE engine would sweep if
it does. Disabled abstractions are listed too (with ``enabled: false``) so the DSE engine sees the
full space and the reason each axis is on or off.

This is a template, not a ranking: it carries the same ``what_is_not_claimed`` discipline as the
rest of the package — no speedup, cycle, area, or energy number, and no design is chosen here.
"""
from __future__ import annotations

from dataclasses import dataclass

from merlin.dse_guidance.contract import ABSTRACTION_MAP, _NOT_CLAIMED
from merlin.dse_guidance.design_envelope import E_DERIVED, E_FQN, E_NA

_NOTE = ("DSE search-space template, not a ranking. `enabled` reflects whether the recovered "
         "workload contract implies the axis; knobs are what a DSE engine would sweep. No speedup, "
         "cycle, area, or energy is claimed and no design is chosen.")


@dataclass
class KnobEntry:
    axis: str
    abstraction: str
    enabled: bool
    reason: str
    knobs: list[str]
    evidence: str


def _enabled_index(pkg) -> dict:
    """axis -> AbstractionCandidate for this workload (the enabled axes)."""
    return {c.axis: c for c in pkg.get("cands", [])}


def template_for_workload(pkg) -> dict:
    enabled = _enabled_index(pkg)
    entries: list[KnobEntry] = []
    for axis, spec in ABSTRACTION_MAP.items():
        cand = enabled.get(axis)
        if cand is not None:
            why = cand.why_this_exists or {}
            reason = why.get("reason") or why.get("signal") or "implied by recovered contract"
            evidence = E_FQN if why.get("attributed_facts") else E_DERIVED
            entries.append(KnobEntry(axis=axis, abstraction=spec["system_abstraction"],
                                     enabled=True, reason=reason,
                                     knobs=list(spec["dse_knobs"]), evidence=evidence))
        else:
            entries.append(KnobEntry(axis=axis, abstraction=spec["system_abstraction"],
                                     enabled=False, reason="axis not implied by this workload",
                                     knobs=list(spec["dse_knobs"]), evidence=E_NA))
    return {"scope": "workload", "name": pkg["case"].workload, "entries": entries}


def template_for_family(family: str, axis_set: set[str]) -> dict:
    entries: list[KnobEntry] = []
    for axis, spec in ABSTRACTION_MAP.items():
        on = axis in axis_set
        entries.append(KnobEntry(
            axis=axis, abstraction=spec["system_abstraction"], enabled=on,
            reason=(f"implied by >=1 member of family '{family}'" if on
                    else f"not implied by any member of family '{family}'"),
            knobs=list(spec["dse_knobs"]), evidence=(E_DERIVED if on else E_NA)))
    return {"scope": "family", "name": family, "entries": entries}


def to_yaml_obj(template: dict) -> dict:
    return {"dse_search_space_template": {
        "scope": template["scope"],
        "name": template["name"],
        "note": _NOTE,
        "what_is_not_claimed": list(_NOT_CLAIMED),
        "search_space": {
            "abstractions": [
                {"axis": e.axis, "abstraction": e.abstraction, "enabled": e.enabled,
                 "reason": e.reason, "knobs": e.knobs, "evidence": e.evidence}
                for e in template["entries"]],
        },
    }}
