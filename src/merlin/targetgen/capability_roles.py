"""Does a capability's STANDALONE form survive an experiment's prohibited instruction roles?

A contract declares how a family runs standalone as evidence paths (``standalone_evidence``): the
derived hardware fact that licenses each path and the instruction roles a program uses to drive it
(:class:`merlin.targetgen.compute_units.EvidencePath`). A policy such as "no hardware loop
descriptors" (``prohibited_instruction_roles``) is a statement in the same closed role vocabulary, so
whether the capability survives it is a set intersection, decided per path:

* a path that needs a prohibited role is blocked;
* a path whose fact a supplied semantic-facts document does not report as derived is unproven;
* the capability is admitted standalone when at least one path is neither.

A capability declared only fused (``composed_with``) is never standalone, and one declared standalone
without checked evidence requires explicit operation review and is reported as declared_unchecked.
Nothing here names a target, a fact or a role: all three are data.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

DERIVED = "derived"
NOT_CHECKED = "not_checked"


def fact_status(semantic_facts: Mapping[str, Any] | None, fact: str) -> str:
    """The named fact's status in a ``semantic_facts`` document, or ``not_checked`` without one."""
    if semantic_facts is None:
        return NOT_CHECKED
    row = (semantic_facts.get("facts") or {}).get(fact)
    return str(row.get("status")) if isinstance(row, Mapping) else "absent"


def standalone_admission(
    capability,
    *,
    prohibited_roles: Iterable[str] = (),
    semantic_facts: Mapping[str, Any] | None = None,
    review_status: str | None = None,
) -> dict[str, Any]:
    """``{"standalone": bool, "basis": str, "paths": [...], "reason": str}`` for one capability."""
    if capability.composed_with:
        return {
            "standalone": False,
            "basis": "composed_only",
            "paths": [],
            "reason": f"{capability.family} is declared only composed with {list(capability.composed_with)}",
        }
    if not capability.standalone_evidence:
        return {
            "standalone": review_status == "reviewed",
            "basis": "declared_unchecked",
            "paths": [],
            "reason": f"{capability.family} is declared standalone with no evidence paths to check",
        }
    prohibited = set(prohibited_roles)
    paths = []
    for path in capability.standalone_evidence:
        blocked = sorted(set(path.roles) & prohibited)
        status = fact_status(semantic_facts, path.fact)
        admitted = not blocked and (status == DERIVED or (status == NOT_CHECKED and review_status == "reviewed"))
        paths.append(
            {
                "fact": path.fact,
                "roles": list(path.roles),
                "fact_status": status,
                "prohibited_roles_used": blocked,
                "admitted": admitted,
            }
        )
    admitted = [p for p in paths if p["admitted"]]
    if admitted:
        checked = [p for p in admitted if p["fact_status"] == DERIVED]
        selected = (checked or admitted)[0]
        return {
            "standalone": True,
            "basis": "evidence" if checked else "declared_unchecked",
            "paths": paths,
            "reason": f"{capability.family} {'runs' if checked else 'is declared'} standalone "
            f"through {selected['fact']} "
            f"using roles {selected['roles']}" + ("" if checked else "; semantic fact not checked"),
        }
    why = "; ".join(
        f"{p['fact']}: "
        + (f"needs prohibited role(s) {p['prohibited_roles_used']}" if p["prohibited_roles_used"] else "")
        + (
            f"{', ' if p['prohibited_roles_used'] else ''}fact is {p['fact_status']}"
            if p["fact_status"] not in (DERIVED, NOT_CHECKED)
            else ""
        )
        for p in paths
    )
    return {
        "standalone": False,
        "basis": "declared_unchecked"
        if any(p["fact_status"] == NOT_CHECKED and not p["prohibited_roles_used"] for p in paths)
        else "evidence_refused",
        "paths": paths,
        "reason": (
            f"no standalone path for {capability.family} survives: {why}; "
            "unchecked declarations require status: reviewed on the selected spec row"
        ),
    }


def family_admission(
    units,
    family: str,
    *,
    prohibited_roles: Iterable[str] = (),
    semantic_facts: Mapping[str, Any] | None = None,
    review_status: str | None = None,
) -> dict[str, Any]:
    """The first unit whose ``family`` capability is admitted standalone, else every unit's refusal."""
    refusals = []
    for unit in units:
        for cap in unit.semantic_capabilities:
            if cap.family != family:
                continue
            verdict = standalone_admission(
                cap, prohibited_roles=prohibited_roles, semantic_facts=semantic_facts, review_status=review_status
            )
            if verdict["standalone"]:
                return {**verdict, "unit": unit.name}
            refusals.append({**verdict, "unit": unit.name})
    if not refusals:
        return {"standalone": False, "basis": "undeclared", "paths": [], "reason": f"no unit declares {family}"}
    return {**refusals[0], "refusals": refusals}


def contract_admission(
    contract: Mapping[str, Any], *, prohibited_roles: Iterable[str] = (), semantic_facts=None, review_status=None
) -> dict[str, dict[str, Any]]:
    """``{family: verdict}`` for every family a contract's units declare, under ``prohibited_roles``."""
    from merlin.targetgen import compute_units as CU

    units = list(CU.compute_units(contract))
    families = sorted({cap.family for unit in units for cap in unit.semantic_capabilities})
    roles = list(prohibited_roles)
    return {
        f: family_admission(
            units, f, prohibited_roles=roles, semantic_facts=semantic_facts, review_status=review_status
        )
        for f in families
    }
