"""Work the members furthest below their roofline first; stop at the declared attainment target.

``attainment = floor / measured`` (1.0 is the floor itself). The floor is each member's derived
roofline: the staged floor a capacity member froze at generation
(:mod:`merlin.perf.staged_roofline`), or the declared-work roofline of a feedback cell
(:mod:`merlin.perf.capsule_roofline`). Both are lower bounds on cycles, so attainment is at most 1.

* **Priority** is ascending attainment over every repeat's worst case: the member furthest below its
  roofline is worked first. A member with no derived floor cannot be placed and is listed apart,
  never sorted in as if it were close.
* **Stopping** is declared, not tuned: the cohort's ``roofline_target.attainment_min``. Phase 2 stops
  on this evidence only when every member has a derived floor and reaches the target on its worst
  repeat. A measurement below its floor refutes the floor (the facts it came from do not describe the
  machine) and blocks the stop, because "converged" against a wrong bound means nothing.

This is guidance over measured rows; it promotes nothing and qualifies no timing model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA = "merlin.phase2.roofline_priority.v1"
CONVERGED, CONTINUE, UNDECIDED, REFUTED = "converged", "continue", "undecided", "floor_refuted"


def _positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def floor_of(row: Mapping[str, Any]) -> float | None:
    """The derived floor a measured row carries, from either roofline vocabulary, or None."""
    capacity = ((row.get("performance") or {}).get("capacity") or {}).get("roofline") or row.get("staged_roofline")
    if isinstance(capacity, Mapping) and capacity.get("status") in ("derived", "partial"):
        floor = capacity.get("floor_cycles")
        return float(floor) if _positive(floor) else None
    roofline = row.get("roofline") or {}
    if (
        isinstance(roofline, Mapping)
        and roofline.get("status") == "derived"
        and _positive(roofline.get("roofline_cycles"))
    ):
        return float(roofline["roofline_cycles"])
    return None


def declared_target(descriptors: Sequence[Mapping[str, Any]]) -> float:
    """The one ``attainment_min`` the cohort's frozen members declare, or a refusal."""
    targets = {
        float(((d.get("performance") or {}).get("capacity") or {}).get("roofline_target", {}).get("attainment_min"))
        for d in descriptors
        if _positive(
            ((d.get("performance") or {}).get("capacity") or {}).get("roofline_target", {}).get("attainment_min")
        )
    }
    if len(targets) != 1:
        raise ValueError(f"the cohort must declare exactly one roofline attainment target, found {sorted(targets)}")
    target = targets.pop()
    if not 0 < target <= 1:
        raise ValueError("a roofline attainment target must be in (0, 1]")
    return target


def roofline_priorities(members: Mapping[Any, Sequence[Mapping[str, Any]]], *, attainment_min: float) -> dict[str, Any]:
    """Rank ``{identity: [repeat rows]}`` by attainment and decide the declared stop.

    Each repeat row carries its measured ``cycles`` and its floor (:func:`floor_of`). Repeats must
    agree on the floor: the floor is a property of the member, not of a run.
    """
    if not _positive(attainment_min) or attainment_min > 1:
        raise ValueError("attainment_min must be in (0, 1]")
    if not members:
        raise ValueError("no measured members were supplied")
    ranked, unknown, refuted = [], [], []
    for identity, repeats in sorted(members.items(), key=lambda item: str(item[0])):
        cycles = [row.get("cycles") for row in repeats]
        if not repeats or not all(_positive(c) for c in cycles):
            raise ValueError(f"{identity}: every repeat needs a positive measured cycle count")
        floors = {floor_of(row) for row in repeats}
        if len(floors) != 1 or None in floors:
            unknown.append({"member": identity, "reason": "no single derived floor across its repeats"})
            continue
        floor = floors.pop()
        if min(cycles) < floor:
            refuted.append({"member": identity, "floor_cycles": floor, "min_cycles": min(cycles)})
            continue
        worst, best = floor / max(cycles), floor / min(cycles)
        ranked.append(
            {
                "member": identity,
                "floor_cycles": floor,
                "attainment": {"worst": worst, "best": best},
                "within_target": worst >= attainment_min,
                "gap_cycles": max(cycles) - floor,
            }
        )
    ranked.sort(key=lambda row: (row["attainment"]["worst"], str(row["member"])))
    below = [row for row in ranked if not row["within_target"]]
    if refuted:
        status = REFUTED
    elif below:
        status = CONTINUE
    elif unknown:
        status = UNDECIDED
    else:
        status = CONVERGED
    return {
        "schema": SCHEMA,
        "attainment_min": attainment_min,
        "status": status,
        "next": [row["member"] for row in below],
        "ranked": ranked,
        "unknown_floor": unknown,
        "floor_refuted": refuted,
        "scope": "guidance over measured rows; stops only on every member's derived floor",
    }
