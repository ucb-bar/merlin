"""The short list of large points run ONCE on the RTL simulator to validate the tuned Spike ranking.

Phase 2 iterates on a calibrated Spike event-timing model (``docs/guides/spike_timing.md``), fitted
and qualified on Phase 1's gSIM labels. Those labels come from small capsules, so the model's
ranking of candidates in the large-shape regimes the capacity cohort exercises
(:mod:`merlin_experiments.phase0.capacity_families`) is an extrapolation. This stage, run after
Phase 1 and before Phase 2, measures a few points of those regimes once on the RTL simulator and
asks one question: **does the tuned Spike model order two schedules of the same shape the way the
RTL simulator does?**

The selection is derived, never chosen by hand:

* every point is a member of the capacity cohort, picked by family and regime as the template's
  ``calibration_validation`` policy declares, the cheapest by its staged floor unless the policy
  says otherwise -- so the RTL budget goes to the regime, not to size;
* the one category the cohort does not hold (DMA cost by packet width) is built from the movement
  ports' own beat width and largest request;
* each point names a PAIR of schedules whose order is the thing checked: a ranking needs two arms.

Categories whose timing depends on the memory system or the host interface, where gSIM and stock
FireSim are reported to disagree, carry ``reconcile_with_board`` as the policy declares: their RTL
verdict must be reconciled against the board before it may qualify Spike's ranking there.

This module selects and writes the plan. It launches nothing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA = "merlin.phase2.calibration_validation_plan.v1"
_CHOICES = ("cheapest", "costliest")


def _floor(entry: Mapping[str, Any]) -> float:
    floor = ((entry.get("performance") or {}).get("capacity") or {}).get("roofline", {}).get("floor_cycles")
    if isinstance(floor, bool) or not isinstance(floor, (int, float)) or floor <= 0:
        raise ValueError(f"cohort member {entry.get('name')!r} carries no staged floor")
    return float(floor)


def _shape(entry: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("op", "M", "K", "N", "ci", "Himg", "Wimg", "kh", "kw", "stride", "padding", "epilogue")
    shape = {key: entry[key] for key in keys if key in entry}
    if entry.get("op") == "resident_reuse":
        shape["chain"] = len(entry.get("matmuls") or [])
        shape["M"] = (entry.get("matmuls") or [{}])[0].get("M")
    return shape


def _pick(cohort: Sequence[Mapping[str, Any]], rule: Mapping[str, Any], *, owner: str) -> Mapping[str, Any]:
    family, regime, choice = rule.get("family"), rule.get("regime"), rule.get("choose", "cheapest")
    if choice not in _CHOICES:
        raise ValueError(f"{owner}: choose must be one of {_CHOICES}")
    variant = rule.get("variant")
    found = [
        entry
        for entry in cohort
        if (block := (entry.get("performance") or {}).get("capacity") or {}).get("family") == family
        and (regime is None or block.get("regime") == regime)
        and (variant is None or block.get("variant") == variant)
    ]
    if not found:
        raise LookupError(f"{owner}: the cohort holds no member of family {family!r} regime {regime!r}")
    ordered = sorted(found, key=lambda e: (_floor(e), str(e.get("name"))))
    return ordered[0] if choice == "cheapest" else ordered[-1]


def _packet_points(basis, rule: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Movement rows one beat wide, one largest-request wide, and one element past that request."""
    rows: list[dict[str, Any]] = []
    for name, port in sorted((basis.ports.get("clients") or {}).items()):
        if port.get("direction") != "read":
            continue
        beat, largest = int(port["beat_bytes"]), int(port["max_request_bytes"])
        for label, width in (
            ("one_beat", beat),
            ("largest_request", largest),
            ("past_largest_request", largest + basis.operand_bytes),
        ):
            n = max(1, width // basis.operand_bytes)
            rows.append(
                {
                    "label": f"{name}_{label}",
                    "shape": {
                        "op": "matmul",
                        "M": basis.tile_rows,
                        "K": basis.tile_rows * int(rule["k_tiles"]),
                        "N": n,
                    },
                    "basis": (
                        f"port {name}: rows of {n * basis.operand_bytes} bytes against a {beat}-byte beat and a "
                        f"{largest}-byte largest request"
                    ),
                }
            )
    return rows


def select(
    cohort: Sequence[Mapping[str, Any]], basis, policy: Mapping[str, Any], *, owner: str = "calibration validation"
) -> dict[str, Any]:
    """The plan: every declared category's points, each a schedule pair to order on the RTL simulator."""
    acceptance = policy.get("acceptance")
    if not isinstance(acceptance, Mapping) or not all(
        isinstance(acceptance.get(k), (int, float)) and not isinstance(acceptance.get(k), bool)
        for k in ("ordering_agreement_min", "separation_min_fraction")
    ):
        raise ValueError(f"{owner}: acceptance must declare ordering_agreement_min and separation_min_fraction")
    categories = policy.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"{owner}: categories must be a non-empty list")
    points, unavailable, seen = [], [], set()
    for category in categories:
        name = str(category.get("name") or "")
        pair = category.get("pair")
        if not name or name in seen or not isinstance(pair, list) or len(pair) != 2 or len(set(pair)) != 2:
            raise ValueError(f"{owner}: every category needs a unique name and a pair of two distinct schedules")
        seen.add(name)
        rows: list[dict[str, Any]] = []
        try:
            if category.get("derive") == "packet_widths":
                rows = _packet_points(basis, category)
            else:
                for rule in category.get("pick") or []:
                    entry = _pick(cohort, rule, owner=f"{owner} {name}")
                    block = entry["performance"]["capacity"]
                    rows.append(
                        {
                            "label": str(entry["name"]),
                            "member": str(entry["name"]),
                            "shape": _shape(entry),
                            "regime": block["regime"],
                            "floor_cycles": _floor(entry),
                            "basis": block["why"],
                        }
                    )
        except LookupError as exc:
            unavailable.append({"category": name, "reason": str(exc)})
            continue
        if not rows:
            unavailable.append({"category": name, "reason": "the facts supply no point for this category"})
            continue
        for row in rows:
            points.append(
                {
                    "category": name,
                    "checks": str(category.get("checks") or ""),
                    "schedules": list(pair),
                    "reconcile_with_board": bool(category.get("reconcile_with_board")),
                    **row,
                }
            )
    plan = {
        "schema": SCHEMA,
        "stage": policy.get("stage"),
        "engine": policy.get("engine"),
        "runs": "each point's two schedules once on the RTL simulator; the timing engine is deterministic",
        "acceptance": dict(acceptance),
        "decision": (
            "per category, the tuned Spike model qualifies for Phase 2 ranking in that regime only when it "
            "orders at least ordering_agreement_min of the pairs the RTL simulator separates by more than "
            "separation_min_fraction the same way; a failing category is ranked by RTL measurement or UNKNOWN"
        ),
        "basis_sha256": basis.digest(),
        "points": points,
        "unavailable": unavailable,
    }
    plan["plan_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    return plan


def decide(plan: Mapping[str, Any], observations: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Per category: does Spike order each RTL-separated pair the way the RTL simulator did?

    ``observations`` maps a point label to ``{"rtl": {schedule: cycles}, "spike": {schedule: cycles}}``.
    A missing observation leaves its category UNKNOWN; it is never counted as agreement.
    """
    acceptance = plan["acceptance"]
    agreement_min = float(acceptance["ordering_agreement_min"])
    separation = float(acceptance["separation_min_fraction"])
    by_category: dict[str, dict[str, Any]] = {}
    for point in plan["points"]:
        row = by_category.setdefault(
            point["category"],
            {"pairs": 0, "agreeing": 0, "missing": [], "reconcile_with_board": point["reconcile_with_board"]},
        )
        seen = observations.get(point["label"])
        a, b = point["schedules"]
        try:
            rtl = (float(seen["rtl"][a]), float(seen["rtl"][b]))
            spike = (float(seen["spike"][a]), float(seen["spike"][b]))
        except (KeyError, TypeError, ValueError):
            row["missing"].append(point["label"])
            continue
        if abs(rtl[0] - rtl[1]) <= separation * max(rtl):
            continue  # the RTL simulator does not separate them; no ordering to check
        row["pairs"] += 1
        row["agreeing"] += int((rtl[0] < rtl[1]) == (spike[0] < spike[1]) and spike[0] != spike[1])
    verdicts = {}
    for name, row in sorted(by_category.items()):
        if row["missing"] or not row["pairs"]:
            status = "unknown"
        else:
            status = "qualified" if row["agreeing"] / row["pairs"] >= agreement_min else "not_qualified"
        if status == "qualified" and row["reconcile_with_board"]:
            status = "qualified_pending_board_reconciliation"
        verdicts[name] = {**row, "status": status}
    return {"plan_sha256": plan["plan_sha256"], "categories": verdicts}
