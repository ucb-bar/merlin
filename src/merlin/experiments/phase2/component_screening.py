"""Replay answer-free experiment ordering against the same component costs.

The existing core selector owns ranking. This projection exposes its observations
and order through the ordinary broker; it grants no cost or runtime authority.
Importance is equal per generated family, then equal per member of that family.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping

from merlin.common.jsonio import canonical_json
from merlin.perf.component_screen import ComponentOpportunity, rank_component_opportunities
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval

from .contracts import StageGateError, document_sha256

BASIS = "equal generated families and equal members within family"
_FIELDS = {"schema", "order", "evidence", "work_shares_sha256", "promotion"}
_ROW_FIELDS = {
    "id", "family", "regime", "status", "conservative_saved_cycles", "priority",
    "evaluation_seconds", "legal", "correct",
}


def project_screening(screening, evidence):
    """Bind ordering, gates, latency and complete costs to an exact public roster."""
    if not isinstance(screening, Mapping) or set(screening) != _FIELDS or not evidence:
        raise StageGateError("component screening lacks its closed original report")
    members = {document_sha256([row["family"], row["capsule"]]): row for row in evidence}
    if len(members) != len(evidence):
        raise StageGateError("component screening has duplicate source members")
    counts = Counter(row["family"] for row in evidence)
    shares = {identity: 1 / len(counts) / counts[row["family"]] for identity, row in members.items()}
    rows = screening["evidence"]
    if not isinstance(rows, list) or len(rows) != len(members):
        raise StageGateError("component screening omits generated members")
    proposals, seen = [], set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _ROW_FIELDS:
            raise StageGateError("component screening member schema differs")
        identity = row["id"]
        if not isinstance(identity, str) or identity not in members or identity in seen:
            raise StageGateError("component screening uses another source member")
        seen.add(identity)
        member = members[identity]
        costs = member.get("complete_cost")
        if (
            not isinstance(costs, Mapping) or row["family"] != member["family"]
            or row["regime"] != costs.get("objective")
        ):
            raise StageGateError("component screening lacks the same complete-cost regime")
        seconds = row["evaluation_seconds"]
        if seconds is not None and (
            isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or seconds <= 0
        ):
            raise StageGateError("component screening evaluation time is invalid")
        intervals = []
        for arm in ("baseline", "candidate"):
            raw = member[arm]
            intervals.append(CycleInterval(raw["lo"], raw["hi"], tuple(raw["provenance"]), tuple(raw["missing"])))
        proposals.append(ComponentOpportunity(
            identity, row["family"], row["regime"], *intervals, seconds, row["legal"], row["correct"],
        ))
    try:
        replayed = rank_component_opportunities(proposals, work_shares=shares)
        identical = canonical_json(dict(screening)) == canonical_json(replayed)
    except (TypeError, ValueError) as error:
        raise StageGateError("component screening gate or independent share is invalid") from error
    if not identical:
        raise StageGateError("component screening order or savings differs from complete-cost replay")
    return {
        **replayed, "work_share_basis": BASIS,
        "evidence": [{**row, "capsule": members[row["id"]]["capsule"]} for row in replayed["evidence"]],
    }


def validate_screening_projection(screening, evidence):
    """Check saved broker feedback without deriving authority from JSON alone."""
    if (
        not isinstance(screening, Mapping) or set(screening) != _FIELDS | {"work_share_basis"}
        or screening["work_share_basis"] != BASIS
    ):
        raise StageGateError("component screening projection has another work-share basis")
    rows = screening["evidence"]
    if not isinstance(rows, list) or any(
        not isinstance(row, Mapping) or set(row) != _ROW_FIELDS | {"capsule"} for row in rows
    ):
        raise StageGateError("component screening projection lacks its source labels")
    original = {key: screening[key] for key in _FIELDS}
    original["evidence"] = [{key: row[key] for key in _ROW_FIELDS} for row in rows]
    projected = project_screening(original, evidence)
    if canonical_json(dict(screening)) != canonical_json(projected):
        raise StageGateError("component screening projection has another source label")
    return projected
