"""Display/record-only classification of a graded capsule's outcome.

The score's ``per_capsule`` rows historically carried a status and a bare per-tier status string but not
the failure block that decided the row.  A capsule failed by the host-compute veto or by
``MUST_REFUSE_VIOLATED`` (both are post-tier gates: every tier may have passed) therefore read in any
summary of the rows as L0/L1 "skipped" -- the only non-pass text the row carried.  This module derives,
from fields the result already holds, five outcome kinds and the TRUE failure category and tier.

It never reads or writes a status, so it cannot change a verdict: callers attach its output beside the
existing fields.

Outcome kinds (capsule level)
  pass            status ``pass``
  fail            status ``fail``; ``failure_category`` / ``failure_plane`` / ``failure_tier`` say why
  skipped         the capsule was deliberately not measured: ``gated``, ``not_graded``, ``screened_only``
  unavailable     a measurement was owed and not obtained: infrastructure fault, budget, cert not measured,
                  or any other non-pass status this module does not recognise
Tier level adds ``not_applicable`` (the tier record's own flag) and ``skipped`` (budget-deferred or a
skipped record).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from . import tier_integrity as _TI

PASS, FAIL, SKIPPED, UNAVAILABLE, NOT_APPLICABLE = "pass", "fail", "skipped", "unavailable", "not_applicable"
OUTCOMES = (PASS, FAIL, NOT_APPLICABLE, SKIPPED, UNAVAILABLE)
UNCATEGORIZED = "UNCATEGORIZED"

_SKIPPED_STATUSES = frozenset({"gated", "not_graded", "screened_only"})
#: Failure planes decided AFTER the tier ladder ran; no tier is "the" failing one.
_POST_TIER_PLANES = frozenset({"host_compute", "expected_refusal", "epilogue_store_path", "cost", "gate"})


def tier_outcome(record: object) -> str:
    """One of :data:`OUTCOMES` for a tier record (dict or bare status string)."""
    if _TI.tier_field(record, "not_applicable"):
        return NOT_APPLICABLE
    if _TI.tier_field(record, "budget_deferred"):
        return SKIPPED
    status = _TI.tier_status(record)
    if status == "pass":
        return PASS
    if status == "fail":
        return FAIL
    if status == "skipped":
        return SKIPPED
    return UNAVAILABLE


def tier_outcomes(tiers: Mapping | None, ladder: Iterable[str] = ()) -> dict[str, str]:
    order = [str(t) for t in ladder if t in (tiers or {})]
    order += [str(t) for t in (tiers or {}) if str(t) not in order]
    return {t: tier_outcome((tiers or {})[t]) for t in order}


def capsule_outcome(status: object) -> str:
    status = str(status)
    if status == "pass":
        return PASS
    if status == "fail":
        return FAIL
    if status in _SKIPPED_STATUSES:
        return SKIPPED
    return UNAVAILABLE


def outcome_fields(result: Mapping[str, Any], ladder: Iterable[str] = ()) -> dict[str, Any]:
    """Fields to attach beside a ``per_capsule`` entry.  Always has ``outcome`` and ``tier_outcomes``;
    a non-pass row also carries ``failure_category``, ``failure_plane``, ``failure_tier`` and
    ``failure_stage`` (``tier`` | ``post_tier`` | ``pre_tier``)."""
    tiers = result.get("tiers") if isinstance(result.get("tiers"), Mapping) else {}
    per_tier = tier_outcomes(tiers, ladder)
    kind = capsule_outcome(result.get("status"))
    out: dict[str, Any] = {"outcome": kind, "tier_outcomes": per_tier}
    if kind == PASS:
        return out
    out["status_detail"] = str(result.get("status"))
    failure = result.get("failure") if isinstance(result.get("failure"), Mapping) else {}
    category = failure.get("category")
    out["failure_category"] = str(category) if category else UNCATEGORIZED
    plane = failure.get("plane")
    out["failure_plane"] = str(plane) if plane else None
    tier = failure.get("tier")
    if not tier and plane not in _POST_TIER_PLANES:
        tier = next((t for t, o in per_tier.items() if o == FAIL), None)
    out["failure_tier"] = str(tier) if tier else None
    if out["failure_tier"]:
        out["failure_stage"] = "tier"
    elif any(o == PASS for o in per_tier.values()):
        out["failure_stage"] = "post_tier"
    else:
        out["failure_stage"] = "pre_tier"
    return out


def summarize(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts over rows carrying :func:`outcome_fields`: ``{"pass": n, "fail": {category: n},
    "skipped": {status: n}, "unavailable": {status: n}, "not_applicable_tiers": {tier: n}}``."""
    out: dict[str, Any] = {PASS: 0, FAIL: {}, SKIPPED: {}, UNAVAILABLE: {}, "not_applicable_tiers": {}}
    for row in rows:
        kind = row.get("outcome") or capsule_outcome(row.get("status"))
        if kind == PASS:
            out[PASS] += 1
        elif kind == FAIL:
            key = f"{row.get('failure_category') or UNCATEGORIZED}@{row.get('failure_tier') or row.get('failure_stage') or '?'}"
            out[FAIL][key] = out[FAIL].get(key, 0) + 1
        else:
            key = str(row.get("status_detail") or row.get("status"))
            out[kind][key] = out[kind].get(key, 0) + 1
        for tier, o in (row.get("tier_outcomes") or {}).items():
            if o == NOT_APPLICABLE:
                out["not_applicable_tiers"][tier] = out["not_applicable_tiers"].get(tier, 0) + 1
    return out
