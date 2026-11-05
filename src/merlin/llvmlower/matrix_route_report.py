"""Summaries of a routing sidecar for the routing report: MACs by model site, and the priced routes.

Kept beside the rewrite (:mod:`int8_contractions`) rather than in the image builder so that every
consumer of a sidecar -- a whole-model build's ``routing_report.json``, a coverage table, a test --
groups the same entries the same way.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

__all__ = ["by_site", "priced_routes"]


def _routed_macs(r: dict) -> int:
    return int(r.get("b") or 1) * int(r["m"]) * int(r["n"]) * int(r["k"])


def by_site(routed: list[dict], not_routed: list[dict]) -> dict[str, dict[str, Any]]:
    """``{site: {routed, not_routed, share, not_routed_by_reason}}`` over a sidecar's two lists.

    ``site`` is the provenance-derived label the rewrite recorded (``conv`` / ``attention`` / ``linear``
    ...; ``unknown`` when a sidecar predates it). A not-routed entry's MACs are its own recorded count.
    """
    from .int8_contractions import GROUPED_REASONS

    out: dict[str, dict[str, Any]] = {}

    def slot(site: str) -> dict[str, Any]:
        return out.setdefault(site or "unknown", {"routed": 0, "not_routed": 0, "not_routed_by_reason": Counter()})

    for r in routed:
        slot(str(r.get("site") or ""))["routed"] += _routed_macs(r)
    for r in not_routed:
        s = slot(str(r.get("site") or ""))
        macs = r.get("macs")
        if macs is None:
            macs = 1
            for extent in (*r.get("parallel", ()), *r.get("k", ())):
                macs *= int(extent)
        s["not_routed"] += int(macs)
        why = str(r.get("why") or "unknown")
        s["not_routed_by_reason"][why.partition(":")[0] if why.startswith(GROUPED_REASONS) else why] += 1
    for s in out.values():
        total = s["routed"] + s["not_routed"]
        s["share"] = (s["routed"] / total) if total else None
        s["not_routed_by_reason"] = dict(s["not_routed_by_reason"].most_common())
    return dict(sorted(out.items(), key=lambda kv: -(kv[1]["routed"] + kv[1]["not_routed"])))


def priced_routes(routed: list[dict]) -> dict[str, Any]:
    """The routes that needed a decision (sub-tile, swapped or re-laid-out), grouped by that decision."""
    groups: Counter = Counter()
    macs: Counter = Counter()
    for r in routed:
        d = r.get("decision") or {}
        est = d.get("estimate") or {}
        if not d or (est.get("utilisation") == 1.0 and not r.get("steps")):
            continue
        key = f"{d.get('reason')}" + (f" [{', '.join(r['steps'])}]" if r.get("steps") else "")
        groups[key] += 1
        macs[key] += _routed_macs(r)
    return {
        "count": sum(groups.values()),
        "by_decision": [{"decision": k, "routes": n, "macs": macs[k]} for k, n in groups.most_common()],
    }
