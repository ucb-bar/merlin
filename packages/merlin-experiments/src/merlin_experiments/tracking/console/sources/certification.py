"""Certification coverage: for every cohort, how many test cases hold each tier and why the rest do not.

Per test case, from the certificate ledger ``qa/tier_state.json``, the joint physical obligation
``qa/joint_physical_correctness.json`` (``obligations``: L3 correctness required, passed, pending, for
the newest exact reported source only) and the newest fast verdict's row
(``qa_history/verdict_fast_*.json``: pass, fail, gated, incomplete with its failure category). Cohorts
come from the :mod:`.capsules` catalog. Test-case names appear only with ``operator_private``
(inspect). Display telemetry.

A tier is *certified* when the ledger's newest record for it (the ``st[case][tier]`` mirror, which
promotion rewrites as ``pending`` when the case's execution artifact changes) passes, and
*re-proving* when that newest record does not pass but an earlier record, for other bytes, did: an
invalidation awaiting its re-check, not a lost certificate. Certified plus re-proving is every case
holding a certificate for some bytes (``L2_held``/``L3_held``, the rule :mod:`.progress` counts).

The headline is the cases passing numerics over every test case and cohort, the CASES panel's own count
(:func:`.cases.numerics`; the console refreshes CASES first and this panel repeats its totals). Tier
certification follows on one line: only cohorts whose cases carry tier records or physical obligations
are tier-certified in the loop, that line divides by their cases, and the other cohorts are listed apart
(decided by the final gate). Rows a grade carried by regression sampling count as ``carried (sampled)``.

Panel fields beyond the shared shape: ``cohorts`` [{cohort, cases, L2, L3, L2_reproving,
L3_reproving, tier_records, tier_certified_in_loop, physical_required, physical_passed,
newest: {status: n}}], ``reasons`` {failure category: n}.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import clean, obj, rows, scan_dir
from .insight_common import line, panel, private_rows, row, table
from .progress import FAIL, PASS

ORDER = ("functional", "calibration", "scalability")
#: Row status of a case regression sampling carried (``phase1.feedback.regression_sampling``).
CARRIED = "carried"
CARRIED_LABEL = "carried (sampled)"
#: Labels for cases without a catalog cohort: unknown membership, never "decided by the final gate".
NO_COHORT = ("not in catalog", "not recorded")


def tier_status(state: Any, tier: str) -> str | None:
    """``pass`` (newest record passes), ``reproving``, ``fail``, ``pending`` or None (no record).

    The newest record is the mirror, else the ledger's last-inserted record for the tier.
    """
    ledger = [r for r in obj(obj(obj(state).get("<certs>")).get(tier)).values() if obj(r)]
    mirror = obj(obj(state).get(tier))
    newest = mirror or (ledger[-1] if ledger else {})
    if not newest:
        return None
    status = newest.get("status")
    if status in PASS:
        return "pass"
    if any(r.get("status") in PASS for r in ledger):
        return "reproving"
    return "fail" if status in FAIL else "pending"


def _held(state: Any, tier: str) -> str | None:
    """A certificate held for some bytes (certified or re-proving), as :mod:`.progress` counts it."""
    status = tier_status(state, tier)
    return "pass" if status in ("pass", "reproving") else status


def _newest_verdict(ctx, run_dir: Path) -> dict[str, dict]:
    history = run_dir / "qa_history"
    names = sorted(
        e.name for e in scan_dir(history, 8192) if e.name.startswith("verdict_fast_") and e.name.endswith(".json")
    )
    if not names:
        return {}
    document = obj(ctx.reader.document(history / names[-1], limit=32 << 20))
    return {r.get("capsule"): r for r in rows(document.get("per_capsule"), 8192) if isinstance(r.get("capsule"), str)}


def _matrix(ctx) -> tuple[list[dict], dict[str, int], dict[str, dict[str, list[str]]]]:
    workspace = Path(ctx.experiment.get("workspace") or "")
    qa = workspace / "qa"
    ledger = obj(ctx.reader.document(qa / "tier_state.json", limit=16 << 20, stale_ok=True)) if workspace.name else {}
    joint = (
        ctx.reader.document(qa / "joint_physical_correctness.json", limit=16 << 20, stale_ok=True)
        if workspace.name
        else None
    )
    physical = obj(obj(joint).get("obligations"))
    newest = _newest_verdict(ctx, Path(ctx.experiment.get("run_dir") or ""))
    catalog = {r.get("name"): r.get("cohort") or NO_COHORT[1] for r in private_rows(ctx, "capsules") if r.get("name")}
    names = set(catalog) | {n for n in ledger if not str(n).startswith("<")} | set(physical) | set(newest)
    cohorts: dict[str, dict[str, Any]] = {}
    members: dict[str, dict[str, list[str]]] = {}
    reasons: dict[str, int] = {}
    for name in names:
        cohort = catalog.get(name, NO_COHORT[0])
        c = cohorts.setdefault(
            cohort,
            {
                "cohort": cohort,
                "cases": 0,
                "L2": 0,
                "L3": 0,
                "L2_reproving": 0,
                "L3_reproving": 0,
                "tier_records": 0,
                "physical_required": 0,
                "physical_passed": 0,
                "newest": {},
            },
        )
        m = members.setdefault(cohort, {})
        c["cases"] += 1
        statuses = {tier: tier_status(ledger.get(name), tier) for tier in ("L2", "L3")}
        if any(statuses.values()):
            c["tier_records"] += 1
        for tier, status in statuses.items():
            if status == "pass":
                c[tier] += 1
                m.setdefault(f"{tier} certified", []).append(name)
            elif status == "reproving":
                c[f"{tier}_reproving"] += 1
                m.setdefault(f"{tier} re-proving", []).append(name)
        obligation = obj(physical.get(name))
        if obligation:
            c["physical_required"] += 1
            if obligation.get("status") in PASS:
                c["physical_passed"] += 1
        verdict = newest.get(name)
        status = str(verdict.get("status") or "not recorded") if verdict else "not graded"
        c["newest"][status] = c["newest"].get(status, 0) + 1
        m.setdefault(status, []).append(name)
        category = obj(verdict).get("failure_category")
        if category:
            reasons[clean(category, 40)] = reasons.get(clean(category, 40), 0) + 1
    for c in cohorts.values():
        # Structural, not by name: the loop certifies tiers only where it keeps tier records or obligations.
        c["tier_certified_in_loop"] = bool(c["tier_records"] or c["physical_required"])
    ordered = sorted(
        cohorts.values(), key=lambda c: (ORDER.index(c["cohort"]) if c["cohort"] in ORDER else 9, c["cohort"])
    )
    return ordered, reasons, members


def _numerics(ctx) -> dict:
    """The CASES panel's cases-passing totals: its section of this refresh, else its own computation."""
    totals = obj(obj(obj(ctx.shared).get("cases")).get("totals"))
    if isinstance(totals.get("cases"), int) and isinstance(totals.get("passing"), int):
        return {key: totals.get(key) for key in ("cases", "passing", "by_cohort")}
    from .cases import numerics

    return numerics(ctx)


def read(ctx) -> dict[str, Any]:
    from .cases import passing_text

    cohorts, reasons, members = _matrix(ctx)
    ctx.state["members"] = members
    keys = ("cases", "L2", "L3", "L2_reproving", "L3_reproving", "physical_required", "physical_passed")
    total = {k: sum(c[k] for c in cohorts) for k in keys}
    certifiable = [c for c in cohorts if c["tier_certified_in_loop"]]
    others = [c for c in cohorts if not c["tier_certified_in_loop"] and c["cohort"] not in NO_COHORT]
    unknown = [c for c in cohorts if not c["tier_certified_in_loop"] and c["cohort"] in NO_COHORT]
    total["certifiable_cases"] = sum(c["cases"] for c in certifiable)
    for tier in ("L2", "L3"):
        total[f"{tier}_held"] = total[tier] + total[f"{tier}_reproving"]
    newest_total: dict[str, int] = {}
    for c in cohorts:
        for k, v in c["newest"].items():
            newest_total[k] = newest_total.get(k, 0) + v

    def pct(a: int, b: int) -> str | None:
        return f"{a}/{b} ({100 * a / b:.0f}%)" if b else None

    def certified(tier: str) -> str | None:
        if not certifiable:
            return "no tier record yet" if total["cases"] else None
        text = pct(total[tier], total["certifiable_cases"])
        reproving = total[f"{tier}_reproving"]
        return f"{text} · re-proving {reproving}" if reproving else text

    denominator = total["certifiable_cases"]
    numerics = _numerics(ctx)
    total["numerics"] = numerics
    tiers = [f"L2 {certified('L2')}", f"L3 {certified('L3')}"] if certifiable else [certified("L2")]
    physical = pct(total["physical_passed"], total["physical_required"])
    if physical:
        tiers.append(f"L3 physical newest source {physical}")
    summary = [
        # Every test case, every cohort -- the CASES panel's own count -- leads; certification follows.
        line("cases passing numerics", passing_text(numerics)),
        line(
            "functional tier certification",
            " · ".join(t for t in tiers if t and not t.endswith("None")) or None,
            "warn"
            if total["physical_required"] and total["physical_passed"] < total["physical_required"]
            else "ok"
            if denominator and total["L3"] == denominator
            else None,
        ),
    ]
    if certifiable and others:
        summary.append(
            line(
                "not tier-certified in the loop",
                " · ".join(f"{c['cohort']} {c['cases']}" for c in others) + " (final gate)",
            )
        )
    if certifiable and unknown:
        summary.append(line("no cohort recorded", " · ".join(f"{c['cohort']} {c['cases']}" for c in unknown)))
    summary += [
        line(
            "newest verdict",
            " · ".join(
                f"{CARRIED_LABEL if k == CARRIED else k} {v}"
                for k, v in sorted(newest_total.items(), key=lambda kv: -kv[1])
            )
            or None,
        ),
        line(
            "not passing because",
            ", ".join(f"{k} ×{v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])[:3])
            or ("none recorded" if newest_total else None),
        ),
    ]
    table_rows = []
    for c in cohorts:
        n = c["newest"]
        loop = c["tier_certified_in_loop"]
        elsewhere = None if c["cohort"] in NO_COHORT else "final gate"
        table_rows.append(
            row(
                [
                    c["cohort"],
                    c["cases"],
                    c["L2"] if loop else elsewhere,
                    c["L3"] if loop else elsewhere,
                    c["L3_reproving"] if loop else None,
                    f"{c['physical_passed']}/{c['physical_required']}" if c["physical_required"] else "",
                    n.get("pass", 0),
                    n.get("fail", 0),
                    n.get("gated", 0),
                    n.get(CARRIED, 0),
                    n.get("incomplete", 0) + n.get("not graded", 0),
                ],
                "warn" if c["cases"] and c["L3"] < c["physical_required"] else None,
                key=c["cohort"],
            )
        )
    notes = [
        "certified = the tier's newest ledger record passes, i.e. it holds for the current compiled bytes",
        "re-proving = an earlier certificate was invalidated by a changed execution artifact and is being"
        " re-checked; certified + re-proving = every case holding a certificate for some bytes",
        "headline denominator = cases in cohorts with tier records or physical obligations; the other"
        " cohorts are never tier-certified in the loop and are decided by the final gate",
        "L3 physical covers only the newest exact reported source, not the author's current edits",
        "newest = the latest fast verdict's row; incomplete counts test cases whose required tier this"
        " phase could not run (NOT_RUN_IS_NOT_PASS)",
        f"{CARRIED_LABEL} = a stable passing case regression sampling did not re-run in the newest grade; it"
        " keeps its newest completed result and is neither a pass of that grade nor a failure",
        "cases passing numerics = the CASES panel's count: every test case's newest L2 result, all cohorts",
    ]
    columns = [
        "cohort",
        "cases",
        "L2",
        "L3",
        "L3 re-proving",
        "L3 phys newest",
        "pass",
        "fail",
        "gated",
        "carried",
        "not run",
    ]
    return panel(
        "CERTIFICATION",
        summary,
        table_=table(columns, table_rows),
        notes=notes,
        cohorts=cohorts,
        reasons=reasons,
        totals=total,
    )


def inspect(ctx, key: str) -> dict[str, Any]:
    groups = obj(ctx.state.get("members")).get(key) or {}
    if not ctx.operator_private:
        return {
            "title": f"cohort {key}",
            "sections": [
                {"title": status, "lines": [f"{len(names)} test cases"]} for status, names in sorted(groups.items())
            ],
            "paths": [],
        }
    return {
        "title": f"cohort {key}",
        "sections": [
            {"title": f"{status} ({len(names)})", "lines": sorted(names)[:200]}
            for status, names in sorted(groups.items())
        ],
        "paths": [],
    }
