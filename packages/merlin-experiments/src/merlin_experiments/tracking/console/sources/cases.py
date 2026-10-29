"""Per-test-case breakdown: status, cycles by evidence kind, roofline gap, utilization and gain since first pass.

One row per test case: the newest completed grade's rows (``qa/latest_completed_grade.json``), the
certificate ledger (``qa/tier_state.json``) and the capsule catalog. Per case:

* **status** -- L3 / L2 certified for the current bytes (``✓``), re-proving after an invalidation
  (``↻``, see :func:`.certification.tier_status`), else the newest grade row's result with its failure
  category;
* **cycles** -- the newest gSIM measurement (*measured*) and the newest Spike timing-model estimate
  (*modeled*), each from :mod:`.case_history`, and modeled ÷ measured for the newest pair of the same
  executable (``execution_digest``) when one exists;
* **roofline** -- from the case's current evidence, chosen as the author's ``perf_summary.py`` chooses it
  (the gSIM row of the certified or current executable, else the newest gSIM row, else the newest
  estimate): the bound and ``bound_by``, gap = cycles ÷ bound, headroom = cycles − bound, array
  utilization and the busiest modeled unit;
* **tile/fence, overlap** -- output stores per full-device fence from the newest grade row's timing
  census (``!`` when the stream is fenced after nearly every output tile) and the static overlap ratio
  (loads issued while compute is outstanding in the same fence epoch) of the grade's development
  headroom (:mod:`merlin_experiments.phase1.feedback.fence_overlap`);
* **gain** -- per evidence kind, the first passing observation ÷ the best and ÷ the latest (above ×1 is
  faster), with when each was observed. Measured and modeled cycles are never put in one ratio.

Names are operator-only: public mode shows ``case NNN`` (a stable ordinal) and the same numbers.
``inspect(key)`` returns the case's whole history. Display telemetry; it grades nothing.

Panel fields beyond the shared shape: ``sorts`` [{label, order: [row index]}] (the frontend's ``s``
cycles them; filtering is the console's ``/``), ``totals`` {cases, passing, measured, modeled,
without_evidence, by_cohort: {cohort: [passing, cases]}, utilization_median_pct,
utilization_cycle_weighted_pct, gap_median, speedup: {measured, modeled}, improved: {measured,
modeled}, compared: {measured, modeled}}, ``history`` {basis, records, pending}.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from ..io import clean, obj
from .case_history import KINDS, definitions, history_for, summarize
from .certification import tier_status
from .insight_common import line, panel, private_rows, quantile, row, series, table
from .performance import _geomean, _speedup_series, family
from .progress import FAIL

#: A case counts as improved when its latest observation is this much faster than its first.
IMPROVED = 1.02
HISTORY_LINES = 48
#: The one empty record (a stable identity for :func:`_signature`; never mutated).
_NONE: dict = {}
COLUMNS = [
    "test case",
    "cohort",
    "family",
    "status",
    "gSIM cyc",
    "model cyc",
    "mod÷meas",
    "bound",
    "by",
    "gap",
    "headroom",
    "util",
    "busiest",
    "tile/fence",
    "overlap",
    "gSIM best/now",
    "model best/now",
    "best seen",
]
SORTS = (
    ("biggest headroom", lambda c: (_evidence_rank(c), -(c["headroom"] or 0), c["name"])),
    ("worst gap", lambda c: (_evidence_rank(c), -(c["gap"] or 0), c["name"])),
    ("lowest utilization", lambda c: (c["util"] if c["util"] is not None else math.inf, c["name"])),
    ("biggest gain", lambda c: (-(_gain(c) or 0), c["name"])),
    ("cohort", lambda c: (str(c["cohort"]), str(c["family"]), c["name"])),
    ("status", lambda c: (_status_rank(c), c["name"])),
    ("name", lambda c: c["name"]),
)


def _compact(value: float | None) -> str | None:
    if value is None:
        return None
    if value < 10_000:
        return f"{value:,.0f}"
    if value < 1_000_000:
        return f"{value / 1e3:.1f}k"
    return f"{value / 1e6:.2f}M"


def _times(value: float | None) -> str | None:
    if value is None:
        return None
    return f"×{value:.2f}" if value < 100 else f"×{value:.0f}"


def _pct(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value:.2f}%" if value < 1 else f"{value:.1f}%" if value < 10 else f"{value:.0f}%"


def _clock(t: float | None, now: float) -> str | None:
    if t is None:
        return None
    same_day = time.gmtime(t)[:3] == time.gmtime(now)[:3]
    return time.strftime("%H:%M" if same_day else "%m-%d %H:%M", time.gmtime(t))


def _stamp(t: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(t)) if t else "—"


def _unit(busiest: Any) -> str | None:
    name = obj(busiest).get("name")
    if not isinstance(name, str) or not name:
        return None
    return ".".join(name.split(".")[-2:])


def _gain(case: dict) -> float | None:
    for kind in KINDS:
        summary = case[kind]
        if summary and summary["distinct"] >= 2:
            return summary["speedup_latest"]
    return None


def _reference_rank(case: dict) -> tuple:
    """Cases furthest behind a reference compiler (largest same-engine ratio) first."""
    ratios = [item["ratio"] for item in case.get("reference") or () if item.get("ratio")]
    return (-max(ratios) if ratios else math.inf, case["name"])


def _evidence_rank(case: dict) -> int:
    """Cases with a roofline comparison first, then passing cases without one, then the rest."""
    return 0 if case["gap"] is not None else 1 if case["passing"] else 2


def _status_rank(case: dict) -> int:
    order = {"bad": 0, "warn": 1, None: 2, "info": 2, "ok": 3}
    return order.get(case["level"], 2) if case["passing"] or case["level"] == "bad" else 4


def _membership(grade: dict) -> dict[str, str]:
    lanes = obj(obj(grade.get("joint_test_selection")).get("performance_cohorts"))
    out: dict[str, str] = {}
    for lane, names in lanes.items():
        for name in names if isinstance(names, list) else ():
            out.setdefault(str(name), str(lane))
    return out


def _excluded(grade: dict) -> set[str]:
    """Cases the grade's selection declares ``performance_scope: excluded``."""
    scope = obj(obj(grade.get("joint_test_selection")).get("performance_scope"))
    names = scope.get("excluded")
    return {str(name) for name in names} if isinstance(names, list) else set()


def _status(state: Any, grade_row: dict | None) -> tuple[str, str | None, bool]:
    """``(text, level, passing L2)`` of one case."""
    ps = definitions()
    l2, l3 = tier_status(state, "L2"), tier_status(state, "L3")
    graded = obj(grade_row)
    passing = l2 in ("pass", "reproving") or (bool(graded) and ps.passes(graded))
    if graded.get("status") in FAIL:
        category = clean(graded.get("failure_category") or graded.get("failure_plane") or "", 24)
        return (f"fail {category}".strip(), "bad", passing)
    if l3 == "pass":
        return ("L3 ✓", "ok", True)
    if l3 == "reproving":
        return ("L3 ↻", "warn", True)
    if l2 == "pass" or (graded and ps.passes(graded)):
        # A grade row reconciled as an older version's result is evidence, not a current pass.
        older = l2 != "pass" and obj(graded.get("graded")).get("currency") == "older_version"
        return ("L2 ✓ older" if older else "L2 ✓", "warn" if older else "info", True)
    if l2 == "reproving":
        return ("L2 ↻", "warn", True)
    if graded.get("status"):
        return (clean(graded["status"], 16), None, passing)
    return ("not graded", None, passing)


def _primary(entry: dict, state: Any) -> tuple[dict | None, bool]:
    """The case's current evidence (``perf_summary.entries``' rule) and whether it is of earlier bytes."""
    ps = definitions()
    current, certified, cert_current = ps.digests(obj(state))
    chosen = None
    for found in reversed(entry["measured"]):
        chosen = chosen or found
        if found["digest"] and found["digest"] in (certified, current):
            chosen = found
            break
    if chosen is None and entry["modeled"]:
        return entry["modeled"][-1], False
    if chosen is None:
        return None, False
    stale = bool(current) and chosen["digest"] != current and not (cert_current and chosen["digest"] == certified)
    return chosen, stale


def _baselines(ctx) -> dict:
    """The workspace's staged reference-compiler baselines (``reference_baselines``), validated, else empty."""
    from merlin_experiments.phase1.feedback import reference_baselines as RB

    workspace = ctx.experiment.get("workspace")
    if not workspace:
        return _NONE
    staged = RB.staged_file(Path(workspace) / RB.REFERENCE_ROOT)
    document = ctx.reader.document(staged, limit=4 << 20, stale_ok=True) if staged is not None else None
    try:
        return RB.validate(document) if document else _NONE
    except ValueError:
        return _NONE


def _reference(case: dict, rows: list[dict]) -> list[dict]:
    """The case's comparisons with each baseline: the newest gSIM cycles only, never a model estimate."""
    from merlin_experiments.phase1.feedback import reference_baselines as RB

    measured = case["measured"]
    if measured:
        return RB.compare(measured["latest"]["cycles"], "measured: gSIM", rows)
    modeled = case["modeled"]
    return RB.compare(modeled["latest"]["cycles"] if modeled else None, "modeled", rows)


def _records(ctx) -> tuple[dict, dict]:
    """``(qa/tier_state.json, qa/latest_completed_grade.json)``, as last read (both are prefetched)."""
    workspace = ctx.experiment.get("workspace")
    if not workspace:
        return _NONE, _NONE
    qa = Path(workspace) / "qa"
    ledger = ctx.reader.document(qa / "tier_state.json", limit=64 << 20, stale_ok=True)
    grade = ctx.reader.document(qa / "latest_completed_grade.json", limit=64 << 20, stale_ok=True)
    return ledger or _NONE, grade or _NONE


def _fences(grade: dict, grade_rows: dict) -> dict[str, dict]:
    """``{case: {per_fence, overlap, per_tile, reading}}`` from the newest grade.

    Output stores per fence come from each row's timing census; the static overlap ratio and the reading
    from the ``fence_overlap`` blocks of the grade's development headroom (:mod:`...fence_overlap`).
    """
    from merlin_experiments.phase1.feedback.fence_overlap import dynamic_profile

    out: dict[str, dict] = {}
    for name, graded in grade_rows.items():
        census = dynamic_profile(obj(obj(graded).get("performance"))) or {}
        if census.get("output_stores_per_fence") is not None:
            out[name] = {"per_fence": census["output_stores_per_fence"]}
    headroom = obj(obj(obj(grade.get("feedback_brief")).get("performance_priority")).get("development_headroom"))
    ranked = [
        (e.get("test_case"), obj(e.get("fence_overlap")), False)
        for e in headroom.get("largest_headroom") or ()
        if isinstance(e, dict)
    ]
    listed = [
        (e.get("test_case"), e, True)
        for e in obj(headroom.get("fence_bound")).get("cases") or ()
        if isinstance(e, dict)
    ]
    for name, block, bound in ranked + listed:
        if not isinstance(name, str) or not block:
            continue
        found = out.setdefault(name, {})
        if found.get("per_fence") is None and block.get("tiles_per_fence") is not None:
            found["per_fence"] = block["tiles_per_fence"]
        for key, source in (("overlap", "static_overlap_ratio"), ("reading", "reading")):
            if block.get(source) is not None:
                found[key] = block[source]
        found["per_tile"] = bool(found.get("per_tile") or bound or block.get("fence_per_tile"))
    return out


def _per_fence(fence: dict) -> str | None:
    value = fence.get("per_fence")
    return None if value is None else f"{value:.2f}" + ("!" if fence.get("per_tile") else "")


def _cases(ctx, history, ledger: dict, grade: dict) -> list[dict]:
    ps = definitions()
    grade_rows = {ps.case_name(r): r for r in ps.case_rows(grade)}
    fences = _fences(grade, grade_rows)
    lanes = _membership(grade)
    excluded = _excluded(grade)
    catalog = {r.get("name"): r for r in private_rows(ctx, "capsules") if isinstance(r.get("name"), str)}
    observed = history.cases()
    names = set(grade_rows) | {n for n in ledger if not str(n).startswith("<")} | set(observed) | set(catalog)
    out = []
    for name in sorted(names):
        state = ledger.get(name)
        entry = observed.get(name) or {"measured": [], "modeled": [], "first_pass_at": None, "pair": None}
        listed = obj(catalog.get(name))
        text, level, passing = _status(state, grade_rows.get(name))
        chosen, stale = _primary(entry, state)
        cycles = chosen["cycles"] if chosen else None
        bound = chosen["bound"] if chosen and chosen["bound"] and cycles >= chosen["bound"] else None
        busiest = (chosen or {}).get("busiest") or next(
            (o["busiest"] for o in reversed(entry["modeled"]) if o.get("busiest")), None
        )
        pair = entry["pair"]
        summaries = {kind: summarize(entry[kind]) for kind in KINDS}
        if level is None and any(
            s and s["distinct"] >= 2 and s["speedup_latest"] >= IMPROVED for s in summaries.values()
        ):
            level = "ok"
        if stale and level in (None, "info", "ok"):
            level = "warn"
        out.append(
            {
                "name": name,
                "cohort": listed.get("cohort") or lanes.get(name) or "not in catalog",
                "lane": lanes.get(name),
                "performance_scope": "excluded"
                if name in excluded or listed.get("performance_scope") == "excluded"
                else "scored",
                "family": family(listed.get("op")) if listed.get("op") else listed.get("family"),
                "semantic_family": listed.get("family"),
                "status": text,
                "level": level,
                "passing": passing,
                "l2": tier_status(state, "L2"),
                "l3": tier_status(state, "L3"),
                "grade_row": grade_rows.get(name),
                "first_pass_at": entry["first_pass_at"],
                "current": chosen,
                "stale": stale,
                "cycles": cycles,
                "bound": bound,
                "by": chosen["by"] if bound else None,
                "gap": cycles / bound if bound else None,
                "headroom": cycles - bound if bound else None,
                "util": chosen["util"] if chosen else None,
                "busiest": busiest,
                "fence": fences.get(name) or {},
                "pair": pair,
                "ratio": pair["modeled"] / pair["measured"] if pair else None,
                "entry": entry,
                **summaries,
            }
        )
    return out


def passing_text(totals: dict) -> str | None:
    """``passing/cases · cohort passing/cases ...`` of :func:`_totals` (also the CERTIFICATION headline)."""
    if not totals.get("cases"):
        return None
    by_cohort = " · ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted((totals.get("by_cohort") or {}).items()))
    return f"{totals['passing']}/{totals['cases']}" + (f" · {by_cohort}" if by_cohort else "")


class _NoHistory:
    """No performance observations: case status and cohort do not depend on them."""

    @staticmethod
    def cases() -> dict:
        return {}


def numerics(ctx) -> dict[str, Any]:
    """This panel's ``totals`` {cases, passing, by_cohort} from its own rules, for a reader that runs
    without the panel's section at hand (cases known only from performance history are not counted)."""
    ledger, grade = _records(ctx)
    totals = _totals(_cases(ctx, _NoHistory(), ledger, grade))
    return {key: totals[key] for key in ("cases", "passing", "by_cohort")}


def _gain_cell(summary: dict | None) -> str | None:
    if not summary:
        return None
    if summary["distinct"] < 2:
        return "first"
    return f"{_times(summary['speedup_best'])}/{_times(summary['speedup_latest'])}"


def _totals(cases: list[dict]) -> dict[str, Any]:
    by_cohort: dict[str, list[int]] = {}
    for case in cases:
        counts = by_cohort.setdefault(str(case["cohort"]), [0, 0])
        counts[0] += case["passing"]
        counts[1] += 1
    passing = [c for c in cases if c["passing"]]
    # Correctness counts cover every case; performance aggregates only the scored ones.
    scored = [c for c in cases if c.get("performance_scope", "scored") == "scored"]
    passing_scored = [c for c in passing if c in scored]
    utils = [c["util"] for c in passing_scored if c["util"] is not None]
    useful = [
        (c["current"]["macs"], c["cycles"] * c["current"]["peak"])
        for c in passing_scored
        if c["current"] and c["current"]["macs"] and c["current"]["peak"] and c["cycles"]
    ]
    gaps = [c["gap"] for c in passing_scored if c["gap"] is not None]
    speedup, improved, compared = {}, {}, {}
    for kind in KINDS:
        moved = [c[kind]["speedup_latest"] for c in scored if c[kind] and c[kind]["distinct"] >= 2]
        speedup[kind] = _geomean(moved)
        improved[kind] = sum(1 for v in moved if v >= IMPROVED)
        compared[kind] = len(moved)
    return {
        "cases": len(cases),
        "passing": len(passing),
        "measured": sum(1 for c in passing if c["current"] and c["current"]["kind"] == "measured"),
        "modeled": sum(1 for c in passing if c["current"] and c["current"]["kind"] == "modeled"),
        "without_evidence": sum(1 for c in passing if not c["current"]),
        "performance_excluded": len(cases) - len(scored),
        "by_cohort": by_cohort,
        "utilization_median_pct": quantile(utils, 0.5),
        "utilization_cycle_weighted_pct": 100 * sum(u for u, _ in useful) / sum(d for _, d in useful)
        if useful
        else None,
        "gap_median": quantile(gaps, 0.5),
        "gap_worst": max(gaps) if gaps else None,
        "speedup": speedup,
        "improved": improved,
        "compared": compared,
    }


def _signature(ctx, history, ledger: dict, grade: dict) -> tuple:
    # The reader returns the same parsed object while a record is unchanged (or kept as last read).
    catalog = hash(
        tuple(
            (r.get("name"), r.get("cohort"), r.get("op"), r.get("family"), r.get("performance_scope"))
            for r in private_rows(ctx, "capsules")
        )
    )
    return (
        history.basis,
        history.version,
        history.pending,
        id(ledger),
        id(grade),
        catalog,
        ctx.operator_private,
        time.gmtime(ctx.now)[:3],
    )


def read(ctx) -> dict[str, Any]:
    history = history_for(ctx)
    ledger, grade = _records(ctx)
    baselines = _baselines(ctx)
    signature = (*_signature(ctx, history, ledger, grade), id(baselines))
    if ctx.state.get("signature") == signature and ctx.state.get("section") is not None:
        return ctx.state["section"]
    cases = _cases(ctx, history, ledger, grade)
    private = ctx.operator_private
    ordinal = {c["name"]: i + 1 for i, c in enumerate(cases)}
    keys = {}
    for case in cases:
        keys[case["name"] if private else f"case {ordinal[case['name']]:03d}"] = case["name"]
    ctx.state["keys"] = keys
    ctx.state["cases"] = {c["name"]: c for c in cases}
    totals = _totals(cases)
    from merlin_experiments.phase1.feedback import reference_baselines as RB

    by_case = RB.index(baselines)
    labels = sorted({r["source"] for rows in by_case.values() for r in rows})
    for case in cases:
        case["reference"] = _reference(case, by_case[case["name"]]) if case["name"] in by_case else []
    columns, at = list(COLUMNS), COLUMNS.index("mod÷meas") + 1
    columns[at:at] = [header for name in labels for header in (f"{name} cyc", f"÷{name}")]
    sorting = [*SORTS, *([("slowest vs reference", _reference_rank)] if labels else [])]

    ranked = sorted(cases, key=SORTS[0][1])
    table_rows = []
    for case in ranked:
        label = case["name"] if private else f"case {ordinal[case['name']]:03d}"
        measured, modeled = case["measured"], case["modeled"]
        best = [s["best"]["t"] for s in (measured, modeled) if s and s["distinct"] >= 2]
        compared = {item["source"]: item for item in case["reference"]}
        reference_cells = []
        for name in labels:
            item = compared.get(name)
            reference_cells += [
                (_compact(item["baseline_cycles"]) + ("" if item["engine"] == "gsim" else " FS")) if item else None,
                _times(item.get("ratio")) if item else None,
            ]
        table_rows.append(
            row(
                [
                    label,
                    case["cohort"],
                    case["family"],
                    case["status"],
                    _compact(measured["latest"]["cycles"]) if measured else None,
                    _compact(modeled["latest"]["cycles"]) if modeled else None,
                    _times(case["ratio"]),
                    *reference_cells,
                    _compact(case["bound"]),
                    case["by"].split()[0] if case["by"] else None,
                    _times(case["gap"]) + ("*" if case["stale"] else "") if case["gap"] else None,
                    _compact(case["headroom"]),
                    _pct(case["util"]),
                    _unit(case["busiest"]),
                    _per_fence(case["fence"]),
                    _pct(100 * case["fence"]["overlap"]) if case["fence"].get("overlap") is not None else None,
                    _gain_cell(measured),
                    _gain_cell(modeled),
                    _clock(max(best), ctx.now) if best else None,
                ],
                case["level"],
                key=label,
            )
        )
    sorts = [
        {"label": label, "order": [i for i, _ in sorted(enumerate(ranked), key=lambda pair: key(pair[1]))]}
        for label, key in sorting
    ]

    speedup = totals["speedup"]
    summary = [
        line("cases passing", passing_text(totals)),
        line(
            "evidence of passing cases",
            f"gSIM {totals['measured']} · model {totals['modeled']} · none {totals['without_evidence']}"
            if totals["passing"]
            else None,
            "warn" if totals["passing"] and totals["without_evidence"] > totals["passing"] // 2 else None,
        ),
        line(
            "array utilization",
            f"median {_pct(totals['utilization_median_pct'])} · cycle-weighted "
            f"{_pct(totals['utilization_cycle_weighted_pct'])}"
            if totals["utilization_median_pct"] is not None
            else None,
            "warn" if (totals["utilization_median_pct"] or 100) < 10 else None,
        ),
        line(
            "gap to roofline",
            f"median {_times(totals['gap_median'])} · worst {_times(totals['gap_worst'])}"
            if totals["gap_median"] is not None
            else None,
        ),
        line(
            "since first pass",
            " · ".join(
                f"{name} geomean {_times(speedup[kind])}, {totals['improved'][kind]}/{totals['compared'][kind]}"
                " improved"
                for kind, name in (("measured", "gSIM"), ("modeled", "model"))
                if totals["compared"][kind]
            )
            or ("no case re-observed yet" if totals["passing"] else None),
            "ok" if any((v or 0) >= IMPROVED for v in speedup.values()) else None,
        ),
        line(
            "history",
            (
                f"{history.records} records, {history.pending} still to read"
                if history.pending
                else f"{history.records} records"
            )
            + (" · qa/perf_history.jsonl" if history.basis == "ledger" else " · derived from responses and verdicts"),
            "info" if history.pending else None,
        ),
    ]
    if labels:
        shown = [
            {
                "test_case": case["name"] if private else f"case {ordinal[case['name']]:03d}",
                "reference_baselines": case["reference"],
            }
            for case in cases
            if case["reference"] and case.get("performance_scope", "scored") == "scored"
        ]
        text = RB.headline_line(RB.summary(shown, baselines))
        summary.insert(4, line("vs reference compilers", text.partition(": ")[2] if text else None))
    tracks = {kind: {} for kind in KINDS}
    for case in cases:
        for kind in KINDS:
            values = case["entry"][kind]
            if values:
                tracks[kind][case["name"]] = [(o["t"], o["cycles"]) for o in values]
    notes = [
        "gSIM = measured barrier cycles of passing L3 rows; model = Spike timing-model estimate with complete"
        " accounting; the two are never combined in one speedup",
        "bound = max(compute floor, ideal read/write request service); gap = cycles ÷ bound; util = useful MACs"
        " ÷ (cycles × structural peak MACs/cycle); * = measured on an earlier executable",
        "mod÷meas = newest modeled ÷ measured cycles of the same executable; busiest = modeled reservation"
        " share, not a hardware counter",
        "best/now = first passing observation ÷ best and ÷ latest (above ×1 is faster); s cycles the sort,"
        " / filters, Enter shows the case's history",
    ]
    if labels:
        notes.append(
            "<label> cyc = operator-measured cycles of that reference compiler on the case (FS = FireSim);"
            " ÷<label> = newest gSIM cycles ÷ that baseline, same engine only (above ×1 is slower); advisory"
        )
    if history.pending:
        notes.append("history is still loading: first-pass values may move as older records are read")
    section = panel(
        "CASES",
        summary,
        table_=table(columns, table_rows),
        series_=[
            series("gSIM speedup vs first", _speedup_series(tracks["measured"])),
            series("model speedup vs first", _speedup_series(tracks["modeled"])),
        ],
        notes=notes,
        sorts=sorts,
        totals=totals,
        history={"basis": history.basis, "records": history.records, "pending": history.pending},
    )
    ctx.state["signature"] = signature
    ctx.state["section"] = section
    return section


def _observation_line(found: dict, marks: list[str], private: bool) -> str:
    gap = found["cycles"] / found["bound"] if found["bound"] and found["cycles"] >= found["bound"] else None
    parts = [
        _stamp(found["t"]),
        f"{found['cycles']:,.0f} cyc",
        f"gap {_times(gap)}" if gap else "gap —",
        f"util {_pct(found['util']) or '—'}",
        clean(found["origin"], 40),
    ]
    if private and found["digest"]:
        parts.append(f"exe {found['digest'][:12]}")
    return "  ".join(parts) + (f"  [{', '.join(marks)}]" if marks else "")


def _history_lines(values: list[dict], summary: dict | None, private: bool) -> list[str]:
    """Each change of cycles or executable once (repeats folded), newest last; first, best, latest marked."""
    if not values or not summary:
        return ["not recorded"]
    folded: list[list[dict]] = []  # runs of observations with the same cycles and executable
    for found in values:
        if folded and (folded[-1][0]["cycles"], folded[-1][0]["digest"]) == (found["cycles"], found["digest"]):
            folded[-1].append(found)
            continue
        folded.append([found])
    lines = []
    for members in folded:
        marks = [name for name in ("first", "best", "latest") if any(summary[name] is m for m in members)]
        if len(members) > 1:
            marks.append(f"seen {len(members)}×, last {_stamp(members[-1]['t'])}")
        lines.append(_observation_line(members[0], marks, private))
    if len(lines) > HISTORY_LINES:
        lines = (
            lines[:1] + [f"… {len(lines) - HISTORY_LINES + 1} earlier changes not shown"] + lines[-HISTORY_LINES + 2 :]
        )
    return lines


def inspect(ctx, key: str) -> dict[str, Any]:
    name = obj(ctx.state.get("keys")).get(key)
    case = obj(ctx.state.get("cases")).get(name) if name else None
    if not case:
        return {"title": f"test case {clean(key, 80)}", "sections": [], "paths": [], "notice": "no such test case"}
    private = ctx.operator_private
    title = f"test case {name}" if private else f"test case {clean(key, 20)}"
    graded = obj(case["grade_row"])
    origin = obj(graded.get("graded"))
    status_lines = [
        f"cohort {case['cohort']}  performance lane {case['lane'] or '—'}  family {case['family'] or '—'}"
        f"  semantic family {case['semantic_family'] or '—'}",
        f"status {case['status']}  (L2 {case['l2'] or '—'}, L3 {case['l3'] or '—'} in qa/tier_state.json)",
        f"newest grade row: {graded.get('status') or '—'}"
        + (
            f", {graded.get('failure_plane') or ''}/{graded.get('failure_category') or ''}"
            if graded.get("failure_category") or graded.get("failure_plane")
            else ""
        )
        + (f", grade {origin.get('token')} at {origin.get('graded_at')}" if origin else "")
        + (f" (result for older version {clean(origin.get('graded_source_sha256'), 12)})"
           if origin.get("currency") == "older_version" else ""),
        f"first passed {_stamp(case['first_pass_at'])}",
    ]
    current = case["current"]
    if current:
        now_lines = [
            f"evidence {current['kind']} ({clean(current['origin'], 40)}) at {_stamp(current['t'])}"
            + ("  — measured on an earlier executable than the current L2" if case["stale"] else ""),
            f"cycles {current['cycles']:,.0f}",
            f"roofline bound {current['bound']:,.0f} cycles ({current['by']})"
            if current["bound"]
            else "roofline bound —",
            f"gap {_times(case['gap']) or '—'}  headroom {case['headroom']:,.0f} cycles"
            if case["headroom"] is not None
            else "gap —",
            f"array utilization {_pct(case['util']) or '—'}  useful MACs {current['macs'] or '—'}"
            f"  peak {current['peak'] or '—'} MACs/cycle",
            "busiest modeled unit "
            + (
                f"{obj(case['busiest']).get('name')} (activity {obj(case['busiest']).get('modeled_activity')})"
                if case["busiest"]
                else "—"
            ),
            f"requested ÷ logical bytes {current['traffic']}" if current.get("traffic") else "traffic —",
        ]
    else:
        now_lines = ["no measured or modeled cycles recorded for this case"]
    if case["fence"].get("reading"):
        now_lines.append(f"fences: {case['fence']['reading']}")
    gains = []
    for kind, name_ in (("measured", "gSIM"), ("modeled", "timing model")):
        s = case[kind]
        if not s:
            gains.append(f"{name_}: not recorded")
            continue
        gains.append(
            f"{name_}: first {s['first']['cycles']:,.0f} ({_stamp(s['first']['t'])}) → best {s['best']['cycles']:,.0f}"
            f" ({_stamp(s['best']['t'])}, {_times(s['speedup_best'])}) → latest {s['latest']['cycles']:,.0f}"
            f" ({_stamp(s['latest']['t'])}, {_times(s['speedup_latest'])}); {s['count']} observations"
        )
    pair = case["pair"]
    calibration = (
        [
            f"modeled {pair['modeled']:,.0f} ÷ measured {pair['measured']:,.0f} = {_times(case['ratio'])}"
            + (f"  (executable {pair['digest'][:12]})" if private else ""),
            "above ×1 the timing model is pessimistic for this case, below ×1 optimistic",
        ]
        if pair
        else ["no modeled and measured cycles of the same executable"]
    )
    history = ctx.state.get("history")
    transitions = []
    if history is not None and hasattr(history, "statuses"):
        previous = None
        for t, passed, status, category in history.statuses().get(name, []):
            now = (passed, status, category)
            if now != previous:
                transitions.append(
                    f"{_stamp(t)}  {'L2 pass' if passed else clean(status or 'not passing', 16)}"
                    + (f"  {clean(category, 40)}" if category and not passed else "")
                )
            previous = now
    sections = [
        {"title": "Status", "lines": status_lines},
        {"title": "Now", "lines": now_lines},
        {"title": "Gain since first pass", "lines": gains},
        {"title": "gSIM measured", "lines": _history_lines(case["entry"]["measured"], case["measured"], private)},
        {
            "title": "Timing model estimates",
            "lines": _history_lines(case["entry"]["modeled"], case["modeled"], private),
        },
        {"title": "Modeled ÷ measured", "lines": calibration},
        *(
            [
                {
                    "title": "Reference compilers (operator-measured, advisory)",
                    "lines": [
                        f"{item['source']}: {item['baseline_cycles']:,.0f} cycles on {item['engine']}"
                        + (
                            f"; newest gSIM ÷ baseline {_times(item['ratio'])}"
                            if item.get("ratio")
                            else f"; not compared ({item.get('not_compared') or 'no cycles'})"
                        )
                        for item in case["reference"]
                    ],
                }
            ]
            if case.get("reference")
            else []
        ),
        {"title": "Grades (changes only)", "lines": transitions[-HISTORY_LINES:] or ["no archived verdict names it"]},
    ]
    paths = []
    if private:
        workspace = ctx.experiment.get("workspace")
        for kind in KINDS:
            s = case[kind]
            if s:
                paths.append(s["latest"]["path"])
        if workspace:
            paths += [
                str(Path(workspace) / "qa" / "tier_state.json"),
                str(Path(workspace) / "qa" / "latest_completed_grade.json"),
            ]
    return {"title": title, "sections": sections, "paths": sorted(set(paths))}


__all__ = ["COLUMNS", "SORTS", "inspect", "read"]
