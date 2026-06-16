"""CSV + Markdown emitters for the DSE guidance artifacts.

Everything here is deterministic text (headless-safe). Plots are optional and live in
:mod:`merlin.dse_guidance.plots`. List-valued fields are joined with ``;`` in CSV. Every
emitted row carries its evidence tag so no ranking is presented as intuition.
"""
from __future__ import annotations

import csv
import io

from merlin.dse_guidance.baseline_cost import BaselineCost
from merlin.dse_guidance.representation import Representation
from merlin.dse_guidance.triage import TRIAGE_COLUMNS

_DIFF_COLUMNS = [
    "workload", "representation", "K", "H", "control_rate_hz", "replan_deadline_ms",
    "deadline_visible", "visible_weight_reuse", "visible_prefix_kv_reuse",
    "dispatches_per_replan", "work_per_dispatch", "recommended_axes",
    "deprioritized_axes", "evidence_type",
]


def _join(value) -> str:
    if isinstance(value, (list, tuple)):
        return ";".join(str(v) for v in value)
    return "" if value is None else str(value)


# --------------------------------------------------------------------------- triage

def triage_csv(triage_result: dict) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=TRIAGE_COLUMNS, extrasaction="ignore")
    w.writeheader()
    for row in triage_result["axes"]:
        w.writerow({k: _join(row.get(k)) for k in TRIAGE_COLUMNS})
    return buf.getvalue()


def triage_md(multirate: dict, flat: dict, baseline: BaselineCost) -> str:
    L: list[str] = []
    L.append(f"# DSE axis triage — {baseline.workload}\n")
    L.append("> Merlin does not perform DSE. Merlin prevents DSE from optimizing the wrong "
             "abstraction.\n")
    L.append("> ⚠️ **QUANTITATIVE — uncalibrated.** These gap_closure / priority numbers are only "
             "as good as the baseline cost. Trust them only where `evidence_type` is "
             "`measured`/`calibrated`; otherwise read them as analytical ordering, not validated "
             "magnitudes. The structural results live in `dse_candidate_axes.md` and "
             "`capture_fidelity_report.md`.\n")
    tg = baseline.target_gap_ms
    L.append(f"- baseline total: **{baseline.baseline_total_ms:g} ms**")
    if baseline.target_total_ms is not None:
        L.append(f"- target total: **{baseline.target_total_ms:g} ms**")
        L.append(f"- target gap: **{tg:g} ms**"
                 + ("" if (tg or 0) > 0 else "  _(baseline already meets target)_"))
    else:
        L.append("- target total: _none provided_ (reporting baseline share only)")
    L.append("")
    L.append("`gap_closure = (baseline_total - intervention_total) / (baseline_total - "
             "target_total)`  ·  "
             "`priority_score = gap_closure * confidence * legality / max(cost_tier, 1)`")
    L.append("")
    for label, tr in (("Multi-rate representation", multirate), ("Flat representation", flat)):
        L.append(f"## {label}\n")
        L.append("| axis | gap_closure | priority | evidence | conf | legal | tier | benefit_ms | reason |")
        L.append("|------|-------------|----------|----------|------|-------|------|------------|--------|")
        for r in tr["axes"]:
            gc = "n/a" if r["gap_closure"] is None else f"{r['gap_closure']:.3f}"
            ps = "n/a" if r["priority_score"] is None else f"{r['priority_score']:.4f}"
            L.append(f"| {r['axis']} | {gc} | {ps} | {r['evidence_type']} | "
                     f"{r['confidence']:.2f} | {r['legality']} | {r['cost_tier']} | "
                     f"{r['benefit_ms']:.3f} | {r['reason']} |")
        L.append("")
    # Headline flip
    L.append("## Why representation matters\n")
    flat_rec = _top_recommended(flat)
    multi_rec = _top_recommended(multirate)
    L.append(f"- Flat capture top axes: {', '.join(flat_rec) or '_none with a gap to close_'}")
    L.append(f"- Multi-rate top axes: {', '.join(multi_rec) or '_none_'}")
    gained = [a for a in multi_rec if a not in flat_rec]
    if gained:
        L.append(f"- Axes the flat capture **hides**: **{', '.join(gained)}** — these only "
                 "become worth exploring once the K-loop / loop-invariant reuse is visible.")
    L.append("")
    return "\n".join(L)


def _top_recommended(tr: dict, n: int = 3) -> list[str]:
    out = [r["axis"] for r in tr["axes"]
           if r["priority_score"] is not None and r["priority_score"] > 0 and r["legality"]]
    return out[:n]


# --------------------------------------------------------------- flat-vs-multirate diff

def _diff_row(rep: Representation) -> dict:
    return {
        "workload": rep.workload,
        "representation": rep.name,
        "K": rep.K,
        "H": rep.H,
        "control_rate_hz": rep.control_rate_hz,
        "replan_deadline_ms": rep.replan_deadline_ms,
        "deadline_visible": rep.deadline_visible,
        "visible_weight_reuse": rep.visible_weight_reuse,
        "visible_prefix_kv_reuse": rep.visible_prefix_kv_reuse,
        "dispatches_per_replan": rep.dispatches_per_replan,
        "work_per_dispatch": rep.work_per_dispatch,
        "recommended_axes": rep.recommended_axis_names,
        "deprioritized_axes": [a["axis"] for a in rep.deprioritized_axes],
        "evidence_type": "structural_bound",
    }


def flat_vs_multirate_csv(flat: Representation, multirate: Representation) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=_DIFF_COLUMNS, extrasaction="ignore")
    w.writeheader()
    for rep in (flat, multirate):
        row = _diff_row(rep)
        w.writerow({k: _join(row.get(k)) for k in _DIFF_COLUMNS})
    return buf.getvalue()


# --------------------------------------------------------------------- bottleneck

_BREAKDOWN_COLUMNS = ["workload", "component", "ms", "share", "evidence_type"]


def bottleneck_breakdown_csv(baseline: BaselineCost) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=_BREAKDOWN_COLUMNS)
    w.writeheader()
    total = baseline.baseline_total_ms or 1.0
    for comp in sorted(baseline.components, key=lambda c: -baseline.components[c]):
        ms = baseline.components[comp]
        w.writerow({
            "workload": baseline.workload, "component": comp,
            "ms": round(ms, 6), "share": round(ms / total, 6),
            "evidence_type": baseline.evidence_for(comp),
        })
    return buf.getvalue()


# --------------------------------------------------------------- negative control

def negative_control_md(triage_multirate: dict, baseline: BaselineCost) -> str:
    L: list[str] = []
    wl = baseline.workload
    L.append(f"# Negative control — {wl}\n")
    L.append("A workload with no cross-step reuse (K=1, no loop-invariant weights) must NOT "
             "trigger residency / autonomous-loop recommendations. This guards against a tool "
             "that always recommends exposing hardware.\n")
    checks = {
        "resident_packed_weights": "residency",
        "autonomous_K_loop": "autonomous K-loop",
    }
    L.append("| axis | legality | gap_closure | priority | verdict |")
    L.append("|------|----------|-------------|----------|---------|")
    ok = True
    for axis in checks:
        r = next((x for x in triage_multirate["axes"] if x["axis"] == axis), None)
        if r is None:
            continue
        gc = "n/a" if r["gap_closure"] is None else f"{r['gap_closure']:.3f}"
        ps = "n/a" if r["priority_score"] is None else f"{r['priority_score']:.4f}"
        deprioritized = (r["legality"] == 0) or (r["priority_score"] in (None, 0)) \
            or (r["gap_closure"] in (None, 0))
        ok = ok and deprioritized
        verdict = "deprioritized ✅" if deprioritized else "RECOMMENDED ❌"
        L.append(f"| {axis} | {r['legality']} | {gc} | {ps} | {verdict} |")
    L.append("")
    L.append(f"**Result: {'PASS' if ok else 'FAIL'}** — residency/autonomous-loop features "
             f"are {'correctly not recommended' if ok else 'incorrectly recommended'} for the "
             "no-reuse control.\n")
    return "\n".join(L)
