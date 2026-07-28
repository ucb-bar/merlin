"""3-arm agentic A/B/C figures (baseline · merlin · merlin+CIRCT), pilot-honest, BOTH dimensions.

Reads experiments/capsule_bench/targets/gemmini/reports/agentic_results.json (built by agg_agentic_results.py,
keyed by bundle_id into 3 arms). Two dimensions:
  • authoring effort      — fig_agentic_effort, fig_agentic_convergence, fig_agentic_per_capsule_effort
  • dialect completeness  — fig_agentic_completeness (passed of 25, public+hidden), fig_agentic_coverage
                            (per-op-class, the conv/movement gap)
Individual points labelled, N per arm shown; never fabricated variance. Figures -> reports/fig_agentic_*.png
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import _pbcommon as PB
import perf_style as S

AG = json.loads((PB.REPO / "experiments/capsule_bench/targets/gemmini/reports/agentic_results.json").read_text())
ARMS = AG.get("arm_order") or ["baseline", "merlin", "merlin_rtlchecks"]
LABEL = {"baseline": "baseline", "merlin": "merlin", "merlin_rtlchecks": "merlin+CIRCT"}
COLOR = {"baseline": "#D98C84", "merlin": "#E6B84C", "merlin_rtlchecks": "#6E93B0"}
XPOS = {a: i for i, a in enumerate(ARMS)}


def _valid(arm):
    return [r for r in AG["arms"].get(arm, []) if r.get("valid")]


def _xlabels(ax):
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([f"{LABEL[a]}\nN={len(_valid(a))}" for a in ARMS], fontsize=8.5)
    ax.set_xlim(-0.6, len(ARMS) - 0.4)


def _save(fig, name):
    out = PB.REPORTS / f"fig_agentic_{name}.png"
    S.save_fig(fig, out)
    print(f"wrote {out}")
    return out


def _strip(ax, getval, sc=1.0):
    """individual points (jittered) + per-arm median bar, for one metric."""
    np.random.seed(0)
    anyv = []
    for a in ARMS:
        vs = [getval(r) * sc for r in _valid(a) if getval(r) is not None]
        anyv += vs
        if not vs:
            continue
        x = XPOS[a] + np.random.uniform(-.05, .05, len(vs))
        ax.scatter(x, vs, s=90, color=COLOR[a], edgecolor=S.INK, lw=1, zorder=5)
        ax.hlines(np.median(vs), XPOS[a] - .18, XPOS[a] + .18, color=S.INK, lw=1.5, zorder=6)
    _xlabels(ax)
    ax.set_ylim(0, max(anyv) * 1.18 if anyv else 1)


def fig_effort():
    metrics = [("cost (USD)", lambda r: r.get("cost_usd"), 1), ("tokens (M)", lambda r: r.get("tokens_total"), 1e-6),
               ("tool calls", lambda r: r.get("tool_calls"), 1), ("wall (min)", lambda r: r.get("wall_s"), 1 / 60),
               ("rounds", lambda r: r.get("n_rounds"), 1)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4.3))
    for ax, (lab, fn, sc) in zip(axes, metrics):
        _strip(ax, fn, sc); ax.set_title(lab, fontsize=11)
    fig.suptitle("Authoring effort — baseline vs merlin vs merlin+CIRCT (dot=run, bar=median)",
                 fontsize=15, fontweight="bold", y=1.02)
    S.caption(fig, "Each dot = one valid converged run; bar = per-arm median. N per arm on the axis. "
              "Same task/model/grading — only the authoring aids differ (see ARMS.md). Cost = real API "
              "pricing. No variance claimed where N=1.")
    return _save(fig, "effort")


def fig_convergence():
    fig, ax = plt.subplots(figsize=(8, 4.8))
    seen = set()
    for a in ARMS:
        for r in _valid(a):
            rd = r.get("rounds") or []
            xs = [d["round"] for d in rd if d.get("round") is not None]
            ys = [d["n_passed"] for d in rd if d.get("n_passed") is not None]
            if xs and ys and len(xs) == len(ys):
                lab = LABEL[a] if a not in seen else None
                seen.add(a)
                ax.step(xs, ys, where="post", color=COLOR[a], lw=2, alpha=.85, marker="o", label=lab)
    ax.set_xlabel("agent round"); ax.set_ylabel("pilot capsules passing (of 4)")
    ax.set_title("Per-round convergence — capsules passing vs round", pad=10)
    ax.set_yticks(range(0, 5))
    if seen:
        ax.legend(fontsize=8.5, loc="lower right")
    S.caption(fig, "Pilot 4-capsule QA loop; converge when all 4 pass. One line per valid run, coloured by "
              "arm. Fewer rounds = faster convergence. N per arm varies (see effort figure).")
    return _save(fig, "convergence")


def fig_completeness():
    """Dialect completeness: full-suite passed of 25 (public+hidden stacked), best valid run per arm."""
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    n_total = AG.get("coverage", {}).get("n_capsules", 25)
    any_data = False
    for a in ARMS:
        runs = [r for r in _valid(a) if r.get("fullsuite") and r["fullsuite"].get("all", {}).get("passed") is not None]
        if not runs:
            ax.text(XPOS[a], 0.5, "audit\npending", ha="center", va="bottom", fontsize=8, color="#8a8276")
            continue
        any_data = True
        best = max(runs, key=lambda r: r["fullsuite"]["all"]["passed"])
        fs = best["fullsuite"]
        pub = fs.get("public", {}).get("passed") or 0
        hid = fs.get("hidden", {}).get("passed") or 0
        ax.bar(XPOS[a], pub, 0.6, color=COLOR[a], edgecolor=S.INK, lw=1, label=None)
        ax.bar(XPOS[a], hid, 0.6, bottom=pub, color=COLOR[a], edgecolor=S.INK, lw=1, hatch="//")
        ax.text(XPOS[a], pub + hid + 0.3, f"{pub+hid}/{n_total}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.axhline(n_total, color=S.INK, lw=1, ls=":", alpha=0.5)
    ax.text(len(ARMS) - 0.5, n_total + 0.2, f"all {n_total}", ha="right", va="bottom", fontsize=8, color="#8a8276")
    _xlabels(ax); ax.set_ylabel("capsules passed (RTL oracle L2+L3)")
    ax.set_ylim(0, n_total * 1.12)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc="#CCC", ec=S.INK, label="public (20)"),
                       Patch(fc="#CCC", ec=S.INK, hatch="//", label="hidden (5)")], fontsize=8, loc="upper left")
    ax.set_title("Dialect completeness — full 25-capsule suite (best run/arm)", pad=10)
    S.caption(fig, "Each frozen dialect re-graded on the ENTIRE 25-capsule corpus (20 public + 5 hidden) on "
              "the RTL oracle (L2 spike + L3 verilator) via full_suite_audit. Solid = public, hatched = "
              "hidden. The QA loop only tuned 4 pilot capsules, so this shows how far each dialect "
              "generalises. 'audit pending' = run not yet graded. Best valid run per arm.")
    return _save(fig, "completeness")


def fig_coverage():
    cov = (AG.get("coverage") or {}).get("class_coverage") or {}
    if not cov:
        return None
    # map run-id -> arm, to aggregate per-class coverage by arm (best run/arm)
    rid_arm = {r["run_id"]: a for a in ARMS for r in _valid(a)}
    classes = list(cov)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(classes)); w = 0.8 / max(len(ARMS), 1)
    for j, a in enumerate(ARMS):
        rids = [r["run_id"] for r in _valid(a)]
        vals = [max([cov[c].get(rid, 0) for rid in rids], default=0) for c in classes]
        if not any(vals):
            continue
        ax.bar(x + (j - (len(ARMS) - 1) / 2) * w, vals, w, color=COLOR[a], edgecolor=S.INK, lw=0.8,
               label=f"{LABEL[a]}")
    ax.plot(x, [cov[c]["n"] for c in classes], "k_", markersize=14, label="capsules in class")
    ax.set_xticks(x); ax.set_xticklabels([c.replace("matmul+", "+") for c in classes], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("capsules passed"); ax.legend(fontsize=8)
    ax.set_title("Capability coverage by op-class (full suite, best run/arm)", pad=10)
    S.caption(fig, "Per-op-class capsules passed on the full suite, best valid run per arm. Black ticks = "
              "capsules in that class. Reveals where each arm's dialect generalises (e.g. conv/movement). "
              "Arms with no audited run yet are omitted.")
    return _save(fig, "coverage")


def fig_per_capsule_effort():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.3))
    _strip(a1, lambda r: (r.get("cost_usd") or 0) / 4); a1.set_title("cost per pilot capsule ($)", fontsize=11)
    _strip(a2, lambda r: (r.get("tokens_total") or 0) / 4, 1e-6); a2.set_title("tokens per pilot capsule (M)", fontsize=11)
    fig.suptitle("Downstream efficiency — authoring effort per pilot capsule passed", fontsize=14, fontweight="bold", y=1.02)
    S.caption(fig, "Total effort ÷ 4 (all valid runs converged 4/4 pilot capsules). dot=run, bar=median, "
              "N per arm on axis. Same task/model/grading; only authoring aids differ.")
    return _save(fig, "per_capsule_effort")


def main():
    S.use_style()
    for f in (fig_effort, fig_convergence, fig_completeness, fig_coverage, fig_per_capsule_effort):
        try:
            f()
        except Exception as e:
            print(f"  (skip {f.__name__}: {type(e).__name__}: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
