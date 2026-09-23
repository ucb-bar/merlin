#!/usr/bin/env python3
"""Render the Gemmini perf-bench results as styled figures (see perf_style.py).

Figures:
  fig_cycles.png        — cycle-accurate cycles per kernel x approach (verilator small + FireSim big),
                          grouped bars, value labels, headline badge.
  fig_capability.png    — correctness/capability heatmap table (spike, 24 kernels x 5 approaches).
  fig_spike_not_timing  — methodology: spike cycles plateau (functional) while RTL cycles scale.

Usage: gen_perf_plots.py [--run-id perf_full_0001]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import _pbcommon as PB
import perf_style as S
import perf_reporting as PR

ARMS5 = ["golden", "baseline", "merlin_targetgen", "iree_dialect", "merlin_native"]


def _load(run_id):
    run = PB.RUNS / run_id
    PR.refuse_legacy_cross_approach(run, "gen_perf_plots.py")
    pr = json.loads((run / "perf_results.json").read_text())
    fs_path = run / "firesim_arm_results.json"
    fs = json.loads(fs_path.read_text()) if fs_path.is_file() else {}
    return run, pr, fs


def _cyc_accurate(row, fs, arm):
    """Cycle-accurate cycles for (kernel, arm): verilator if present, else FireSim L5."""
    v = ((row["approaches"].get(arm, {}) or {}).get("per_sim") or {}).get("verilator") or {}
    if v.get("cycles"):
        return v["cycles"], "L3"
    f = (fs.get(row["kernel"], {}) or {}).get(arm) or {}
    if f.get("cycles"):
        return f["cycles"], "L5"
    return None, None


def fig_cycles(run, pr, fs):
    """Cycle-accurate comparison of the generated COMPILER BACKENDS (v0 baseline-gen, v1 merlin-gen)
    against the hand-tuned golden C reference. v0≈v1 on matmul/attention is EXPECTED, not a coincidence:
    there is essentially one canonical WS tiling for a fixed shape on the 16×16 array, so any correct
    backend converges to the same RoCC (v1 differs only by a few epilogue cyc). The v1 backend's real
    advantage is CAPABILITY (it alone lowers conv2d+movement — see fig_capability), not matmul speed.
    NB: these are codegen backends, NOT the agentic baseline-vs-merlin agents (that A/B = authoring
    effort, in capsule_bench). IREE excluded here (diff. verify + 10-40x outlier; see fig_iree_profile)."""
    series = ["golden", "baseline", "merlin_targetgen"]   # hand-C ref + two generated backends (v0, v1)
    rows = []
    for r in sorted(pr, key=lambda x: x["macs"]):
        vals = {a: _cyc_accurate(r, fs, a)[0] for a in series}
        if any(vals.values()):
            rows.append((r["kernel"], r["macs"], vals))
    if not rows:
        return None
    labels = [k.split("_", 1)[0] + "\n" + f"{m//1000}K" if m >= 1000 else k.split("_", 1)[0]
              for k, m, _ in rows]
    x = np.arange(len(rows)); w = 0.26
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.15), 5.0))
    n_l5 = 0
    for i, a in enumerate(series):
        ys = [v[a] if v[a] else np.nan for _, _, v in rows]
        bars = ax.bar(x + (i - 1) * w, ys, w, label=S.LABEL[a], color=S.COLOR[a],
                      edgecolor=S.INK, linewidth=1.0)
        # stratify oracle tier: hatch FireSim-L5 bars so L3 vs L5 is visible, not just a footnote
        for b, (kname, _m, _v) in zip(bars, rows):
            t = _cyc_accurate(next(rr for rr in pr if rr["kernel"] == kname), fs, a)[1]
            if t == "L5":
                b.set_hatch("//"); n_l5 += 1
        S.bar_labels(ax, bars, fmt="{:.0f}", fontsize=7.5, rot=90)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("cycles (cycle-accurate RTL)")
    ax.set_title("Gemmini cycles per kernel — baseline-gen vs merlin-gen vs hand-tuned (lower=faster)", pad=34)
    allv = [v[a] for _, _, v in rows for a in series if v[a]]
    if allv:
        ax.set_ylim(0, max(allv) * 1.30)
    # tier legend proxy (hatch = L5)
    from matplotlib.patches import Patch
    hands, labs = ax.get_legend_handles_labels()
    hands.append(Patch(facecolor="white", edgecolor=S.INK, hatch="//"))
    labs.append("FireSim L5 (hatched); plain = verilator L3")
    ax.legend(hands, labs, loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2, fontsize=8.5)
    S.caption(fig, "Cycle-accurate RTL: plain bars = verilator L3 (≤32K MACs), hatched = FireSim L5 — SAME "
              "RTL, directly comparable. Single run per cell (RTL deterministic). v0 (baseline-gen) and v1 "
              "(merlin-gen) are generated COMPILER BACKENDS; they emit the SAME canonical WS matmul tiling so "
              "matmul cycles match within ~0.05% (expected — one optimal tiling per shape), v1 differing only "
              "in epilogue. v1's real edge is capability (conv+movement; see fig_capability), not matmul speed. "
              "golden = hand-C. These are backends, NOT the agentic agents. IREE: see fig_iree_profile.")
    out = PB.REPORTS / "fig_cycles.png"
    S.save_fig(fig, out)
    return out


def fig_capability(run, pr, fs):
    """Spike correctness/capability heatmap: 24 kernels x 5 approaches (image copy 4/5 style)."""
    rows = pr
    fig, ax = plt.subplots(figsize=(8.6, max(5, len(rows) * 0.32 + 1.2)))
    ax.set_xlim(0, len(ARMS5)); ax.set_ylim(-1.4, len(rows) + 1.0); ax.invert_yaxis()
    ax.axis("off")
    # header
    for j, a in enumerate(ARMS5):
        ax.text(j + 0.5, -0.15, S.LABEL[a].replace(" — ours", "\n(ours)"), ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", rotation=0)
    for i, r in enumerate(rows):
        ax.text(-0.1, i + 0.5, r["kernel"][:26], ha="right", va="center", fontsize=7.5)
        for j, a in enumerate(ARMS5):
            ps = ((r["approaches"].get(a, {}) or {}).get("per_sim") or {}).get("spike") or {}
            ap = r["approaches"].get(a, {}) or {}
            if ps.get("correct") is True:
                fc, txt = S.HEAT_GOOD, "✓"
            elif ps.get("correct") is False:
                fc, txt = S.HEAT_BAD, "✗"
            elif ap.get("error") and "deferred" in str(ap.get("error", "")):
                fc, txt = "#E8E2D4", "·"
            else:
                fc, txt = "#E8E2D4", "·"
            ax.add_patch(plt.Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, fc=fc, ec="white", lw=1.5))
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white")
    # per-approach pass counts
    for j, a in enumerate(ARMS5):
        n = sum(1 for r in rows if (((r["approaches"].get(a, {}) or {}).get("per_sim") or {})
                                    .get("spike") or {}).get("correct"))
        ax.text(j + 0.5, len(rows) + 0.4, f"{n}/{len(rows)}", ha="center", va="top", fontsize=9,
                fontweight="bold")
    ax.set_title("Functional correctness & op coverage (spike L2) — NOT a timing result",
                 pad=42)
    S.caption(fig, "spike L2 = compiles + functionally correct (exact-int == golden), NOT performance "
              "(spike doesn't model the systolic array — see spike≠timing figure). ✓ correct · ✗ cannot "
              "lower/wrong · '·' not attempted (golden conv template deferred). Capability story: only "
              "merlin-gen (v1) covers conv2d + movement. Timing lives only in the cycle figures.")
    out = PB.REPORTS / "fig_capability.png"
    S.save_fig(fig, out)
    return out


def fig_spike_not_timing(run, pr, fs):
    """Methodology figure: spike cycles plateau (functional) vs RTL cycles that scale with MACs."""
    pts_spike, pts_rtl = [], []
    for r in pr:
        macs = r["macs"]
        if macs <= 0:
            continue
        sp = ((r["approaches"].get("golden", {}) or {}).get("per_sim") or {}).get("spike") or {}
        if sp.get("cycles"):
            pts_spike.append((macs, sp["cycles"]))
        rtl, _ = _cyc_accurate(r, fs, "golden")
        if rtl:
            pts_rtl.append((macs, rtl))
    if not pts_rtl:
        return None
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for pts, color, lab, mk in [(pts_spike, S.COLOR["baseline"], "spike (functional)", "s"),
                                (pts_rtl, S.COLOR["merlin_targetgen"], "verilator/FireSim (RTL)", "o")]:
        pts = sorted(pts)
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=70, color=color, edgecolor=S.INK, lw=1.0, label=lab, marker=mk,
                       zorder=5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("kernel size (MACs)"); ax.set_ylabel("reported cycles (golden)")
    ax.set_title("Why spike ≠ performance: functional sim doesn't model the systolic array")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.42), fontsize=10)
    if pts_spike:
        sx = sorted(pts_spike)
        S.badge(ax, sx[-1][0], sx[-1][1], "spike plateaus ~120 cyc\n(util > 100% — impossible)",
                color=S.COLOR["baseline"], fontsize=8.5)
    S.caption(fig, "Each point = golden on one kernel. spike (functional) cycles are flat ~120 regardless "
              "of MACs ⇒ would imply util>100% (impossible) ⇒ spike is NOT a timing oracle. RTL "
              "(verilator+FireSim) cycles scale with work. Single run per point.")
    out = PB.REPORTS / "fig_spike_not_timing.png"
    S.save_fig(fig, out)
    return out


def fig_iree_profile(run, pr, fs):
    """IREE profiled on its CORRECT oracle (FireSim L5): 3-way golden/merlin/IREE on the kernels where
    all three have FireSim cells. IREE can't run on verilator (530KB runtime at ~kHz), so this is how we
    still profile it — via the IREE ELF's own per-dispatch rdcycle dump. Log cycles (IREE is 10-40x) +
    utilization% (IREE 1-6%): the honest 'IREE dialect path is slow' result, not hidden."""
    macs = {r["kernel"]: r["macs"] for r in pr}
    arms = ["golden", "baseline", "merlin_targetgen", "iree_dialect"]  # hand ref + both agentic arms + IREE
    rows = []
    for k in sorted(fs):
        c = {a: (fs[k].get(a) or {}).get("cycles") for a in arms}
        if all(c.values()):
            rows.append((k, c))
    if not rows:
        return None
    n = len(arms)
    labels = [k.split("_", 1)[0] for k, _ in rows]
    x = np.arange(len(rows)); w = 0.8 / n
    fig, (axc, axu) = plt.subplots(1, 2, figsize=(max(12, len(rows) * 1.1), 5.2))
    for i, a in enumerate(arms):
        off = (i - (n - 1) / 2) * w
        cyc = [c[a] for _, c in rows]
        axc.bar(x + off, cyc, w, label=S.LABEL[a], color=S.COLOR[a], edgecolor=S.INK, lw=0.7)
        util = [(PB.utilization_pct(macs.get(k, 0), c[a]) if macs.get(k) and c[a] else np.nan)
                for k, c in rows]
        axu.bar(x + off, util, w, color=S.COLOR[a], edgecolor=S.INK, lw=0.7)
    axc.set_yscale("log")
    axc.set_ylabel("cycles (FireSim L5, log)"); axc.set_xticks(x); axc.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    axc.set_title("Cycles — log scale (IREE 10–40× golden)", fontsize=12)
    axc.legend(fontsize=8, loc="upper left")
    axu.set_ylabel("PE-array utilization (%)"); axu.set_xticks(x); axu.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    axu.set_title("Utilization — IREE dialect path is 1–6%", fontsize=12)
    fig.suptitle("FireSim L5 — golden vs baseline-gen vs merlin-gen vs IREE on shared kernels",
                 fontsize=14, fontweight="bold", y=1.0)
    S.caption(fig, f"N={len(rows)} kernels where all four arms have FireSim L5 cells. baseline-gen (v0) and "
              "merlin-gen (v1) are generated COMPILER BACKENDS (not the agentic agents); they emit the same "
              "canonical matmul so cycles match within ~0.05% — expected. IREE 10–40× slower at 1–6% util "
              "(can't run on verilator: 530KB runtime at ~kHz, so FireSim is its oracle; cycles = per-dispatch "
              "rdcycle dump; all-ones self-check rc=0, not exact-int golden). Low IREE util is a real lowering "
              "result, not a measurement artifact.")
    out = PB.REPORTS / "fig_iree_profile.png"
    S.save_fig(fig, out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="perf_full_0001")
    a = ap.parse_args(argv)
    S.use_style()
    run, pr, fs = _load(a.run_id)
    outs = [fig_capability(run, pr, fs), fig_cycles(run, pr, fs), fig_spike_not_timing(run, pr, fs),
            fig_iree_profile(run, pr, fs)]
    for o in outs:
        print(f"wrote {o}" if o else "(skipped a figure — no data yet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
