"""Figures for the abc4 deep analysis -> reports/abc4_analysis/*.png (perf_style)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

EXP = Path("/scratch/agustin/projects/oscar-merlin/experiments/gemmini_capsule_bench_v0")
A = EXP / "reports" / "abc4_analysis"
sys.path.insert(0, str(EXP.parent / "gemmini_perf_bench" / "scripts"))
import perf_style as S
S.use_style()
COL = {"baseline-C++": S.COLOR["golden"], "merlin-xDSL": S.COLOR["merlin_targetgen"],
       "merlin+CIRCT": S.COLOR["iree_dialect"]}
T = json.loads((A / "trajectory.json").read_text())
C = json.loads((A / "circt_vs_verilator.json").read_text())


def fig_circt():
    """The headline: CIRCT verdict vs sim outcome confusion + the false-clean=0 result."""
    conf = C["confusion"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})
    # 2x2 confusion heatmap
    grid = np.array([[conf["true_neg"], conf["false_clean"]], [conf["false_alarm"], conf["true_pos"]]])
    ax.imshow(grid, cmap="Blues", alpha=0.85)
    labs = [["CIRCT ok\n& sim PASS\n(correct skip)", "CIRCT ok\n& sim FAIL\n★ FALSE-CLEAN"],
            ["CIRCT reject\n& sim PASS\n(false alarm)", "CIRCT reject\n& sim FAIL\n(caught)"]]
    for i in range(2):
        for j in range(2):
            c = "#C0392B" if (i == 0 and j == 1 and grid[i, j] > 0) else S.INK
            ax.text(j, i, f"{labs[i][j]}\n\n{grid[i,j]}", ha="center", va="center", fontsize=9,
                    fontweight="bold", color=c)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title("CIRCT static check vs sim outcome\n(119 arm×round×capsule points)", fontsize=11)
    # the decision bar
    ax2.axis("off")
    safe = C["circt_safe_gate"]
    ax2.text(0.0, 0.92, "Can CIRCT replace the sim?", fontsize=13, fontweight="bold", transform=ax2.transAxes)
    lines = [
        f"FALSE-CLEAN (CIRCT-ok but sim-FAIL):  {C['false_clean_count']}",
        f"failures CIRCT caught:  {conf['true_pos']}/{conf['true_pos']+conf['false_clean']}  (100%)",
        f"false alarms:  {conf['false_alarm']}",
        "",
        "[YES] all 21 failures were STRUCTURAL (trace_check)",
        "   → CIRCT-reject ⟹ sim-fail, no exceptions",
        "   → skip the sim on every reject (ms vs min)",
        "",
        "[CAVEAT] CIRCT is structural-only; the 1 numeric",
        "   failure was pre-trace (out of scope).",
        "   → still need ≥1 sim pass to certify numerics.",
        "",
        "Recipe: iterate on CIRCT (instant), skip sim while",
        "rejecting; run ONE sim when CIRCT is clean.",
    ]
    for k, ln in enumerate(lines):
        ax2.text(0.0, 0.82 - k * 0.058, ln, fontsize=9.5, transform=ax2.transAxes,
                 color=("#1a7a3a" if ln.startswith("[YES]") else "#9a6a00" if ln.startswith("[CAVEAT]") else S.INK),
                 fontweight="bold" if (ln.startswith(("[YES]", "[CAVEAT]", "Recipe")) or "FALSE-CLEAN" in ln) else "normal")
    fig.suptitle("Does CIRCT predict the RTL-sim result? — abc4 (N=1/arm)", fontsize=15, fontweight="bold", y=1.0)
    S.caption(fig, "Replayed the EXACT live CIRCT screen (rtl_check_runner.prescreen) over every emitted "
              "trace; verdict vs that round's spike(L2) functional outcome. 0 false-clean = CIRCT never "
              "passed a structurally-broken dialect. Structural-only: cannot certify numerics. N=1/arm.")
    S.save_fig(fig, A / "fig_circt_predicts_sim.png")
    print("wrote fig_circt_predicts_sim.png")


def fig_effort():
    arms = list(T)
    mets = [("cost ($)", "cost_usd", 1), ("tokens (M)", "tokens_total", 1e-6),
            ("tool-calls", "tool_calls", 1), ("self-checks", "n_self_checks", 1),
            ("active wall (min)", "active_wall_min", 1)]
    fig, axes = plt.subplots(1, len(mets), figsize=(14, 4.2))
    for ax, (lab, key, sc) in zip(axes, mets):
        vals = [T[a][key] * sc for a in arms]
        ax.bar(range(len(arms)), vals, color=[COL[a] for a in arms], edgecolor=S.INK, lw=1)
        ax.set_xticks(range(len(arms))); ax.set_xticklabels([a.replace("merlin", "m").replace("baseline-", "") for a in arms], rotation=30, ha="right", fontsize=8)
        ax.set_title(lab, fontsize=11)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    fig.suptitle("Effort to reach 25/25 verilator-correct — all 3 arms converged (N=1/arm)", fontsize=14, fontweight="bold", y=1.02)
    S.caption(fig, "All three reach functional+numerical correctness (25/25 incl. hidden). baseline-C++ ≈ "
              "merlin-xDSL on cost/tokens (xDSL not worse — earlier gap was a broken-setup artifact); "
              "merlin+CIRCT spends ~60% more via 2× self-checks (thorough, but all converge). N=1 — deltas "
              "directional, need N>1 for significance.")
    S.save_fig(fig, A / "fig_effort_abc4.png")
    print("wrote fig_effort_abc4.png")


if __name__ == "__main__":
    fig_circt(); fig_effort()
