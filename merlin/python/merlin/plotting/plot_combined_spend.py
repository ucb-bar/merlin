#!/usr/bin/env python3
"""Combined spend-by-model: Muon (radiance) + Gemmini-MX campaigns in one stacked-bar figure.

Reads each campaign's project spend total (no new runs) and renders one bar per model, stacked by
campaign so the bar height is the combined per-model spend. House style imported from merlin_plotstyle.
"""
import matplotlib
matplotlib.use("Agg")
import sys, json, collections
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merlin.plotting.merlin_plotstyle import *          # noqa: F401,F403
use_merlin_style()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def _repo_root():
    from pathlib import Path as _P
    p = _P(__file__).resolve()
    while p != p.parent and not (p / "merlin" / "python").is_dir():
        p = p.parent
    return p
_ROOT = _repo_root()

ROOT = _ROOT
MUON = ROOT / "tmp/kernels/radiance_only_kernels/index/muon-spend_total.json"
MX   = ROOT / "tmp/kernels/mx_gemmini_only_kernels/cost/project-spend-total.json"
OUT  = ROOT / "artifacts/presentation/combined"
OUT.mkdir(parents=True, exist_ok=True)

# different ARNs / aliases -> one canonical display name
CANON = {
    "us.anthropic.claude-sonnet-4-6":     "Claude\nSonnet 4.6",
    "global.anthropic.claude-sonnet-4-6": "Claude\nSonnet 4.6",
    "gemini-3.5-flash":                   "Gemini\n3.5 Flash",
    "gemini-3.1-pro-preview":             "Gemini\n3.1 Pro",
    "qwen.qwen3-coder-480b-a35b-v1:0":    "Qwen3-Coder\n480B",
    "gemini-3-flash-preview":             "Gemini\n3 Flash",
    "gemini-2.5-flash":                   "Gemini\n2.5 Flash",
}


def by_canon(path):
    d = json.load(open(path))
    agg = collections.defaultdict(float)
    for k, v in d["by_model"].items():
        agg[CANON.get(k, k)] += v
    return agg, d["total_usd"], d["calls"]


muon, muon_tot, muon_calls = by_canon(MUON)
mx,   mx_tot,   mx_calls   = by_canon(MX)

models = set(muon) | set(mx)
combined = {m: muon.get(m, 0) + mx.get(m, 0) for m in models}
models = [m for m in sorted(combined, key=lambda m: -combined[m]) if combined[m] >= 1.0]
tiny = {m: combined[m] for m in combined if 0 < combined[m] < 1.0}

x = np.arange(len(models))
w = 0.6
muon_v = [muon.get(m, 0) for m in models]
mx_v = [mx.get(m, 0) for m in models]

fig, ax = plt.subplots(figsize=(11, 6.4))
style_ax(ax)
for xi, mv, xv in zip(x, muon_v, mx_v):
    tot = mv + xv
    # one hard 3-D block behind the whole stacked bar (not per segment)
    block_shadow(ax, xi - w/2, 0, w, tot, z=2.2)
    ax.bar(xi, mv, w, color=NAVY, edgecolor=INK, linewidth=1.3, zorder=3)
    ax.bar(xi, xv, w, bottom=mv, color=MAUVE, edgecolor=INK, linewidth=1.3, zorder=3)
    # segment value labels (only when the segment is tall enough to hold text)
    if mv > 8:
        ax.annotate(f"${mv:.1f}", (xi, mv/2), ha="center", va="center",
                    fontsize=9.5, color=BG, fontweight="bold", zorder=5)
    if xv > 8:
        ax.annotate(f"${xv:.1f}", (xi, mv + xv/2), ha="center", va="center",
                    fontsize=9.5, color=BG, fontweight="bold", zorder=5)
    ax.annotate(f"${tot:.0f}", (xi, tot), xytext=(0, 7), textcoords="offset points",
                ha="center", fontsize=11.5, fontweight="bold", color=INK)

ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10)
ax.set_ylabel("cumulative cost (USD)")
ax.set_ylim(0, max(combined[m] for m in models) * 1.16)

gtot = muon_tot + mx_tot
gcalls = muon_calls + mx_calls
emph(ax, len(models) - 0.5, max(combined[m] for m in models) * 1.07,
     f"combined ${gtot:.0f} · {gcalls:,} calls", color=GOLD, fs=12, ha="right", va="center")
ax.legend(handles=[Patch(fc=NAVY, ec=INK, label=f"Muon / radiance  (${muon_tot:.0f})"),
                   Patch(fc=MAUVE, ec=INK, label=f"Gemmini-MX  (${mx_tot:.0f})")],
          loc="upper center", fontsize=10.5, ncol=1)
if tiny:
    note = " · ".join(f"{m.replace(chr(10),' ')} ${v:.2f}" for m, v in sorted(tiny.items(), key=lambda x: -x[1]))
    fig.text(0.5, 0.015, f"negligible: {note}", ha="center", fontsize=8.3, color=INK, fontstyle="italic")
title(ax, "Where the budget went, by model")
suptitle(fig, "Spend by model — Muon + Gemmini-MX campaigns combined")
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
fig.savefig(OUT / "fig_spend_by_model_combined.png", bbox_inches="tight", dpi=180, facecolor=BG)
fig.savefig(OUT / "fig_spend_by_model_combined.svg", bbox_inches="tight", facecolor=BG)
print("wrote ->", OUT / "fig_spend_by_model_combined.png")
print(f"combined total ${gtot:.2f}, calls {gcalls}, models {models}")
