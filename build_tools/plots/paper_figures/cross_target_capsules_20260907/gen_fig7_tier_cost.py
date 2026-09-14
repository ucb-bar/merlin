#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_common import *

use_merlin_style()
d = load_snapshot()
caps = d["capsules"]
rows = {r["target"]: r for r in d["campaigns"]}
x = np.arange(3)
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
for ax in axes:
    style_ax(ax)

def percentiles(target, field):
    vals = np.array([r[field] for r in caps if r["target"] == target and r[field] is not None], dtype=float)
    return (np.median(vals), np.percentile(vals, 95), len(vals)) if len(vals) else (np.nan, np.nan, 0)

l2 = [percentiles(t, "l2_wall_s") for t in TARGETS]
for i, t in enumerate(TARGETS):
    median, p95, n = l2[i]
    axes[0].plot([i, i], [median, p95], color=INK, lw=1.4, zorder=4)
    axes[0].scatter([i], [median], marker="o", s=78, color=COLORS[t], edgecolor=INK, zorder=5)
    axes[0].scatter([i], [p95], marker="D", s=34, color=GOLD, edgecolor=INK, zorder=5)
    axes[0].text(i, median/1.35, f"n={n}", ha="center", fontweight="bold", fontsize=8.5)
axes[0].set_yscale("log")
axes[0].set_ylabel("L2 adapter wall time (s, log)")
axes[0].set_xticks(x, [LABELS[t] for t in TARGETS])
axes[0].text(-.13, 1.04, "(a) functional tier", transform=axes[0].transAxes, fontweight="bold", color=BLUE)

l3 = [percentiles(t, "l3_wall_s") for t in TARGETS]
for i, t in enumerate(TARGETS):
    median, p95, n = l3[i]
    if n:
        axes[1].plot([i, i], [median, p95], color=INK, lw=1.4, zorder=4)
        axes[1].scatter([i], [median], marker="o", s=78, color=COLORS[t], edgecolor=INK, zorder=5)
        axes[1].scatter([i], [p95], marker="D", s=34, color=GOLD, edgecolor=INK, zorder=5)
        axes[1].text(i, median/1.7, f"n={n}", ha="center", fontweight="bold", fontsize=8.5)
    else:
        axes[1].text(i, .12, "N/A", ha="center", color=GOLD, fontweight="bold")
axes[1].set_yscale("log")
axes[1].set_ylim(.02, max(v[1] for v in l3 if v[2]) * 2.0)
axes[1].set_ylabel("Timed L3 adapter wall time (s, log)")
axes[1].set_xticks(x, [LABELS[t] for t in TARGETS])
axes[1].text(-.13, 1.04, "(b) RTL tier attempts", transform=axes[1].transAxes, fontweight="bold", color=BLUE)

fresh = np.array([rows[t]["l3_pass_executed"] for t in TARGETS])
carried = np.array([rows[t]["l3_pass_carried"] for t in TARGETS])
unspecified = np.array([rows[t]["l3_pass"] for t in TARGETS]) - fresh - carried
axes[2].bar(x, fresh, .58, color=SAGE, edgecolor=INK, linewidth=1.3, zorder=3, label="fresh pass")
axes[2].bar(x, carried, .58, bottom=fresh, color=SLATE, edgecolor=INK, linewidth=1.3, zorder=3, label="carried pass")
axes[2].bar(x, unspecified, .58, bottom=fresh+carried, color=GOLD, edgecolor=INK, linewidth=1.3, zorder=3, label="origin unspecified")
for i, total in enumerate(fresh + carried + unspecified):
    if total:
        block_shadow(axes[2], i-.29, 0, .58, total, z=2.4)
        axes[2].text(i, total+2, f"{total}", ha="center", fontweight="bold")
axes[2].set_ylim(0, max(fresh+carried+unspecified)*1.18)
axes[2].set_ylabel("Latest capsules with L3 pass")
axes[2].set_xticks(x, [LABELS[t] for t in TARGETS])
axes[2].legend(fontsize=8.5)
axes[2].text(-.13, 1.04, "(c) evidence provenance", transform=axes[2].transAxes, fontweight="bold", color=BLUE)

fig.tight_layout(w_pad=2.0)
save_all(fig, "fig7_tier_cost")
