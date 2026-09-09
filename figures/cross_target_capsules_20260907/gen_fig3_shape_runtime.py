#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

from plot_common import *

use_merlin_style()
d = load_snapshot()
caps = d["capsules"]
regimes = [x for x, _ in Counter(r["shape_regime"] for r in caps).most_common(8)]

fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
ax = axes[0]
style_ax(ax)
offsets = [-.24, 0, .24]
target_markers = {"gemmini": "o", "atlas": "s", "radiance": "^"}
for ti, target in enumerate(TARGETS):
    medians = []
    for regime in regimes:
        vals = [r["l2_wall_s"] for r in caps if r["target"] == target and r["shape_regime"] == regime and r["l2_wall_s"]]
        medians.append(np.median(vals) if vals else np.nan)
    ax.scatter(np.arange(len(regimes)) + offsets[ti], medians, s=62, marker=target_markers[target],
               color=COLORS[target], edgecolor=INK, linewidth=1.0, zorder=3, label=LABELS[target])
ax.set_yscale("log")
ax.set_ylabel("Median L2 adapter wall time (s, log)")
ax.set_xticks(range(len(regimes)), regimes, rotation=35, ha="right")
ax.legend(fontsize=9)
ax.text(-.08, 1.04, "(a) shape regime", transform=ax.transAxes, fontweight="bold", color=BLUE)

ax = axes[1]
style_ax(ax)
markers = {"pass": "o", "fail": "X", "error": "X", "declined": "s"}
for target in TARGETS:
    rows = [r for r in caps if r["target"] == target and r["input_elements"] > 0 and r["l2_wall_s"]]
    for status in sorted({r["status"] for r in rows}):
        subset = [r for r in rows if r["status"] == status]
        ax.scatter([r["input_elements"] for r in subset], [r["l2_wall_s"] for r in subset],
                   s=34 if status == "pass" else 54,
                   marker=target_markers[target] if status == "pass" else markers.get(status, "X"),
                   facecolors=COLORS[target] if status == "pass" else "none", edgecolors=COLORS[target],
                   linewidths=1.2, alpha=.8, label=LABELS[target] if status == "pass" else None, zorder=3)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Declared input elements (log)")
ax.set_ylabel("L2 adapter wall time (s, log)")
ax.legend(fontsize=9)
ax.text(.02, .98, "Open × = non-pass", transform=ax.transAxes, va="top", color=GOLD, fontweight="bold")
ax.text(-.08, 1.04, "(b) problem size", transform=ax.transAxes, fontweight="bold", color=BLUE)

fig.tight_layout(w_pad=2.2)
save_all(fig, "fig3_shape_runtime")
