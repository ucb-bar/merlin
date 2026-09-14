#!/usr/bin/env python3
"""Observational comparison restricted to family × shape cells shared by all targets."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from plot_common import *

use_merlin_style()
d = load_snapshot()
caps = d["capsules"]
cells = defaultdict(lambda: defaultdict(list))
for row in caps:
    cells[(row["family"], row["shape_regime"])][row["target"]].append(row)
shared = sorted(k for k, by_target in cells.items() if all(t in by_target for t in TARGETS))
labels = [f"{family.title()} · {shape}" for family, shape in shared]

rates = np.full((len(shared), 3), np.nan)
times = np.full_like(rates, np.nan)
counts = np.zeros_like(rates)
passes = np.zeros_like(rates)
for y, key in enumerate(shared):
    for x, target in enumerate(TARGETS):
        rows = cells[key][target]
        counts[y, x] = len(rows)
        passes[y, x] = sum(r["status"] == "pass" for r in rows)
        rates[y, x] = passes[y, x] / counts[y, x]
        vals = [r["l2_wall_s"] for r in rows if r["l2_wall_s"]]
        if vals:
            times[y, x] = np.median(vals)

fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.0), gridspec_kw={"width_ratios": [1, 1.05]})
cmap_rate = matplotlib.colors.LinearSegmentedColormap.from_list("merlin_rate", [MAUVE, GOLD, SAGE])
im0 = axes[0].imshow(rates, vmin=0, vmax=1, cmap=cmap_rate)
for y in range(len(shared)):
    for x in range(3):
        axes[0].text(x, y, f"{int(passes[y,x])}/{int(counts[y,x])}", ha="center", va="center", fontweight="bold")
axes[0].set_xticks(range(3), [LABELS[t] for t in TARGETS])
axes[0].set_yticks(range(len(shared)), labels)
axes[0].text(-.03, 1.025, "(a) pass rate in shared coarse strata", transform=axes[0].transAxes, color=BLUE, fontweight="bold")
c0 = fig.colorbar(im0, ax=axes[0], fraction=.045, pad=.03)
c0.set_label("Pass fraction")

positive = times[np.isfinite(times) & (times > 0)]
norm = matplotlib.colors.LogNorm(vmin=max(positive.min(), .03), vmax=positive.max())
cmap_time = matplotlib.colors.LinearSegmentedColormap.from_list("merlin_time", [BG, GOLD, MAUVE])
im1 = axes[1].imshow(times, norm=norm, cmap=cmap_time)
for y in range(len(shared)):
    for x in range(3):
        val = times[y, x]
        axes[1].text(x, y, "—" if not np.isfinite(val) else f"{val:.1f}s", ha="center", va="center", fontweight="bold", fontsize=9)
axes[1].set_xticks(range(3), [LABELS[t] for t in TARGETS])
axes[1].set_yticks(range(len(shared)), [])
axes[1].text(-.03, 1.025, "(b) median L2 adapter time", transform=axes[1].transAxes, color=BLUE, fontweight="bold")
c1 = fig.colorbar(im1, ax=axes[1], fraction=.045, pad=.03)
c1.set_label("Seconds (log color scale)")
for ax in axes:
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(1.0)
fig.tight_layout(w_pad=1.5)
save_all(fig, "fig5_matched_cohorts")
