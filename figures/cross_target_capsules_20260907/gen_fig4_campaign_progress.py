#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

from plot_common import *

use_merlin_style()
d = load_snapshot()
fig, ax = plt.subplots(figsize=(8.4, 4.8))
style_ax(ax)
markers = {"gemmini": "o", "atlas": "s", "radiance": "^"}
styles = {"gemmini": "-", "atlas": "--", "radiance": "-."}
for target in TARGETS:
    rows = [r for r in d["history"] if r["target"] == target]
    rows.sort(key=lambda r: r["graded_at"])
    t0 = datetime.fromisoformat(rows[0]["graded_at"])
    xs = [(datetime.fromisoformat(r["graded_at"]) - t0).total_seconds() / 3600 for r in rows]
    ys = [100 * r["pass_fraction"] for r in rows]
    ax.plot(xs, ys, color=COLORS[target], lw=2.4, ls=styles[target], marker=markers[target], ms=5,
            label=LABELS[target], zorder=3)
    ax.text(xs[-1] + .08, ys[-1], f"{rows[-1]['passed']}/{rows[-1]['capsules']}", color=COLORS[target], va="center", fontweight="bold")
ax.set_xlabel("Hours since target's first grade")
ax.set_ylabel("Pass rate at each grade (%)")
ax.set_ylim(0, 105)
ax.set_xlim(left=0)
ax.legend(loc="lower right")
ax.text(.01, .98, "Corpus size can change between grades", transform=ax.transAxes, va="top", color=GOLD, fontweight="bold")
fig.tight_layout()
save_all(fig, "fig4_campaign_progress")
