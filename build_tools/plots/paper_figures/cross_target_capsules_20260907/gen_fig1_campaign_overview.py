#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_common import *

use_merlin_style()
d = load_snapshot()
rows = {r["target"]: r for r in d["campaigns"]}
x = np.arange(3)

fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))
for ax in axes:
    style_ax(ax)

# Coverage: normalized and raw together.
vals = [100 * rows[t]["pass_fraction"] for t in TARGETS]
for i, t in enumerate(TARGETS):
    vbars(axes[0], [i], [vals[i]], COLORS[t], width=.58)
    axes[0].text(i, vals[i] + 2.0, f"{rows[t]['passed']}/{rows[t]['capsules']}", ha="center", fontweight="bold")
axes[0].set_ylim(0, 108)
axes[0].set_ylabel("Latest pass rate (%)")
axes[0].set_xticks(x, [LABELS[t] for t in TARGETS])
axes[0].text(-.13, 1.04, "(a) coverage", transform=axes[0].transAxes, fontweight="bold", color=BLUE)

# Agent time decomposition.
think = np.array([rows[t]["think_generate_s"] / 3600 for t in TARGETS])
tool = np.array([rows[t]["tool_wait_s"] / 3600 for t in TARGETS])
axes[1].bar(x, think, .58, color=GOLD, edgecolor=INK, linewidth=1.3, zorder=3, label="think + generate")
axes[1].bar(x, tool, .58, bottom=think, color=SLATE, edgecolor=INK, linewidth=1.3, zorder=3, label="tools + wait")
for i, total in enumerate(think + tool):
    block_shadow(axes[1], i - .29, 0, .58, total, z=2.4)
    axes[1].text(i, total + .25, f"{total:.1f} h", ha="center", fontweight="bold")
axes[1].set_ylim(0, max(think + tool) * 1.14)
axes[1].set_ylabel("Measured agent span (hours)")
axes[1].set_xticks(x, [LABELS[t] for t in TARGETS])
axes[1].legend(loc="upper right", fontsize=9)
axes[1].text(-.13, 1.04, "(b) activity", transform=axes[1].transAxes, fontweight="bold", color=BLUE)

# Token composition from a consistent response-level rollout snapshot.
cached = np.array([rows[t]["tokens_cache_read"] / 1e6 for t in TARGETS])
fresh = np.array([rows[t]["tokens_fresh_input"] / 1e6 for t in TARGETS])
out = np.array([rows[t]["tokens_output"] / 1e6 for t in TARGETS])
axes[2].bar(x, cached, .58, color=SLATE, edgecolor=INK, linewidth=1.3, zorder=3, label="cached input")
axes[2].bar(x, fresh, .58, bottom=cached, color=GOLD, edgecolor=INK, linewidth=1.3, zorder=3, label="fresh input")
axes[2].bar(x, out, .58, bottom=cached + fresh, color=SAGE, edgecolor=INK, linewidth=1.3, zorder=3, label="output")
for i, total in enumerate(cached + fresh + out):
    block_shadow(axes[2], i - .29, 0, .58, total, z=2.4)
    axes[2].text(i, total + 5, f"{rows[TARGETS[i]]['cache_read_share']*100:.1f}% cached", ha="center", fontweight="bold", fontsize=8.5)
axes[2].set_ylabel("Rollout tokens (million)")
axes[2].set_xticks(x, [LABELS[t] for t in TARGETS])
axes[2].legend(loc="upper right", fontsize=9)
axes[2].set_ylim(0, max(cached + fresh + out) * 1.18)
axes[2].text(-.13, 1.04, "(c) telemetry", transform=axes[2].transAxes, fontweight="bold", color=BLUE)

fig.tight_layout(w_pad=2.2)
save_all(fig, "fig1_campaign_overview")
