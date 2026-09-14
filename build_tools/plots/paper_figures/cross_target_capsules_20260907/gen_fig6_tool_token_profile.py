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
fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.3))
for ax in axes:
    style_ax(ax)

tools = {(r["target"], r["tool"]): r for r in d["tools"]}
bash = np.array([tools[(t, "Bash")]["calls_started"] for t in TARGETS])
edit = np.array([tools[(t, "Edit")]["calls_started"] for t in TARGETS])
axes[0].bar(x, bash, .58, color=SLATE, edgecolor=INK, linewidth=1.3, zorder=3, label="Bash")
axes[0].bar(x, edit, .58, bottom=bash, color=GOLD, edgecolor=INK, linewidth=1.3, zorder=3, label="Edit")
for i, total in enumerate(bash + edit):
    block_shadow(axes[0], i - .29, 0, .58, total, z=2.4)
    axes[0].text(i, total + max(bash + edit)*.025, f"{total:,}", ha="center", fontweight="bold")
axes[0].set_ylim(0, max(bash + edit) * 1.13)
axes[0].set_ylabel("Tool calls started")
axes[0].set_xticks(x, [LABELS[t] for t in TARGETS])
axes[0].legend(fontsize=9)
axes[0].text(0, 1.04, "(a) tool volume", transform=axes[0].transAxes, fontweight="bold", color=BLUE)

inp = np.array([rows[t]["input_chars"] / 1e6 for t in TARGETS])
out = np.array([rows[t]["output_chars"] / 1e6 for t in TARGETS])
axes[1].bar(x, inp, .58, color=MAUVE, edgecolor=INK, linewidth=1.3, zorder=3, label="input")
axes[1].bar(x, out, .58, bottom=inp, color=SAGE, edgecolor=INK, linewidth=1.3, zorder=3, label="output")
for i, total in enumerate(inp + out):
    block_shadow(axes[1], i - .29, 0, .58, total, z=2.4)
axes[1].set_ylabel("Logged tool transport (million chars)")
axes[1].set_xticks(x, [LABELS[t] for t in TARGETS])
axes[1].legend(fontsize=9)
axes[1].text(0, 1.04, "(b) tool I/O", transform=axes[1].transAxes, fontweight="bold", color=BLUE)

rates = [rows[t]["output_tokens_per_client_turnaround_s"] for t in TARGETS]
for i, t in enumerate(TARGETS):
    vbars(axes[2], [i], [rates[i]], COLORS[t], width=.58)
    axes[2].text(i, rates[i] + .45, f"{rates[i]:.1f}", ha="center", fontweight="bold")
axes[2].set_ylim(0, max(rates) * 1.16)
axes[2].set_ylabel("Output tokens / client-observed response s")
axes[2].set_xticks(x, [LABELS[t] for t in TARGETS])
axes[2].text(0, 1.04, "(c) response throughput", transform=axes[2].transAxes, fontweight="bold", color=BLUE)

fig.tight_layout(w_pad=2.1)
save_all(fig, "fig6_tool_token_profile")
