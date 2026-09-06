#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import COLORS, save_figure


HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "k1_executorch_status_20260906.json").read_text(encoding="utf-8"))


fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.25), gridspec_kw={"wspace": 0.38})

# (a) The exact historical slide result, kept visually separate because it is FP32 kernel routing.
historical = DATA["historical_fp32_bitvla"]
names = list(historical["latency_ms"])
values = [historical["latency_ms"][name] for name in names]
colors = [COLORS["merlin"], COLORS["xnnpack"], COLORS["openblas"]]
xpos = np.arange(len(names))
for x0, name, best, color in zip(xpos, names, values, colors):
    runs = historical["launches_ms"][name]
    axes[0].scatter(np.full(len(runs), x0), runs, s=33, color=color, edgecolor="black",
                    linewidth=0.45, zorder=3)
    axes[0].hlines(best, x0 - 0.25, x0 + 0.25, color="black", linewidth=1.3, zorder=4)
    axes[0].text(x0, max(runs) + 2.5, f"best {best:.1f}", ha="center", va="bottom", fontsize=8.2)
axes[0].set_ylabel("Latency (ms; lower is better)")
axes[0].set_xticks(xpos, ["Merlin", "XNNPACK\nkernels", "OpenBLAS\nkernels"])
axes[0].set_ylim(0, max(values) * 1.22)
axes[0].text(0.02, 0.98, "(a)", transform=axes[0].transAxes,
             ha="left", va="top", fontweight="bold")

# (b) Current one-core INT8 status as a latency ratio. Missing smolVLA is explicit.
rows = DATA["int8_one_core"]["models"]
model_names = [row["model"] for row in rows]
ratios = [(row["executorch_ms"] / row["merlin_ms"]
           if row["executorch_ms"] is not None and row["merlin_ms"] is not None else np.nan)
          for row in rows]
y = np.arange(len(rows))
axes[1].axvline(1.0, color="black", linestyle="--", linewidth=1)
axes[1].set_yticks(y, model_names)
axes[1].invert_yaxis()
axes[1].set_xlabel(r"Latency ratio $T_{ET}/T_{Merlin}$")
axes[1].set_xlim(0, 1.12)
for y0, value in zip(y, ratios):
    if np.isnan(value):
        axes[1].scatter([0.03], [y0], marker="x", color=COLORS["pending"], s=32)
        axes[1].text(0.07, y0, "not measured", ha="left", va="center",
                     color=COLORS["pending"], fontstyle="italic", fontsize=8.5)
    else:
        axes[1].hlines(y0, 0, value, color=COLORS["merlin"], linewidth=2)
        axes[1].scatter([value], [y0], color=COLORS["merlin"], edgecolor="black",
                        linewidth=0.45, s=42, zorder=3)
        axes[1].text(value + 0.025, y0, f"{value:.2f}",
                     ha="left", va="center", fontsize=8.5)
axes[1].text(0.02, 0.98, "(b)", transform=axes[1].transAxes,
             ha="left", va="top", fontweight="bold")

# (c) TinyLlama's complete-output diagnostic points, grouped by compiler configuration so the
# apparent scaling curve never splices together the best walls from different binaries.
scaling = DATA["tinyllama_core_scaling"]
x = np.arange(len(scaling["cores"]))
config_colors = [COLORS["merlin"], COLORS["xnnpack"]]
config_markers = ["o", "s"]
for offset, config, color, marker in zip((-0.10, 0.0), scaling["merlin_configs"],
                                          config_colors, config_markers):
    best = []
    for x0, core in zip(x, scaling["cores"]):
        sessions = config["sessions_ms"][str(core)]
        jitter = np.linspace(-0.025, 0.025, len(sessions))
        axes[2].scatter(x0 + offset + jitter, sessions,
                        label=config["label"] if x0 == 0 else None,
                        marker=marker, color=color, edgecolor="black",
                        linewidth=0.45, s=34, zorder=3)
        best.append(min(sessions))
    axes[2].plot(x + offset, best, color=color, linewidth=1.15, alpha=0.8, zorder=2)
for x0, value in zip(x, scaling["executorch_ms"]):
    axes[2].scatter(x0 + 0.10, value, marker="D",
                    label="ExecuTorch warm estimate" if x0 == 0 else None,
                    color=COLORS["executorch"], edgecolor="black", linewidth=0.45, s=39, zorder=3)
axes[2].set_xticks(x, [f"{core} core" if core == 1 else f"{core} cores"
                       for core in scaling["cores"]])
axes[2].set_ylabel("Latency (ms; lower is better)")
all_merlin = [v for config in scaling["merlin_configs"]
              for sessions in config["sessions_ms"].values() for v in sessions]
axes[2].set_ylim(0, max(all_merlin) * 1.17)
axes[2].legend(frameon=False, loc="upper right")
axes[2].text(0.02, 0.98, "(c)", transform=axes[2].transAxes,
             ha="left", va="top", fontweight="bold")

for ax in axes:
    ax.grid(axis="x" if ax is axes[1] else "y", color="#dddddd", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

save_figure(fig, HERE / "k1_executorch_status.pdf")
