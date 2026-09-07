#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import COLORS, save_figure


HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "k1_executorch_status_20260906.json").read_text(encoding="utf-8"))


fig, grid = plt.subplots(2, 2, figsize=(11.0, 6.4),
                         gridspec_kw={"wspace": 0.33, "hspace": 0.42})
axes = grid.ravel()

# (a) The exact historical slide result, kept visually separate because it is FP32 kernel routing.
historical = DATA["historical_fp32_bitvla"]
names = list(historical["latency_ms"])
values = [historical["latency_ms"][name] for name in names]
colors = [COLORS["merlin"], COLORS["xnnpack"], COLORS["openblas"]]
xpos = np.arange(len(names))
for x0, name, best, color in zip(xpos, names, values, colors):
    runs = historical["launches_ms"][name]
    jitter = np.linspace(-0.035, 0.035, len(runs))
    axes[0].scatter(x0 + jitter, runs, s=33, color=color, edgecolor="black",
                    linewidth=0.45, zorder=3)
    axes[0].hlines(best, x0 - 0.25, x0 + 0.25, color="black", linewidth=1.3, zorder=4)
    axes[0].text(x0, max(runs) + 2.5, f"best {best:.1f}", ha="center", va="bottom", fontsize=8.2)
axes[0].set_ylabel("Latency (ms; lower is better)")
axes[0].set_xticks(xpos, ["Merlin", "XNNPACK\nkernels", "OpenBLAS\nkernels"])
axes[0].set_ylim(0, max(values) * 1.22)
axes[0].text(0.02, 0.98, "(a)", transform=axes[0].transAxes,
             ha="left", va="top", fontweight="bold")

# (b) Current INT8 status at both requested core counts. These are diagnostic best-valid cells,
# not paired confidence intervals, so points are deliberately not joined into trend lines.
rows = DATA["int8_one_core"]["models"]
rows8 = DATA["int8_eight_core"]["models"]
model_names = [row["model"] for row in rows]
ratios = [(row["executorch_ms"] / (row["merlin_ms"] or row.get("merlin_lower_bound_ms"))
           if row["executorch_ms"] is not None
           and (row["merlin_ms"] is not None or row.get("merlin_lower_bound_ms")) else np.nan)
          for row in rows]
ratios8 = [(row["executorch_ms"] / (row["merlin_ms"] or row.get("merlin_lower_bound_ms"))
            if row["executorch_ms"] is not None
            and (row["merlin_ms"] is not None or row.get("merlin_lower_bound_ms")) else np.nan)
           for row in rows8]
y = np.arange(len(rows))
axes[1].axvline(1.0, color="black", linestyle="--", linewidth=1)
axes[1].set_yticks(y, model_names)
axes[1].invert_yaxis()
axes[1].set_xlabel(r"Latency ratio $T_{ET}/T_{Merlin}$")
axes[1].set_xlim(0, 1.12)
for label, values, source_rows, offset, marker, color in (
        ("1 core", ratios, rows, -0.10, "o", COLORS["merlin"]),
        ("8 cores", ratios8, rows8, 0.10, "s", COLORS["xnnpack"])):
    for y0, value in zip(y, values):
        if not np.isnan(value):
            is_bound = source_rows[y0].get("merlin_lower_bound_ms") is not None
            point_marker = "<" if is_bound else marker
            axes[1].scatter([value], [y0 + offset], marker=point_marker, color=color,
                            edgecolor="black", linewidth=0.45, s=38, zorder=3,
                            label=label if y0 == 0 else None)
            value_text = f"<{value:.3f}" if is_bound else f"{value:.2f}"
            axes[1].text(value + 0.025, y0 + offset, value_text,
                         ha="left", va="center", fontsize=8.0)
if np.isnan(ratios8[-1]):
    axes[1].scatter([0.03], [y[-1] + 0.10], marker="x", color=COLORS["pending"], s=32)
    axes[1].text(0.07, y[-1] + 0.10, "8-core not measured", ha="left", va="center",
                 color=COLORS["pending"], fontstyle="italic", fontsize=8.5)
axes[1].legend(frameon=False, loc="upper right", fontsize=8.0)
axes[1].text(0.02, 0.98, "(b)", transform=axes[1].transAxes,
             ha="left", va="top", fontweight="bold")

# (c) TinyLlama's complete-output diagnostic points, grouped by compiler configuration so the
# apparent scaling curve never splices together the best walls from different binaries.
scaling = DATA["tinyllama_core_scaling"]
x = np.arange(len(scaling["cores"]))
config_colors = [COLORS["merlin"], COLORS["pending"], COLORS["xnnpack"]]
config_markers = ["o", "^", "s"]
for offset, config, color, marker in zip((-0.14, 0.0, 0.14), scaling["merlin_configs"],
                                          config_colors, config_markers):
    medians = []
    for x0, core in zip(x, scaling["cores"]):
        sessions = config["sessions_ms"][str(core)]
        jitter = np.linspace(-0.025, 0.025, len(sessions))
        axes[2].scatter(x0 + offset + jitter, sessions,
                        label=config["label"] if x0 == 0 else None,
                        marker=marker, color=color, edgecolor="black",
                        linewidth=0.45, s=34, zorder=3)
        medians.append(float(np.median(sessions)))
    axes[2].plot(x + offset, medians, color=color, linewidth=1.15, alpha=0.8, zorder=2)
for x0, value in zip(x, scaling["executorch_ms"]):
    axes[2].scatter(x0 + 0.25, value, marker="D",
                    label="ExecuTorch estimate" if x0 == 0 else None,
                    color=COLORS["executorch"], edgecolor="black", linewidth=0.45, s=39, zorder=3)
axes[2].set_xticks(x, [f"{core} core" if core == 1 else f"{core} cores"
                       for core in scaling["cores"]])
axes[2].set_ylabel("Latency (ms; lower is better)")
all_merlin = [v for config in scaling["merlin_configs"]
              for sessions in config["sessions_ms"].values() for v in sessions]
axes[2].set_ylim(0, max(all_merlin) * 1.42)
axes[2].legend(frameon=False, loc="upper right", fontsize=7.5)
axes[2].text(0.02, 0.98, "(c)", transform=axes[2].transAxes,
             ha="left", va="top", fontweight="bold")

# (d) LSTMNetVIT: the new grouped-direct result is separated from the previous im2col+coarsening
# arm. Log scale keeps both improvements and the remaining ExecuTorch gap legible.
lstm = DATA["lstmnetvit_core_scaling"]
x = np.arange(len(lstm["cores"]))
direct_medians = []
im2col_medians = []
for x0, core in zip(x, lstm["cores"]):
    direct = lstm["grouped_direct_sessions_ms"][str(core)]
    im2col = lstm["merlin_coarsened_sessions_ms"][str(core)]
    direct_jitter = np.linspace(-0.025, 0.025, len(direct))
    axes[3].scatter(x0 - 0.13 + direct_jitter, direct, marker="o", color=COLORS["merlin"],
                    edgecolor="black", linewidth=0.45, s=34, zorder=3,
                    label="Merlin, grouped direct + AOT + fold" if x0 == 0 else None)
    im2col_jitter = np.linspace(-0.025, 0.025, len(im2col))
    axes[3].scatter(x0 + im2col_jitter, im2col, marker="s", color=COLORS["xnnpack"],
                    edgecolor="black", linewidth=0.45, s=27, zorder=3,
                    label="Merlin, grouped im2col" if x0 == 0 else None)
    direct_medians.append(float(np.median(direct)))
    im2col_medians.append(float(np.median(im2col)))
axes[3].plot(x - 0.13, direct_medians, color=COLORS["merlin"], linewidth=1.2)
axes[3].plot(x, im2col_medians, color=COLORS["xnnpack"], linewidth=1.0)
axes[3].plot(x + 0.13, lstm["executorch_ms"], marker="D", color=COLORS["executorch"],
             markeredgecolor="black", markeredgewidth=0.45, linewidth=1.2,
             label="ExecuTorch")
axes[3].set_xticks(x, [f"{core} core" if core == 1 else f"{core} cores"
                       for core in lstm["cores"]])
axes[3].set_yscale("log")
axes[3].set_ylabel("Latency (ms, log scale)")
axes[3].legend(frameon=False, loc="upper right", fontsize=7.5)
axes[3].text(0.02, 1.02, "(d)", transform=axes[3].transAxes,
             ha="left", va="bottom", fontweight="bold")

for ax in axes:
    ax.grid(axis="x" if ax is axes[1] else "y", color="#dddddd", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

save_figure(fig, HERE / "k1_executorch_status.pdf")
