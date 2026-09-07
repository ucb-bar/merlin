#!/usr/bin/env python3
"""Diagnostic eight-core LSTMNetVIT W8A8 instrumented-IR intervals."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from paper_plot_style import COLORS, save_figure


HERE = Path(__file__).resolve().parent
PROFILE = HERE / "k1_lstmnetvit_8core_breakdown.json"
data = json.loads(PROFILE.read_text(encoding="utf-8"))
rows = [row for row in data["by_category"] if row["ms"] >= 1.0]

labels = {
    "unclassified_generic": "other generic loops",
    "quantize_scale_search": "activation scale search",
    "elementwise": "broadcast / elementwise",
    "quantize_requant": "requantization",
    "layout_copy": "transpose / layout copy",
    "contraction": "matmul / batch matmul",
    "unclassified:linalg.add": "adds",
    "fill_init": "fills",
    "alloc": "tensor allocation",
}

fig, ax = plt.subplots(figsize=(7.2, 4.2))
y = list(range(len(rows)))
colors = [COLORS["merlin"] if row["category"] != "contraction"
          else COLORS["executorch"] for row in rows]
bars = ax.barh(y, [row["ms"] for row in rows], color=colors,
               edgecolor="black", linewidth=0.45)
ax.set_yticks(y, [labels.get(row["category"], row["category"]) for row in rows])
ax.invert_yaxis()
ax.set_xlabel("Attributed operator time per inference (ms; 8 cores)")
ax.grid(axis="x", color="#dddddd", linewidth=0.6)
ax.set_axisbelow(True)
for bar, row in zip(bars, rows):
    ax.text(bar.get_width() + 0.35, bar.get_y() + bar.get_height() / 2,
            f"{row['ms']:.1f} ms  ({100 * row['share']:.1f}%)",
            va="center", fontsize=8.2)
ax.set_xlim(0, max(row["ms"] for row in rows) * 1.35)
ax.text(0.5, 1.01,
        "Instrumented-IR only; release-IR intersection found 39 foldable named-op broadcasts (+2.7%)",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, color="#555555")
save_figure(fig, HERE / "k1_lstmnetvit_8core_breakdown.pdf")
