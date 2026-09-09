#!/usr/bin/env python3
"""Generate one aligned, slide-style overview across every requested K1 model."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

from merlin.plotting.merlin_plotstyle import (
    BLUE as ACCENT,
    GOLD,
    INK,
    MAUVE,
    NAVY,
    SAGE,
    SANS,
    SERIF,
    SLATE,
    use_merlin_style,
)


HERE = Path(__file__).resolve().parent
MANIFEST = json.loads(
    (HERE / "k1_centralized_overview_20260907.json").read_text(encoding="utf-8")
)


def load_source(key: str) -> dict:
    return json.loads((HERE / MANIFEST["sources"][key]).read_text(encoding="utf-8"))


INT8 = load_source("int8_comparison")
VLA = load_source("vla_comparison")
STATUS = load_source("core_scaling")

# Repository-wide series identities.  The white canvas below is an explicit presentation override
# requested for this figure; it must not silently create a second palette or typography system.
MERLIN = NAVY
EXECUTORCH = MAUVE
XNNPACK = SLATE
LAYOUT = GOLD
CONTRACTION = NAVY
ELEMENTWISE = MAUVE
REDUCTION = SAGE
OTHER = "#B9B3AC"
WARNING = "#D55E00"
GRID = "#D9D6D1"
MUTED = "#88827B"
WHITE = "#FFFFFF"
CARD = WHITE

use_merlin_style()
plt.rcParams.update({
    # White is intentional here even though the reusable house default is warm cream.
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "savefig.facecolor": WHITE,
    "font.family": SANS,
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def by_model(rows: list[dict], name: str) -> dict | None:
    folded = name.lower().replace("_", "")
    return next(
        (row for row in rows if row["model"].lower().replace("_", "") == folded), None)


def decorate_panel(ax: plt.Axes, number: int, heading: str, subtitle: str) -> None:
    bbox = ax.get_position()
    card = FancyBboxPatch(
        (bbox.x0 - 0.012, bbox.y0 - 0.035), bbox.width + 0.024, bbox.height + 0.085,
        boxstyle="round,pad=0.004,rounding_size=0.012", transform=ax.figure.transFigure,
        facecolor=CARD, edgecolor="#E9E1D8", linewidth=0.8, zorder=-5,
        path_effects=[pe.SimplePatchShadow(offset=(5, -5), shadow_rgbFace="#9C8E82", alpha=0.18),
                      pe.Normal()],
    )
    ax.figure.patches.append(card)
    ax.set_facecolor("none")
    ax.text(-0.075, 1.095, str(number), transform=ax.transAxes, ha="center", va="center",
            color="white", fontsize=13, fontweight="bold",
            bbox={"boxstyle": "circle,pad=0.33", "facecolor": GOLD, "edgecolor": "none"},
            clip_on=False)
    ax.text(0.0, 1.105, heading, transform=ax.transAxes, ha="left", va="center",
            color=INK, fontsize=13.5, fontfamily=SERIF, clip_on=False)
    ax.text(0.0, 1.045, f"\N{BLACK RIGHT-POINTING TRIANGLE} {subtitle}",
            transform=ax.transAxes, ha="left", va="center", color=ACCENT,
            fontsize=8.8, fontstyle="italic", clip_on=False)


def add_group_bands(ax: plt.Axes) -> None:
    ax.axhspan(2.5, 6.5, color="#F8EDE3", alpha=0.44, zorder=-4)
    ax.axhspan(-0.5, 2.5, color="#EAF4EF", alpha=0.48, zorder=-4)


def format_ms(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.2f}s"
    if value >= 100:
        return f"{value:.0f}ms"
    return f"{value:.2f}ms"


def validate() -> None:
    assert INT8["status"] == "diagnostic"
    assert "not a paired certification set" in INT8["comparability"]
    assert len(MANIFEST["model_order"]) == 7
    assert len(set(MANIFEST["model_order"])) == 7
    for row in VLA["models"]:
        assert row["hybrid_status"] == "measured"
        assert row["hybrid_ms"] is not None and row["hybrid_cosine"] is not None
        assert abs(sum(row["categories_ms"].values()) - row["profile_wall_ms"]) < 1e-6


validate()
names = MANIFEST["model_order"]
rows: list[dict] = []
for name in names:
    int8 = by_model(INT8["models"], name)
    vla = by_model(VLA["models"], name)
    if int8 is not None:
        rows.append({
            "model": name,
            "tier": "INT8",
            "merlin_ms": int8["merlin_ms"],
            "merlin_kind": int8["merlin_kind"],
            "peer_ms": int8["executorch_ms"],
            "peer": "ExecuTorch",
            "profile": None,
        })
    elif vla is not None:
        rows.append({
            "model": name,
            "tier": "FP32",
            "merlin_ms": vla["warm_ms"],
            "merlin_kind": "measured",
            "peer_ms": vla["hybrid_ms"],
            "peer": "Merlin + XNNPACK GEMM",
            "profile": vla,
        })
    else:
        raise AssertionError(f"missing requested model: {name}")

# Reverse positions so manifest order reads top-to-bottom.
y = np.arange(len(rows))[::-1]
fig, axes = plt.subplots(1, 4, figsize=(18.0, 9.0), sharey=True,
                         gridspec_kw={"width_ratios": [2.25, 1.25, 1.85, 1.25]})
fig.subplots_adjust(left=0.12, right=0.975, top=0.86, bottom=0.24, wspace=0.17)

for ax in axes:
    add_group_bands(ax)
    ax.set_ylim(-0.55, 6.55)
    ax.tick_params(axis="y", length=0)

# 1. Absolute warm latency, one row for every model.
ax = axes[0]
for y0, row in zip(y, rows):
    peer_color = EXECUTORCH if row["peer"] == "ExecuTorch" else XNNPACK
    lo, hi = sorted((row["merlin_ms"], row["peer_ms"]))
    lower_bound = row["merlin_kind"] == "lower_bound"
    ax.plot([lo, hi], [y0, y0], color="#A8A29B", linewidth=2.0,
            linestyle="--" if lower_bound else "-", zorder=1)
    ax.scatter(row["merlin_ms"], y0, marker=">" if lower_bound else "s", s=76, color=MERLIN,
               edgecolor="white", linewidth=0.8, zorder=3)
    ax.scatter(row["peer_ms"], y0, marker="D" if row["peer"] == "ExecuTorch" else "o",
               s=65, color=peer_color, edgecolor="white", linewidth=0.8, zorder=3)
    merlin_label = ((">" if lower_bound else "") + format_ms(row["merlin_ms"])
                    + ("\N{DOUBLE DAGGER}" if lower_bound else ""))
    ax.text(row["merlin_ms"] * 1.12, y0 + 0.16, merlin_label,
            color=MERLIN, fontsize=8, fontweight="bold", ha="left", va="bottom")
    ax.text(row["peer_ms"] * 1.12, y0 - 0.16, format_ms(row["peer_ms"]),
            color=peer_color, fontsize=8, fontweight="bold", ha="left", va="top")
    if lower_bound:
        ax.annotate("", xy=(220000, y0), xytext=(row["merlin_ms"], y0),
                    arrowprops={"arrowstyle": "->", "color": MERLIN, "linewidth": 1.3})
    missing_label = "XNN qd8 blocked" if row["tier"] == "INT8" else "ET unmatched"
    missing_color = WARNING if row["tier"] == "INT8" else MUTED
    ax.scatter(350000, y0, marker="x", s=42, color=missing_color, linewidth=1.8, zorder=3)
    ax.text(430000, y0, missing_label, ha="left", va="center", color=missing_color,
            fontsize=7.2, fontstyle="italic")
ax.set_xscale("log")
ax.set_xlim(1.0, 2000000)
ax.set_xlabel("Warm latency (ms, log scale)")
ax.set_yticks(y, [f"{row['model']}   [{row['tier']}]" for row in rows])
ax.grid(axis="x", color=GRID, linewidth=0.7)
ax.set_axisbelow(True)
decorate_panel(ax, 1, "Warm latency", "same seven model rows; logarithmic milliseconds")

# 2. Within-row peer gap for all seven models.
ax = axes[1]
for y0, row in zip(y, rows):
    ratio = row["merlin_ms"] / row["peer_ms"]
    color = EXECUTORCH if row["peer"] == "ExecuTorch" else XNNPACK
    hatch = "//" if row["merlin_kind"] == "lower_bound" else None
    ax.barh(y0, max(ratio - 1.0, 0.015), left=1.0, height=0.52, color=color,
            edgecolor="#574F47", linewidth=0.6, hatch=hatch)
    prefix = ">" if row["merlin_kind"] == "lower_bound" else ""
    suffix = "\N{DAGGER}" if row["model"] == "SmallLLaMA" else (
        "\N{DOUBLE DAGGER}" if row["merlin_kind"] == "lower_bound" else "")
    ax.text(ratio * 1.07, y0, f"{prefix}{ratio:.2f}\N{MULTIPLICATION SIGN}{suffix}",
            ha="left", va="center", color=INK, fontsize=9, fontweight="bold")
ax.axvline(1.0, color=ACCENT, linestyle=(0, (4, 3)), linewidth=1.6)
ax.set_xscale("log")
ax.set_xlim(0.85, 30)
ax.set_xlabel("Merlin / available peer")
ax.grid(axis="x", color=GRID, linewidth=0.7)
ax.set_axisbelow(True)
decorate_panel(ax, 2, "Gap to peer", "within-row ratio; lower is better")

# 3. Current operator profiles, keeping unavailable rows visible.
ax = axes[2]
categories = ["layout_copy", "contraction", "elementwise", "reduction_softmax", "other"]
category_labels = ["layout / copy", "contraction", "elementwise", "reduce / softmax", "other"]
colors = [LAYOUT, CONTRACTION, ELEMENTWISE, REDUCTION, OTHER]
left = np.zeros(len(rows))
for category, color in zip(categories, colors):
    values = np.zeros(len(rows))
    for index, row in enumerate(rows):
        if row["profile"] is not None:
            profile = row["profile"]
            values[index] = 100 * profile["categories_ms"][category] / profile["profile_wall_ms"]
    bars = ax.barh(y, values, left=left, height=0.54, color=color, edgecolor="white",
                   linewidth=0.7)
    for index, (bar, value) in enumerate(zip(bars, values)):
        if value >= 14:
            text_color = INK if color == LAYOUT else "white"
            ax.text(left[index] + value / 2, bar.get_y() + bar.get_height() / 2,
                    f"{value:.0f}%", ha="center", va="center", fontsize=8.5,
                    color=text_color, fontweight="bold")
    left += values
for y0, row in zip(y, rows):
    if row["profile"] is None:
        ax.plot([2, 98], [y0, y0], color="#C9C4BE", linewidth=1.0, linestyle=(0, (2, 3)))
        ax.text(50, y0, "current matched profile not measured", ha="center", va="center",
                color=MUTED, fontsize=7.5, fontstyle="italic",
                bbox={"facecolor": CARD, "edgecolor": "none", "pad": 1.5})
ax.set_xlim(0, 100)
ax.set_xlabel("Share of profiled wall time (%)")
ax.grid(axis="x", color=GRID, linewidth=0.7)
ax.set_axisbelow(True)
decorate_panel(ax, 3, "Operator mix", "same rows; unavailable cells stay visible")

# 4. Matched one-to-eight-core scaling where it exists.
ax = axes[3]
one_rows = STATUS["int8_one_core"]["models"]
eight_rows = STATUS["int8_eight_core"]["models"]
for y0, row in zip(y, rows):
    one = by_model(one_rows, row["model"])
    eight = by_model(eight_rows, row["model"])
    if (one is None or eight is None or one.get("merlin_ms") is None
            or eight.get("merlin_ms") is None or one.get("executorch_ms") is None
            or eight.get("executorch_ms") is None):
        ax.text(50, y0, "not measured", ha="center", va="center", color=MUTED,
                fontsize=8, fontstyle="italic")
        continue
    merlin_speedup = one["merlin_ms"] / eight["merlin_ms"]
    peer_speedup = one["executorch_ms"] / eight["executorch_ms"]
    merlin_eff = 100 * merlin_speedup / 8
    peer_eff = 100 * peer_speedup / 8
    ax.plot([merlin_eff, peer_eff], [y0, y0], color="#A8A29B", linewidth=2)
    ax.scatter(merlin_eff, y0, marker="s", s=58, color=MERLIN, zorder=3)
    ax.scatter(peer_eff, y0, marker="D", s=54, color=EXECUTORCH, zorder=3)
    ax.text(merlin_eff, y0 + 0.20, f"{merlin_speedup:.2f}\N{MULTIPLICATION SIGN}",
            ha="center", va="bottom", fontsize=8, color=MERLIN, fontweight="bold")
    ax.text(peer_eff, y0 - 0.20, f"{peer_speedup:.2f}\N{MULTIPLICATION SIGN}",
            ha="center", va="top", fontsize=8, color=EXECUTORCH, fontweight="bold")
ax.axvline(100, color=REDUCTION, linestyle=(0, (4, 3)), linewidth=1.5)
ax.set_xlim(0, 108)
ax.set_xlabel("8-core efficiency (%)")
ax.grid(axis="x", color=GRID, linewidth=0.7)
ax.set_axisbelow(True)
decorate_panel(ax, 4, "CPU scaling", "same rows; one-to-eight cores")

legend_handles = [
    Line2D([], [], marker="s", linestyle="", color=MERLIN, label="Merlin generated"),
    Line2D([], [], marker="D", linestyle="", color=EXECUTORCH, label="ExecuTorch"),
    Line2D([], [], marker="o", linestyle="", color=XNNPACK, label="Merlin + XNNPACK"),
    Patch(facecolor=LAYOUT, label="layout / copy"),
    Patch(facecolor=CONTRACTION, label="contraction"),
    Patch(facecolor=ELEMENTWISE, label="elementwise"),
    Patch(facecolor=REDUCTION, label="reduce / softmax"),
    Patch(facecolor=OTHER, label="other"),
]
fig.legend(handles=legend_handles, frameon=False, ncol=8, loc="lower center",
           bbox_to_anchor=(0.5, 0.145), columnspacing=1.2, handlelength=1.5)
fig.text(0.5, 0.087,
         "Every model occupies the same row and the same measurement slots",
         ha="center", va="center", color=INK, fontsize=13, fontweight="bold",
         bbox={"boxstyle": "round,pad=0.55", "facecolor": "#F4EFE8", "edgecolor": "#DED4C8"})
fig.text(0.5, 0.015, MANIFEST["footnote"], ha="center", va="bottom", color="#625D57",
         fontsize=8.8, linespacing=1.35)

pdf = HERE / "k1_centralized_overview.pdf"
png = HERE / "k1_centralized_overview.png"
fig.savefig(pdf, bbox_inches="tight", pad_inches=0.18)
fig.savefig(png, dpi=240, bbox_inches="tight", pad_inches=0.18)
print(f"saved {pdf}")
print(f"saved {png}")
