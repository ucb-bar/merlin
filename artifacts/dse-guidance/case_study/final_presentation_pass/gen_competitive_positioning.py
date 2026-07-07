#!/usr/bin/env python3
"""House-style competitive-positioning map for OSCAR Merlin (companion to COMPETITIVE_POSITIONING.md).

A qualitative 2-axis map — target-class reach (datacenter -> embedded) x compile scope
(single kernel -> whole model) — placing the 8 source-verified competitors + Merlin. Marker SHAPE
encodes MLIR-class (square) vs not (circle); a GOLD ring marks "emits portable C as a first-class
output". The point is the empty top-right quadrant (embedded + whole-model) that Merlin occupies, and
the four capabilities that quadrant actually requires. NO performance axis on purpose — our only
credited speedup is isolated-kernel (deck p13). Styled via scripts/merlin_plotstyle.py (never re-derive).
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D

REPO = Path("/scratch/agustin/projects/oscar-merlin")
sys.path.insert(0, str(REPO / "scripts"))
from merlin_plotstyle import (use_merlin_style, style_ax, title, suptitle,
                              BG, INK, GOLD, BLUE, NAVY, SLATE, MAUVE, SAGE, SERIF, SANS)

OUT = REPO / "merlin/benchmarks/dse_guidance/case_study/final_presentation_pass"

GRAY = "#A9A296"   # de-emphasised "context" competitors

# name, x(reach), y(scope), color, mlir?, emitsC?, size, (label dx,dy in pts), ha, va
PTS = [
    ("Triton",      1.1, 2.6, GRAY,  True,  False, 360, (0,  20), "center", "bottom"),
    ("XLA",         1.7, 8.4, GRAY,  True,  False, 360, (-6, 20), "center", "bottom"),
    ("Triton-CPU",  3.2, 2.6, GRAY,  True,  False, 360, (0, -24), "center", "top"),
    ("XNNPACK",     4.0, 1.5, GRAY,  False, False, 360, (0, -24), "center", "top"),
    ("OpenBLAS",    5.1, 1.0, GRAY,  False, False, 360, (0, -24), "center", "top"),
    ("IREE",        4.6, 8.5, GRAY,  True,  False, 360, (-10, 18),"center", "bottom"),
    ("EXO",         6.2, 2.1, MAUVE, False, True,  430, (0,  22), "center", "bottom"),
    ("ExecuTorch",  7.6, 8.1, SAGE,  False, False, 430, (4,  20), "center", "bottom"),
    ("OSCAR Merlin",9.1, 9.1, NAVY,  True,  True,  760, (0, -30), "center", "top"),
]


def main():
    use_merlin_style()
    fig, ax = plt.subplots(figsize=(13.5, 8.4))
    style_ax(ax, grid=None)
    ax.set_xlim(0, 11.6); ax.set_ylim(0, 10.6)

    # Merlin's quadrant: embedded + whole-model (faint navy wash + dotted crosshair)
    ax.add_patch(Rectangle((6.6, 6.4), 4.0, 3.6, facecolor=NAVY, alpha=0.055, zorder=0))
    ax.axvline(6.6, ymin=0.0, ymax=1.0, color=INK, ls=(0, (2, 4)), lw=0.9, alpha=0.20, zorder=0)
    ax.axhline(6.4, xmin=0.0, xmax=0.915, color=INK, ls=(0, (2, 4)), lw=0.9, alpha=0.20, zorder=0)
    ax.text(8.6, 6.62, "embedded  x  whole-model", ha="center", va="bottom",
            fontsize=10, style="italic", color="#7d756a", zorder=1)

    # points: gold ring first (behind), then the shaped marker
    for name, x, y, col, mlir, emitsC, s, (dx, dy), ha, va in PTS:
        if emitsC:
            ax.scatter([x], [y], s=s * 2.05, marker=("s" if mlir else "o"),
                       facecolors="none", edgecolors=GOLD, linewidths=2.8, zorder=4)
        ax.scatter([x], [y], s=s, marker=("s" if mlir else "o"),
                   color=col, edgecolor=INK, linewidth=1.5, zorder=5)
        hero = name.startswith("OSCAR")
        ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va, fontsize=12.5 if hero else 11,
                    fontweight="bold" if hero else "normal",
                    color=INK, zorder=6)

    # axis semantics (Inter labels — arrows OK in Inter, not in the serif title)
    ax.set_xlabel("target-class reach     datacenter GPU  →  server CPU  →  edge  →  embedded / bare-metal / RTOS",
                  fontsize=12)
    ax.set_ylabel("compile scope     single kernel  →  whole-model compile + run", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    title(ax, "Where OSCAR Merlin sits — and what its quadrant requires", fs=18, pad=14)

    # legend: marker grammar (top-left, out of the data)
    leg = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=SLATE, markeredgecolor=INK,
               markersize=12, label="MLIR-class"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SLATE, markeredgecolor=INK,
               markersize=12, label="not MLIR-based"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=GOLD,
               markeredgewidth=2.6, markersize=15, label="emits portable C"),
    ]
    ax.legend(handles=leg, loc="upper left", fontsize=10.5, frameon=True,
              facecolor="white", edgecolor="#d9cfc0", handletextpad=0.6,
              borderpad=0.7, labelspacing=0.7)

    # callout: the four things the quadrant needs (rounded card, bottom-right, clear of points)
    cx, cy, cw, ch = 6.95, 0.55, 4.35, 3.35
    card = FancyBboxPatch((cx, cy), cw, ch, boxstyle="round,pad=0.10,rounding_size=0.18",
                          facecolor="white", edgecolor=NAVY, linewidth=1.6, zorder=7,
                          mutation_aspect=0.62)
    ax.add_patch(card)
    ax.text(cx + 0.30, cy + ch - 0.42, "Merlin's quadrant needs all four:",
            fontsize=11.5, fontweight="bold", color=NAVY, fontfamily=SANS, zorder=8)
    bullets = [
        "C as the default output (not a port)",
        "RISC-V / RVV + Gemmini bringup",
        "contract + out-of-tree + L0–L5 cert ladder",
        "AI proposes, deterministic gate disposes",
    ]
    for i, b in enumerate(bullets):
        yb = cy + ch - 0.90 - i * 0.55
        ax.text(cx + 0.34, yb, "•", fontsize=12, color=GOLD, fontweight="bold", zorder=8)
        ax.text(cx + 0.64, yb, b, fontsize=10.4, color=INK, va="baseline", zorder=8)

    # honest footnote (no perf axis; fork caveat) — below the axes
    fig.text(0.5, 0.012,
             "Qualitative map from on-disk source (paths + du -sh in the companion doc). No performance "
             "axis: our only credited speedup is isolated-kernel (deck p13).  "
             "The on-disk IREE is a UCB-BAR fork; its Zephyr/Gemmini bits are ours, not stock IREE.",
             ha="center", va="bottom", fontsize=8.6, color="#7d756a")

    fig.tight_layout(rect=(0, 0.035, 1, 1))
    png = OUT / "fig_competitive_positioning.png"
    svg = OUT / "fig_competitive_positioning.svg"
    fig.savefig(png, bbox_inches="tight", dpi=300, facecolor=BG)
    fig.savefig(svg, bbox_inches="tight", facecolor=BG)
    print(f"wrote {png}\nwrote {svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
