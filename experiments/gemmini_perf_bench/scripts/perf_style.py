"""Shared matplotlib style for the Gemmini perf-bench figures.

Captures the aesthetic of the reference plots: muted pastel palette on a soft cream ground, consistent
method->colour mapping (golden=steel, baseline=salmon, merlin-gen=GOLD "ours", IREE=sage, native=tan),
value labels printed on the marks, rounded-rect callout badges for the punchline, light grid + no top/
right spines, booktabs-style heatmap tables.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---- palette -------------------------------------------------------------------------------------
CREAM = "#F6F1E7"        # parchment page ground (image copy 7)
INK = "#2B2B2B"
GRID = "#D9D2C4"

# method -> colour (reused identically across every figure). "ours" = gold.
COLOR = {
    "golden":           "#6E93B0",   # steel blue — hand-tuned C reference
    "baseline":         "#D98C84",   # salmon — generated v0
    "merlin_targetgen": "#E6B84C",   # GOLD — ours (v1)
    "iree_dialect":     "#9DB682",   # sage — deprecated IREE dialect
    "merlin_native":    "#C9A86B",   # tan — native ref (ours variant)
}
LABEL = {
    "golden": "golden (C lib)",
    "baseline": "baseline-gen (v0)",
    "merlin_targetgen": "merlin-gen (v1) — ours",
    "iree_dialect": "IREE dialect (depr.)",
    "merlin_native": "merlin-native",
}
# heat scale for tables: low=cool, high=warm (image copy 4/5)
HEAT_GOOD = "#7FB0D6"    # cool blue (good / correct)
HEAT_BAD = "#E08A7D"     # warm red (bad / fail)


def use_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": CREAM,
        "axes.facecolor": CREAM,
        "savefig.facecolor": CREAM,
        "axes.edgecolor": INK,
        "axes.linewidth": 1.1,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "xtick.color": INK, "ytick.color": INK,
        "text.color": INK, "axes.labelcolor": INK,
        "legend.frameon": False,
    })


def badge(ax, x, y, text, color="#E6B84C", fg=INK, fontsize=10):
    """Rounded-rect callout badge (the '13.6x faster' style)."""
    ax.annotate(text, xy=(x, y), xycoords="data", ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=fg, zorder=10,
                bbox=dict(boxstyle="round,pad=0.4", fc=color, ec=INK, lw=1.2, alpha=0.95))


def bar_labels(ax, bars, fmt="{:.0f}", dy=0.0, fontsize=9, rot=0):
    for b in bars:
        h = b.get_height()
        if h is None or h != h:  # nan
            continue
        ax.annotate(fmt.format(h), xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3 + dy), textcoords="offset points",
                    ha="center", va="bottom", fontsize=fontsize, fontweight="bold", rotation=rot)


def card_title(fig, text, sub=None):
    fig.suptitle(text, fontsize=16, fontweight="bold", y=0.98)
    if sub:
        fig.text(0.5, 0.93, sub, ha="center", fontsize=10.5, style="italic", color="#5A5A5A")


def caption(fig, text, y=0.015):
    """Provenance / methods caption band at the figure bottom (reference-image style). State N, oracle
    tier, and measured-vs-estimated here so every figure is self-documenting under review. Pair with
    savefig that reserves bottom space (see save_fig) so it never overlaps axis labels."""
    fig._has_caption = True  # signal to save_fig to reserve bottom margin
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=7.6, style="italic", color="#6b6256",
             wrap=True)


def save_fig(fig, path, dpi=150):
    """tight_layout that RESERVES bottom space when a caption() was added, then save (cream bg)."""
    rect = [0, 0.07, 1, 1] if getattr(fig, "_has_caption", False) else [0, 0, 1, 1]
    try:
        fig.tight_layout(rect=rect)
    except Exception:
        pass
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=CREAM)
    import matplotlib.pyplot as _plt
    _plt.close(fig)


def smooth(y, k=5):
    """Light moving-average smoothing for trajectory lines (odd window k). Returns same length."""
    import numpy as np
    y = np.asarray(y, float)
    if len(y) < 3 or k < 3:
        return y
    k = min(k, len(y) | 1)
    pad = k // 2
    yp = np.pad(y, pad, mode="edge")
    ker = np.ones(k) / k
    return np.convolve(yp, ker, mode="valid")
