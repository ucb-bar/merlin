#!/usr/bin/env python3
"""Shared visual style — muted-pastel "ML-paper" aesthetic (Physical Intelligence / Cosmos look).

Signature elements distilled from the reference figures:
  - soft pastel fills, each paired with a DARKER matching edge (the defining trait)
  - value labels printed directly on bars / points
  - rounded callout boxes with thin leader lines for key insights
  - light shaded highlight regions; clean white axes, no top/right spines, faint grid
  - optional hatch patterns + cream rounded-card panels
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import FancyBboxPatch, Rectangle

# (fill, edge) pairs — pastel fill + darker outline, in a pleasant default series order
PAIRS = [
    ("#7ba7c7", "#3f6f8f"),   # dusty blue
    ("#a9c08a", "#6f8f4e"),   # sage green
    ("#ecc24e", "#c2962f"),   # mustard
    ("#dd9089", "#b5564e"),   # salmon
    ("#b69ac8", "#8763a8"),   # lavender
    ("#7cc0b3", "#3f8f7f"),   # teal
    ("#e0a36a", "#b5712f"),   # amber
    ("#9aa0a8", "#5f656d"),   # slate grey
]
FILLS = [p[0] for p in PAIRS]
EDGES = [p[1] for p in PAIRS]
HATCHES = [None, "////", "....", "xxxx", "\\\\\\\\", "++"]

# named roles mapped onto the palette (kept stable across figures)
BLUE, GREEN, MUSTARD, SALMON, LAVENDER, TEAL, AMBER, GREY = (p[0] for p in PAIRS)
BLUE_E, GREEN_E, MUSTARD_E, SALMON_E, LAVENDER_E, TEAL_E, AMBER_E, GREY_E = (p[1] for p in PAIRS)

CREAM = "#f4efe2"          # rounded-card background
HIGHLIGHT = "#f3dede"      # light shaded "selected zone"
INK = "#2b2b2b"            # near-black for the bold outcome line / text


def apply_theme():
    """Global rcParams for the muted-paper look."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#9a9a9a",
        "axes.linewidth": 0.9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.labelcolor": "#333333",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e9e9e9",
        "grid.linewidth": 0.8,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "font.size": 10,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#cccccc",
        "legend.fontsize": 8.5,
    })


def style_axes(ax):
    """Drop the top/right spines and soften the rest (call after plotting)."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9a9a9a")
    ax.tick_params(labelsize=8)


def solid_shadow(ax, patches, dx=3.5, dy=-3.5, color="#2b2b2b", alpha=0.22):
    """Hard (non-blurred) offset drop-shadow behind each patch — the '3D sticker' look from the
    reference bar charts. Draws an offset duplicate Rectangle, in POINTS, just below the patch."""
    trans = mtransforms.offset_copy(ax.transData, fig=ax.figure, x=dx, y=dy, units="points")
    for b in patches:
        sh = Rectangle((b.get_x(), b.get_y()), b.get_width(), b.get_height(),
                       transform=trans, facecolor=color, edgecolor="none",
                       alpha=alpha, zorder=b.get_zorder() - 0.1)
        ax.add_patch(sh)


def bar(ax, *args, ci=0, hatch=None, shadow=False, edge="#2b2b2b", lw=1.3, **kw):
    """ax.bar with a darker matching edge from the palette (ci selects the colour pair).
    shadow=True adds the solid 3D offset shadow; edge defaults to ink for the bold-outline look."""
    fill, pedge = PAIRS[ci % len(PAIRS)]
    kw.setdefault("color", fill)
    kw.setdefault("edgecolor", edge if edge else pedge)
    kw.setdefault("linewidth", lw)
    if hatch:
        kw["hatch"] = hatch
    bars = ax.bar(*args, **kw)
    if shadow:
        solid_shadow(ax, bars)
    return bars


def barh(ax, *args, ci=0, shadow=True, edge="#2b2b2b", lw=1.4, **kw):
    """Horizontal bar with bold ink outline + solid shadow (the image-copy-7 card look)."""
    fill, pedge = PAIRS[ci % len(PAIRS)]
    kw.setdefault("color", fill)
    kw.setdefault("edgecolor", edge if edge else pedge)
    kw.setdefault("linewidth", lw)
    bars = ax.barh(*args, **kw)
    if shadow:
        solid_shadow(ax, bars)
    return bars


def label_bars(ax, bars, fmt="{:.0f}", fontsize=8, color="#333333", dy=0):
    """Print the value directly above each bar (paper style)."""
    for b in bars:
        h = b.get_height()
        if h is None:
            continue
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=fontsize, color=color,
                    xytext=(0, 1 + dy), textcoords="offset points")


def callout(ax, text, xy, xytext, *, fc="white", ec="#b9b9b9", fontsize=8.5,
            color="#333333", weight="normal", arrow=True):
    """Rounded callout box with a thin leader line pointing at xy (data coords)."""
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=fontsize, color=color, weight=weight,
                ha="center", va="center", zorder=10,
                bbox=dict(boxstyle="round,pad=0.35", fc=fc, ec=ec, lw=1.0),
                arrowprops=(dict(arrowstyle="-", color=ec, lw=1.0,
                                 connectionstyle="arc3,rad=0.0") if arrow else None))


def highlight_span(ax, x0, x1, label=None):
    """Light shaded 'selected zone' (like the paper's grey/pink highlight region)."""
    ax.axvspan(x0, x1, color=HIGHLIGHT, alpha=0.6, lw=0, zorder=0)
    if label:
        ax.text((x0 + x1) / 2, 0.97, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color="#b5564e", style="italic")


def card(fig, rect, facecolor=CREAM, shadow=True):
    """Rounded-corner card panel (rect = [x, y, w, h] in fig coords), with a solid offset shadow —
    the image-copy-7 'card with hard shadow' look."""
    x, y, w, h = rect
    box = "round,pad=0.012,rounding_size=0.02"
    if shadow:
        sh = mtransforms.offset_copy(fig.transFigure, fig=fig, x=6, y=-6, units="points")
        fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle=box, transform=sh,
                                          fc="#000000", ec="none", alpha=0.16, zorder=-3))
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, boxstyle=box, transform=fig.transFigure,
        fc=facecolor, ec="#d8cfb8", lw=1.2, zorder=-1))
