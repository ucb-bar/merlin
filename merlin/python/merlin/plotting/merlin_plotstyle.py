"""Merlin house plotting style — the single source of truth for every figure in this repo.

Import this at the top of ANY plotting script and call ``use_merlin_style()`` once:

    import matplotlib
    matplotlib.use("Agg")
    from merlin_plotstyle import *        # palette, SERIES, helpers
    use_merlin_style()                    # fonts + rcParams (warm cream canvas, ash ink)

    fig, ax = plt.subplots()
    style_ax(ax)                          # cream bg, ink spines, dotted value grid
    vbars(ax, x, heights, NAVY)           # solid bar + hard 3-D block shadow
    title(ax, "My panel")                 # DM Serif Display, left-aligned
    suptitle(fig, "My figure")

House rules (see docs/plot_style.md for the full guide):
  - background  #FDF7EF  on every axes + figure (never white)
  - ink / edges / shadow / body text  #2E2D2C  (ash black)
  - emphasis text  #AB9A89 (gold, bold)  or  #333351 (blue)
  - bar palette: 0F3759 · 333351 · 8B93A6 · 815E5E · 7D886C  (one colour per series)
  - SOLID fills — colour carries series identity.  Hatch ONLY when it means something
    (a conditional / not-directly-comparable bar), never as decoration.
  - bars are 3-D blocks: a hard-edged solid-ink offset behind each bar (use vbars/hbars).
  - NO descriptive caption paragraph baked into the figure — the speaker narrates it.
  - fonts: DM Serif Display (titles) + Inter (everything else).
  - bar value axes need not start at 0 — pick limits from the data range so differences read.
  - never let text overlap a bar/line or fall outside the axes.
"""
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import font_manager as fm
from matplotlib.transforms import offset_copy
from matplotlib.patches import Rectangle
import numpy as np

# ----------------------------------------------------------------- house palette
BG    = "#FDF7EF"   # cream background (all plots)
INK   = "#2E2D2C"   # ash black — text, edges, the 3-D block shadow
GOLD  = "#AB9A89"   # california gold — emphasis text (bold)
BLUE  = "#333351"   # indigo — emphasis text / bars
NAVY  = "#0F3759"   # deep navy — hero bars (ours)
SLATE = "#8B93A6"   # gray-blue — bars
MAUVE = "#815E5E"   # mauve — bars (baseline / killed)
SAGE  = "#7D886C"   # sage — bars (OpenBLAS)

# consistent series identity across every figure — solid fills (colour carries identity)
SERIES = {
    "baseline": dict(c=MAUVE, h="", lab="baseline (hand_v0)"),
    "ours":     dict(c=NAVY,  h="", lab="ours (compiler)"),
    "xnnpack":  dict(c=SLATE, h="", lab="XNNPACK"),
    "openblas": dict(c=SAGE,  h="", lab="OpenBLAS"),
    "ceiling":  dict(c=BLUE,  h="", lab="hand ceiling"),
}

SERIF = "DM Serif Display"
SANS = "Inter"

# soft shadow for rounded diagram cards (flow-charts, CCA cards) — NOT used on bars
SHADOW = pe.withSimplePatchShadow(offset=(3.0, -3.0),
                                  shadow_rgbFace=(0.18, 0.178, 0.173), alpha=0.26, rho=1.0)


def use_merlin_style():
    """Register the house fonts and apply the rcParams. Call once per script."""
    for fp in (glob.glob("/usr/share/fonts/opentype/inter/Inter-*.otf")
               + glob.glob("/usr/share/fonts/opentype/inter/InterDisplay-*.otf")
               + glob.glob(str(Path.home() / ".local/share/fonts/DMSerifDisplay-*.ttf"))):
        try:
            fm.fontManager.addfont(fp)
        except Exception:
            pass
    plt.rcParams.update({
        "font.family": SANS,
        "font.size": 11.5,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": INK, "axes.linewidth": 1.0,
        "axes.facecolor": BG, "figure.facecolor": BG, "savefig.facecolor": BG,
        "legend.frameon": True, "legend.framealpha": 0.95,
        "legend.facecolor": "white", "legend.edgecolor": "#d9cfc0",
        "svg.fonttype": "none",
    })


# ----------------------------------------------------------------- style helpers
def style_ax(ax, *, grid="y"):
    """Cream background, ink left/bottom spines (top/right off), dotted value-axis grid."""
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(1.0)
    if grid:
        ax.grid(True, axis=grid, ls=":", lw=0.8, color=INK, alpha=0.22, zorder=0)
    ax.tick_params(length=0)


def title(ax, text, fs=15, pad=12):
    ax.set_title(text, loc="left", color=INK, pad=pad, fontfamily=SERIF, fontsize=fs)


def suptitle(fig, text, y=0.99, fs=18):
    fig.suptitle(text, color=INK, fontfamily=SERIF, fontsize=fs, y=y)


def emph(ax, x, y, text, color=GOLD, fs=11, **kw):
    """Bold emphasis label (gold by default; pass color=BLUE for the cooler accent)."""
    kw.setdefault("fontweight", "bold")
    return ax.text(x, y, text, color=color, fontsize=fs, **kw)


def block_shadow(ax, x, y, w, h, dx=5.5, dy=-5.5, z=2.4):
    """Hard-edged SOLID ink block offset behind a bar → reads as a 3-D block with volume."""
    trans = offset_copy(ax.transData, fig=ax.figure, x=dx, y=dy, units="points")
    r = Rectangle((x, y), w, h, facecolor=INK, edgecolor="none", zorder=z, transform=trans)
    ax.add_patch(r)
    return r


def vbars(ax, x, heights, color, hatch="", width=0.6, base=0.0, z=3, shadow=True):
    """Vertical bars: solid fill + ink border + hard 3-D block shadow.

    ``hatch`` should stay "" except for a bar that is conditional / not-directly-comparable,
    where a single pattern (e.g. "///") is a deliberate signal, not decoration.
    """
    cont = ax.bar(x, np.asarray(heights) - base, width, bottom=base,
                  color=color, edgecolor=INK, linewidth=1.3, zorder=z, hatch=(hatch or None))
    if shadow:
        for p in cont.patches:
            block_shadow(ax, p.get_x(), p.get_y(), p.get_width(), p.get_height(), z=z - 0.6)
    return cont


def hbars(ax, y, widths, color, hatch="", height=0.6, left=0.0, z=3, shadow=True):
    """Horizontal bars (same contract as vbars). For STACKED bars pass shadow=False on the
    segments and draw one block_shadow() spanning the whole bar so depth wraps the total."""
    cont = ax.barh(y, np.asarray(widths) - left, height, left=left,
                   color=color, edgecolor=INK, linewidth=1.3, zorder=z, hatch=(hatch or None))
    if shadow:
        for p in cont.patches:
            block_shadow(ax, p.get_x(), p.get_y(), p.get_width(), p.get_height(), z=z - 0.6)
    return cont
