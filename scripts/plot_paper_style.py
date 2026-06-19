#!/usr/bin/env python
"""Paper-style (Physical-Intelligence / ML-paper aesthetic) figures of the RVV results.

Style vocabulary borrowed from the reference figures:
  - muted natural palette (dusty salmon / sage / steel / warm gold); "ours" = gold, prior = salmon
  - black bar outlines, value label on every bar
  - rounded cream "card" panels, bold serif titles + figure captions
  - boxed callout annotations with leader lines on the key datapoint
  - a soft shaded band marking the crossover / selected regime

All numbers transcribed verbatim from the committed result docs (sources in each section).
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "output" / "kernels" / "ceiling"

# ---- palette ----
INK     = "#2b2b2b"
SALMON  = "#cf8b7d"   # prior / baseline  (their π0.5 colour)
SAGE    = "#9bb08a"   # competitor A (OpenBLAS)
STEEL   = "#6f93b0"   # competitor B (XNNPACK)
GOLD    = "#e7c25c"   # OURS  (their "ours" colour)
TEAL    = "#7bb4c4"   # ceiling reference (dashed)
GREY    = "#9a9a9a"
CREAM   = "#f5f1e6"
CARD_EC = "#33312b"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.edgecolor": "#555", "axes.linewidth": 0.9,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})

def card(ax, title):
    """Give an axes the cream rounded-card look with a bold title."""
    ax.set_facecolor(CREAM)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, loc="left", color=INK, pad=8)

def callout(ax, xy, text, xytext, fc="#fff6e0", ec=GOLD):
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=9.5, color=INK, ha="center",
                bbox=dict(boxstyle="round,pad=0.35", fc=fc, ec=ec, lw=1.3),
                arrowprops=dict(arrowstyle="-", color=ec, lw=1.3,
                                connectionstyle="arc3,rad=0.0"))


# ============================================================================
# FIGURE A — whole-model e2e "card" (the headline; image-copy-7 style)
# bitvla 3-config + per-model speedups. Source: output/rvv_bench/k1_e2e_*.md
# ============================================================================
def fig_e2e():
    fig = plt.figure(figsize=(12, 5.4))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.22)

    # -- left: bitvla three-way, horizontal gold-bar card (latency ms) --
    ax = fig.add_subplot(gs[0]); card(ax, "bitvla — whole-model latency on K1 silicon")
    rows = [("baseline\n(hand_v0)", 2517, SALMON, "1.00×", False),
            ("ours\n(vfmacc)", 274, GOLD, "9.18×", False),
            ("xnnpack\nkernels", 184, STEEL, "13.65×", True)]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, ms, col, sp, fast) in zip(y, rows):
        ax.barh(yi, ms, height=0.62, color=col, edgecolor=CARD_EC, linewidth=1.6, zorder=3)
        tag = f"{ms} ms   ({sp})" + ("   ← fastest" if fast else "")
        ax.text(ms + 70, yi, tag, va="center", ha="left", fontsize=11,
                fontweight="bold", color=(STEEL if fast else INK))
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)
    ax.set_xlim(0, 4200); ax.set_xlabel("latency (ms / forward) — lower is better")
    ax.set_xticks([0, 1000, 2000, 3000])
    callout(ax, (184, y[2]+0.30), "same graph + weights,\nonly GEMM kernel swapped → 1.49× over ours",
            (1950, y[2]+0.30), fc="#eef3f7", ec=STEEL)
    ax.set_ylim(-0.6, len(rows)-0.35)

    # -- right: per-model compiler speedup (gold bars) --
    ax = fig.add_subplot(gs[1]); card(ax, "compiler-emitted speedup vs frozen baseline")
    models = [("rdt2", 2.35, "73.7 → 31.4 s"), ("openvla", 3.61, "5.85 → 1.62 s"),
              ("bitvla", 9.18, "2.52 → 0.274 s")]
    y = np.arange(len(models))
    for yi, (m, sp, lab) in zip(y, models):
        ax.barh(yi, sp, height=0.6, color=GOLD, edgecolor=CARD_EC, linewidth=1.6, zorder=3)
        ax.text(sp + 0.15, yi, f"{sp}×", va="center", fontsize=12, fontweight="bold", color=INK)
        ax.text(0.2, yi, lab, va="center", fontsize=9, color=INK)
    ax.axvline(1.0, color=GREY, lw=1, ls="--")
    ax.set_yticks(y); ax.set_yticklabels([m[0] for m in models], fontsize=11)
    ax.set_xlim(0, 10.5); ax.set_xlabel("whole-model speedup (×) — higher is better")
    ax.set_ylim(-0.6, len(models)-0.4)

    fig.text(0.5, -0.02,
             "Figure A:  Whole-model RVV speedups on real K1 silicon.  Compiler-emitted vfmacc lowering "
             "gives 2.4–9.2× over the frozen baseline across three VLA models (cos ≥ 0.99999).  Swapping in\n"
             "XNNPACK's hand-written RVV GEMM for bitvla's matmuls reaches 13.65×, isolating ~1.49× of "
             "remaining matmul-codegen headroom; the rest of the win is shared runtime.",
             ha="center", fontsize=9.3, color=INK)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"paper_e2e.{ext}", bbox_inches="tight", dpi=160)
    print("wrote", OUT / "paper_e2e.png")
    plt.close(fig)


# ============================================================================
# FIGURE B — GEMM ceiling crossover on K1 (image-copy-2 annotated-line style)
# Source: output/kernels/ceiling/large_shape_packing_k1.md (inner-compute)
# ============================================================================
def fig_crossover():
    shapes = [32, 64, 128, 256, 384, 512]
    series = [
        ("ours-baseline",  [39246, 306608, 2516021, 19613446, None, None], SALMON, "-",  "o"),
        ("ours-tiled (compiler)", [2694, 20309, 168957, 1297217, None, None], GOLD, "-", "o"),
        ("OpenBLAS",       [409, 2329, 17789, 149008, 491134, 1146939], SAGE, "-", "s"),
        ("XNNPACK",        [238, 1976, 31761, 248282, 891310, 2079033], STEEL, "-", "^"),
        ("ours-intrinsic (hand ceiling)", [172, 1350, 14144, 106034, 441356, 1394602], TEAL, "--", "D"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 5.6)); ax.set_facecolor(CREAM)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    # shaded "ours-wins" region up to the crossover
    ax.axvspan(28, 430, color=GOLD, alpha=0.10, zorder=0)
    for name, ys, col, ls, mk in series:
        xs = [s for s, v in zip(shapes, ys) if v]; yy = [v for v in ys if v]
        ax.plot(xs, yy, ls=ls, marker=mk, ms=6, lw=2.4, color=col, label=name,
                markeredgecolor=CARD_EC, markeredgewidth=0.6, zorder=4)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(shapes); ax.set_xticklabels([f"{s}³" for s in shapes])
    ax.set_xlabel("GEMM size  (M=N=K)"); ax.set_ylabel("K1 rdtime ticks  (log; lower is faster)")
    ax.set_title("Single-GEMM ceiling on real K1 silicon", loc="left", color=INK, pad=10)
    ax.grid(True, which="both", ls=":", alpha=0.35, zorder=1)
    callout(ax, (384, 8.0e4), "ours-intrinsic beats both\nexperts through 384³",
            (96, 4.5e5), fc="#eaf4f6", ec=TEAL)
    callout(ax, (512, 1146939), "OpenBLAS retakes\nlead at 512³ (0.82×)",
            (512, 9.0e4), fc="#eef3ec", ec=SAGE)
    ax.text(40, 1.1e7, "ours-baseline never\nforms vfmacc (~100× off)", fontsize=8.6,
            color=SALMON, style="italic")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95, facecolor="white")
    fig.text(0.5, -0.03,
             "Figure B:  GEMM compute ceiling, K1 inner-compute ticks.  A compiler-emitted register-blocked "
             "kernel (ours-intrinsic, dashed = hand-written ceiling reference) beats OpenBLAS and XNNPACK\n"
             "from 32³ to 384³; OpenBLAS's cache-blocking retakes the lead at 512³.  The shipped compiler "
             "path (ours-tiled) trails the experts ~10×; the baseline ~100×.",
             ha="center", fontsize=9.3, color=INK)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"paper_crossover.{ext}", bbox_inches="tight", dpi=160)
    print("wrote", OUT / "paper_crossover.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_e2e()
    fig_crossover()
