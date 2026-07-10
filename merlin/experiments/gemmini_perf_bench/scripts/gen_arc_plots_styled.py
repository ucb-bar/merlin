#!/usr/bin/env python3
"""House-style (merlin_plotstyle) versions of fig_arc_checks + fig_arc_landscape — paper/presentation
ready: short clean titles, no baked caption paragraph, no text clipping through objects or dark fills.
Imports the single house style module (scripts/merlin_plotstyle.py) — never re-derives the palette.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = Path("/path/to/oscar-merlin")
sys.path.insert(0, str(REPO / "scripts"))
from merlin_plotstyle import (use_merlin_style, style_ax, title, suptitle, emph, vbars, block_shadow,
                              BG, INK, GOLD, BLUE, NAVY, SLATE, MAUVE, SAGE, SERIF, SANS)

REPORTS = REPO / "merlin" / "experiments" / "gemmini_perf_bench" / "reports"
RF = REPO / "merlin/targets/gemmini/contracts/rtl_facts"
ARC = json.loads((RF / "arc_results.json").read_text())
_aw = [c["wall_s"] for c in ARC["capsules"] if c.get("wall_s")]
ARC_WALL_MED = sorted(_aw)[len(_aw) // 2] if _aw else 3.7e-3
_ref = ARC.get("rtl_wall_ref", {})
VERI_WALL = _ref.get("verilator_wall_s_median") or 655.0
FSIM_WALL = _ref.get("firesim_per_run_s_typ") or 210.0


def _savefig(fig, name):
    png = REPORTS / f"fig_arc_{name}.png"
    svg = REPORTS / f"fig_arc_{name}.svg"
    fig.savefig(png, bbox_inches="tight", dpi=170, facecolor=BG)
    fig.savefig(svg, bbox_inches="tight", facecolor=BG)
    print(f"wrote {png}")


# ----------------------------------------------------------------------------- checks
def fig_checks():
    fig, (axm, axr) = plt.subplots(1, 2, figsize=(12, 5.0), gridspec_kw={"width_ratios": [1, 1]})
    # --- confusion matrix (flat ink-edged cells; bold high-contrast labels) ---
    axm.set_xlim(-0.05, 2.05); axm.set_ylim(-0.05, 2.05); axm.invert_yaxis(); axm.axis("off")
    cells = [  # (row, col, text, facecolor, textcolor)
        (0, 0, "TN\n242", SLATE, "white"),
        (0, 1, "FP\n0",   "#E8E0D2", INK),     # 0 false positives — pale, ink text
        (1, 0, "FN\n17",  MAUVE, "white"),
        (1, 1, "TP\n124", NAVY,  "white"),
    ]
    for r, c, txt, fc, tc in cells:
        axm.add_patch(Rectangle((c + .04, r + .04), .92, .92, fc=fc, ec=INK, lw=1.4, zorder=3))
        axm.text(c + .5, r + .5, txt, ha="center", va="center", fontsize=20,
                 fontweight="bold", color=tc, zorder=4, fontfamily=SANS)
    # axis annotations placed OUTSIDE the grid (no clipping into cells or title)
    axm.text(1.0, 2.16, "check verdict  →   accept / reject", ha="center", va="top",
             fontsize=10.5, color=INK)
    axm.text(-0.14, 1.0, "oracle   pass / fail", rotation=90, va="center", ha="center",
             fontsize=10.5, color=INK)
    title(axm, "RTL-checks vs oracle — 0 false positives", fs=14, pad=16)

    # --- recall lift bars (house vbars w/ 3-D block shadow; data-scaled axis) ---
    style_ax(axr, grid="y")
    vals = [91 / 141 * 100, 124 / 141 * 100]
    xb = np.arange(2)
    vbars(axr, [xb[0]], [vals[0]], MAUVE, width=0.58)
    vbars(axr, [xb[1]], [vals[1]], NAVY, width=0.58)
    for i, (v, n) in enumerate(zip(vals, [91, 124])):
        axr.text(i, v + 2.2, f"{n}/141", ha="center", fontsize=11, fontweight="bold", color=INK)
        axr.text(i, v - 5.5, f"{v:.0f}%", ha="center", fontsize=12, fontweight="bold",
                 color="white")
    emph(axr, 1, vals[1] + 8.5, "+23 pts", color=GOLD, fs=12, ha="center")
    axr.set_xticks(xb); axr.set_xticklabels(["base checks", "+ fence / preload\n+ config order"], fontsize=10.5)
    axr.set_ylim(0, 105)
    axr.set_ylabel("recall on real RTL-tier failures (%)", fontsize=11.5)
    title(axr, "Recall lift from general invariants", fs=14, pad=16)

    fig.tight_layout(rect=(0, 0, 1, 0.99))
    _savefig(fig, "checks")


# ----------------------------------------------------------------------------- landscape
def fig_landscape():
    arc_thru = 1.0 / ARC_WALL_MED
    veri_thru = 1.0 / VERI_WALL
    fsim_thru = 1.0 / FSIM_WALL
    # name, throughput, fidelity, color, short-note, label-side ('L'/'R'/'T'), note-side
    pts = [
        ("spike (functional)",        1e3,       1.0, MAUVE, "fast · not faithful",            "T"),
        ("static RTL-checks",         3e2,       2.0, SAGE,  "~3 ms · structural only",        "T"),
        ("arc middle-tier (ours)",    arc_thru,  3.2, NAVY,  f"RTL datapath: bit-exact numerics · cycles @ ideal-mem · {ARC_WALL_MED*1e3:.0f} ms", "B"),
        ("verilator (L3)",            veri_thru, 3.7, SLATE, f"RTL-faithful · ~{VERI_WALL:.0f} s", "R"),
        ("FireSim (L5)",              fsim_thru, 4.4, BLUE,  f"RTL+FPGA · ~{FSIM_WALL:.0f} s/run", "R"),
    ]
    fig, ax = plt.subplots(figsize=(12, 6.6))
    style_ax(ax, grid=None)
    ax.set_xscale("log")
    ax.set_xlim(veri_thru / 4, 1e4); ax.set_ylim(0.4, 5.0)
    # RTL-faithful band + right-edge label (clear of every bubble)
    ax.axhspan(2.9, 4.7, color=NAVY, alpha=0.05, zorder=0)
    ax.text(7e3, 4.55, "RTL-derived\ntiers", ha="right", va="top", fontsize=10,
            color="#9a8f78", style="italic", zorder=1)
    # bubbles
    for name, spd, fid, col, note, side in pts:
        ax.scatter([spd], [fid], s=560, color=col, edgecolor=INK, lw=1.6, zorder=5)
        if side == "T":
            ax.annotate(name, (spd, fid), xytext=(0, 30), textcoords="offset points",
                        ha="center", va="bottom", fontsize=11, fontweight="bold", zorder=6)
            ax.annotate(note, (spd, fid), xytext=(0, -32), textcoords="offset points",
                        ha="center", va="top", fontsize=9.5, color="#5A5A5A", zorder=6)
        elif side == "B":
            ax.annotate(name, (spd, fid), xytext=(0, 30), textcoords="offset points",
                        ha="center", va="bottom", fontsize=11, fontweight="bold", zorder=6)
            ax.annotate(note, (spd, fid), xytext=(0, -32), textcoords="offset points",
                        ha="center", va="top", fontsize=9.5, color="#5A5A5A", zorder=6)
        else:  # 'R' — label to the right, clear of the bubble + the band label
            ax.annotate(name, (spd, fid), xytext=(34, 6), textcoords="offset points",
                        ha="left", va="center", fontsize=11, fontweight="bold", zorder=6)
            ax.annotate(note, (spd, fid), xytext=(34, -12), textcoords="offset points",
                        ha="left", va="center", fontsize=9.5, color="#5A5A5A", zorder=6)
    # arc -> verilator speedup arrow + emphasis, routed BELOW the bubbles (no label crossing)
    ratio = arc_thru / veri_thru
    ax.annotate("", xy=(arc_thru, 3.05), xytext=(veri_thru, 3.5),
                arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=2.4, alpha=0.95,
                                connectionstyle="arc3,rad=-0.18"), zorder=4)
    # HONEST framing: not "same fidelity" (arc is plotted a tier lower) and not false-precise. The big
    # wall-clock gap is real but apples-to-oranges — verilator wall is SoC-boot-dominated; arc is an
    # isolated, ideal-memory, boot-free kernel run. Round to an order of magnitude and state the caveat.
    ratio_round = int(round(ratio, -4))
    ax.text(np.sqrt(arc_thru * veri_thru), 2.60,
            f"~{ratio_round:,}× less wall-clock\nno SoC boot · ideal-mem · isolated",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color=GOLD, zorder=7)
    ax.set_yticks([1.0, 2.0, 3.2, 3.7, 4.4])
    ax.set_yticklabels(["functional\nnumerics", "structural\n(ISA-legal)", "RTL numerics\n+ cycles",
                        "RTL (full)", "RTL + FPGA"], fontsize=9.5)
    ax.set_ylabel("fidelity  →", fontsize=12)
    ax.set_xlabel("throughput = 1 / wall  (kernels/s, log)   →  faster", fontsize=12)
    title(ax, "Where each Gemmini oracle sits", fs=16, pad=12)
    fig.tight_layout()
    _savefig(fig, "landscape")


if __name__ == "__main__":
    use_merlin_style()
    fig_checks()
    fig_landscape()
