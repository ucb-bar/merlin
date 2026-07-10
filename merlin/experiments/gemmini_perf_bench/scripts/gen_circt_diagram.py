#!/usr/bin/env python3
"""Slide: how the CIRCT tooling works (fig7-CCA visual idiom — pipeline ribbon + detail cards).

Faithful to capsule METHODOLOGY §7:
  Gemmini Chisel/FIRRTL --firtool --ir-hw--> CIRCT HW-dialect MLIR
     --introspect (port widths · 16x16 array · scratchpad/acc memories · ISA funct table)--> facts.json
     --compile (+ capsule declared shape)--> machine-checkable FileCheck assertions + numeric bounds
  each round: the agent's emitted gemmini-MLIR + RoCC trace is checked by FileCheck (LLVM) + a numeric
  screen -> findings, fed back to the agent as ADVISORY feedback (spike/verilator still decide pass/fail).
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

REPO = Path("/path/to/merlin")
sys.path.insert(0, str(REPO / "scripts"))
from merlin_plotstyle import use_merlin_style, suptitle, BG, INK, GOLD, NAVY, SLATE, MAUVE, SAGE, SERIF, SANS

REPORTS = REPO / "merlin" / "experiments" / "gemmini_perf_bench" / "reports"
SH = [pe.withSimplePatchShadow(offset=(2.5, -2.5), shadow_rgbFace=(0.18, 0.178, 0.173), alpha=0.20, rho=1.0)]
BODY = "#FCFAF5"
MONO = "monospace"


def _tc(hexc):
    r, g, b = (int(hexc.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    return INK if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else "white"


def chip(ax, cx, cy, w, h, text, fc, *, fs=11.5):
    b = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                       facecolor=fc, edgecolor=INK, linewidth=1.4, zorder=4)
    b.set_path_effects(SH); ax.add_patch(b)
    ax.text(cx, cy, text, ha="center", va="center", color=_tc(fc), fontsize=fs,
            fontfamily=SANS, fontweight="bold", zorder=5, linespacing=1.0)


def panel(ax, x, y, w, h, header):
    body = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                          facecolor=BODY, edgecolor=INK, linewidth=1.5, zorder=3)
    body.set_path_effects(SH); ax.add_patch(body)
    hh = 0.52
    hd = FancyBboxPatch((x + 0.04, y + h - hh - 0.04), w - 0.08, hh,
                        boxstyle="round,pad=0.01,rounding_size=0.05", facecolor=NAVY, edgecolor="none", zorder=4)
    ax.add_patch(hd)
    ax.text(x + 0.22, y + h - hh / 2 - 0.04, header, ha="left", va="center", color="white",
            fontsize=12, fontfamily=SERIF, zorder=5)
    return x + 0.28, y + h - hh - 0.30, w - 0.56     # content left, content top, content width


def arrow(ax, p0, p1, *, rad=0.0, color=INK, lw=2.0, dotted=False):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", zorder=3, shrinkA=1, shrinkB=1,
                                 linestyle=("dotted" if dotted else "solid")))


def alabel(ax, x, y, s, *, fs=9.5, color=INK, weight="normal"):
    ax.text(x, y, s, ha="center", va="center", fontsize=fs, color=color, fontweight=weight, zorder=6,
            fontfamily=SANS, linespacing=1.0, bbox=dict(boxstyle="round,pad=0.18", fc=BG, ec="#d9cfc0", lw=0.7))


def codelines(ax, lx, ty, lines, *, fs=10, dy=0.345, comment="#9a8b6e"):
    for i, ln in enumerate(lines):
        col = comment if ln.strip().startswith("//") else INK
        ax.text(lx, ty - i * dy, ln, ha="left", va="top", color=col, fontsize=fs, fontfamily=MONO, zorder=6)


def main():
    use_merlin_style()
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off"); ax.set_facecolor(BG)

    # ---------- top pipeline ribbon ----------
    chips = [("Gemmini RTL", SAGE), ("HW-dialect\nMLIR", NAVY), ("facts.json", NAVY),
             ("FileCheck\nchecks", NAVY), ("findings", MAUVE), ("Coding agent", GOLD)]
    cw, cy, ch = 1.95, 7.95, 0.74
    cxs = [1.4 + i * 2.66 for i in range(6)]
    for (txt, fc), cx in zip(chips, cxs):
        chip(ax, cx, cy, cw, ch, txt, fc, fs=11)
    albls = ["firtool\n--ir-hw", "introspect", "compile\n(+ shape)", "run over\nagent's trace", "advisory\nevery round"]
    for i, lab in enumerate(albls):
        x0, x1 = cxs[i] + cw / 2, cxs[i + 1] - cw / 2
        col = GOLD if i == 4 else INK
        arrow(ax, (x0, cy), (x1, cy), color=col, lw=2.4 if i == 4 else 2.0)
        alabel(ax, (x0 + x1) / 2, cy + 0.04, lab, fs=8.7, color=(NAVY if i < 4 else "#7a6a40"),
               weight="bold")

    # ---------- detail card 1: HW-dialect — what we read ----------
    lx, ty, cwd = panel(ax, 0.45, 4.15, 3.75, 2.75, "HW-dialect MLIR — what we read")
    rows = [("module port widths", "i8 · i32"), ("systolic array", "16 × 16"),
            ("scratchpad / acc mem", "256 KB"), ("ISA funct table", "legal funct")]
    for i, (k, v) in enumerate(rows):
        yy = ty - i * 0.52
        ax.add_patch(Rectangle((lx - 0.08, yy - 0.21), cwd + 0.02, 0.42, facecolor=GOLD, alpha=0.16, zorder=4))
        ax.text(lx, yy, k, ha="left", va="center", color=INK, fontsize=10.5, fontfamily=SANS, zorder=6)
        ax.text(lx + cwd, yy, v, ha="right", va="center", color=NAVY, fontsize=10.5, fontfamily=MONO,
                fontweight="bold", zorder=6)

    # ---------- detail card 2: facts.json ----------
    lx, ty, _ = panel(ax, 4.35, 4.15, 3.5, 2.75, "facts.json — the artifact")
    codelines(ax, lx, ty, ['{', '  "array": [16, 16],', '  "scratchpad_bytes": 262144,',
                           '  "legal_funct": [3,4,5,6,7,8],', '  "custom_opcode": "0x7b"  }'],
              fs=9.2, dy=0.37)

    # ---------- bottom-wide panel: findings against the agent's kernel ----------
    lx, ty, cwd = panel(ax, 0.45, 0.55, 7.4, 3.25, "checked against the agent's kernel — findings (this round)")
    finds = [("funct opcode", "in legal table", "0x1f", "illegal"),
             ("spad_addr", "< 0x40000", "0x48000", "over capacity"),
             ("mvout (tiles)", "16  (16×16 array)", "240", "15× over")]
    cL, cG, cV = lx + 2.35, lx + 4.25, lx + cwd - 0.15      # legal · agent-got · verdict columns
    ax.text(lx, ty + 0.04, "check", ha="left", va="center", color="#7a6a5a", fontsize=9.5, fontfamily=SANS, zorder=6)
    ax.text(cL, ty + 0.04, "legal (from facts)", ha="left", va="center", color="#7a6a5a", fontsize=9.5, zorder=6)
    ax.text(cG, ty + 0.04, "agent got", ha="left", va="center", color="#7a6a5a", fontsize=9.5, zorder=6)
    ax.text(cV, ty + 0.04, "verdict", ha="right", va="center", color="#7a6a5a", fontsize=9.5, zorder=6)
    for i, (k, legal, got, verd) in enumerate(finds):
        yy = ty - 0.55 - i * 0.62
        ax.add_patch(Rectangle((lx - 0.08, yy - 0.26), cwd + 0.04, 0.52, facecolor=MAUVE, alpha=0.13, zorder=4))
        ax.text(lx, yy, k, ha="left", va="center", color=INK, fontsize=10, fontfamily=MONO, zorder=6)
        ax.text(cL, yy, legal, ha="left", va="center", color=INK, fontsize=9.6, fontfamily=MONO, zorder=6)
        ax.text(cG, yy, got, ha="left", va="center", color=INK, fontsize=9.6, fontfamily=MONO, zorder=6)
        ax.text(cV, yy, verd, ha="right", va="center", color=MAUVE, fontsize=10, fontfamily=SANS,
                fontweight="bold", zorder=6)

    # ---------- right tall card: machine-checkable + how it's used ----------
    lx, ty, cwd = panel(ax, 8.1, 0.55, 7.45, 6.35, "rtl_checks — machine-checkable, advisory")
    ax.text(lx, ty + 0.02, "compiled assertions  ·  run by the LLVM FileCheck binary",
            ha="left", va="top", color=NAVY, fontsize=11, fontfamily=SANS, fontweight="bold", zorder=6)
    codelines(ax, lx + 0.1, ty - 0.55,
              ['// CHECK:        funct in legal_funct',
               '// CHECK:        spad_addr < 0x40000',
               '// CHECK-COUNT-16: mvout      // 16×16 → tiles',
               '// CHECK-NOT:     UNKNOWN opcode'], fs=10.3, dy=0.42)
    yb = ty - 2.15
    ax.plot([lx, lx + cwd], [yb, yb], color="#d9cfc0", lw=1.0, zorder=5)
    ax.text(lx, yb - 0.28, "how it is used", ha="left", va="top", color=NAVY, fontsize=11,
            fontfamily=SANS, fontweight="bold", zorder=6)
    # pills
    def pill(px, py, w, txt, active):
        fc = NAVY if active else "#e7ddcd"
        b = FancyBboxPatch((px, py - 0.22), w, 0.44, boxstyle="round,pad=0.02,rounding_size=0.2",
                           facecolor=fc, edgecolor=INK, linewidth=1.1, zorder=5)
        ax.add_patch(b)
        ax.text(px + w / 2, py, txt, ha="center", va="center", color=("white" if active else "#8a8174"),
                fontsize=10, fontfamily=SANS, fontweight="bold", zorder=6)
    pill(lx, yb - 0.95, 2.1, "ADVISORY", True)
    pill(lx + 2.35, yb - 0.95, 3.0, "gates pass / fail", False)
    bullets = ["grounded in the real RTL — not documentation",
               "a tool checks the agent's code — not prose",
               "fed back to the agent as feedback every round",
               "spike / verilator still decide correctness"]
    for i, bt in enumerate(bullets):
        yy = yb - 1.65 - i * 0.46
        ax.text(lx + 0.05, yy, "•", ha="left", va="center", color=NAVY, fontsize=12, zorder=6)
        ax.text(lx + 0.35, yy, bt, ha="left", va="center", color=INK, fontsize=10.5, fontfamily=SANS, zorder=6)

    # ---------- faint "zoom" connectors ribbon -> cards ----------
    for cx, tgt in [(cxs[1], (2.0, 6.9)), (cxs[2], (6.1, 6.9)), (cxs[3], (11.8, 6.9))]:
        arrow(ax, (cx, cy - ch / 2), tgt, dotted=True, color="#c7bca9", lw=1.4)

    suptitle(fig, "How the CIRCT tooling works", y=0.985, fs=23)
    fig.text(0.5, 0.93, "it compiles the real Gemmini RTL into machine-checkable assertions that grade the agent's kernel each round",
             ha="center", va="center", fontsize=13, color=INK, fontfamily=SANS, style="italic")

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    REPORTS.mkdir(exist_ok=True)
    png, svg = REPORTS / "fig_circt_tooling.png", REPORTS / "fig_circt_tooling.svg"
    fig.savefig(png, bbox_inches="tight", dpi=300, facecolor=BG)
    fig.savefig(svg, bbox_inches="tight", facecolor=BG)
    print(f"wrote {png}\nwrote {svg}")


if __name__ == "__main__":
    raise SystemExit(main())
