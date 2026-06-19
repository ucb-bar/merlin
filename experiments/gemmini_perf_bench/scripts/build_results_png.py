"""Compose all RTL-checks + arc + perf figures into a single poster PNG (cream background + title).
-> reports/RTL_ARC_RESULTS.png"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import _pbcommon as PB

CREAM, INK = "#F6F1E7", "#2B2B2B"
R = PB.REPORTS

# (figure stem, section caption) in reading order; hero first
PANELS = [
    ("fig_arc_landscape", "1 · Oracle landscape — the arc middle-tier fills the spike↔verilator gap"),
    ("fig_arc_bitexact", "2 · Arc middle-tier — 20/20 bit-exact vs golden + per-capsule cycles"),
    ("fig_arc_speed", "3 · Arc vs verilator wall time — RTL-faithful at ~10⁴× (no SoC boot)"),
    ("fig_arc_latency", "4 · Timing realism — arc cycles scale with modelled memory latency"),
    ("fig_arc_hostcomm", "5 · Host↔accelerator telemetry (RoCC control + DMA traffic)"),
    ("fig_arc_checks", "6 · RTL-derived static checks vs oracle — 0/242 FP, 65%→88% recall"),
    ("fig_arc_mutation", "7 · Pre-screen catches RTL failures in ms vs verilator seconds"),
    ("fig_cycles", "8 · Cross-approach cycles (verilator + FireSim L5, 24/24 kernels)"),
    ("fig_capability", "9 · Correctness & capability (spike) — who compiles each op"),
    ("fig_spike_not_timing", "10 · Why spike ≠ performance (functional sim plateaus)"),
]
imgs = [(c, mpimg.imread(R / f"{s}.png")) for s, c in PANELS if (R / f"{s}.png").is_file()]

PAGE_W = 13.0          # inches
TITLE_H = 1.1          # inches for the header band
CAP_H = 0.42           # inches per caption row
heights = [PAGE_W * (im.shape[0] / im.shape[1]) + CAP_H for _, im in imgs]
total_h = TITLE_H + sum(heights) + 0.3

fig = plt.figure(figsize=(PAGE_W, total_h), facecolor=CREAM)
y = total_h
# title band
fig.text(0.5, 1 - 0.45 / total_h, "RTL-derived checks + arcilator middle-tier — results",
         ha="center", va="center", fontsize=22, fontweight="bold", color=INK)
fig.text(0.5, 1 - 0.85 / total_h,
         "deterministic, RTL-grounded (CIRCT) · 20/20 arc bit-exact · 0/242 check FP · 24/24 cycle-accurate · >10⁴× vs verilator",
         ha="center", va="center", fontsize=11, color="#5A5A5A")
y -= TITLE_H
for (cap, im), h in zip(imgs, heights):
    # caption
    fig.text(0.06, (y - CAP_H * 0.55) / total_h, cap, ha="left", va="center",
             fontsize=12.5, fontweight="bold", color=INK)
    # image axes below the caption
    ax_h = h - CAP_H
    ax = fig.add_axes([0.04, (y - h) / total_h, 0.92, ax_h / total_h])
    ax.imshow(im); ax.axis("off")
    y -= h

out = R / "RTL_ARC_RESULTS.png"
fig.savefig(out, dpi=130, facecolor=CREAM, bbox_inches="tight")
plt.close(fig)
import os
print(f"wrote {out} ({os.path.getsize(out)//1024} KB, {len(imgs)} panels, {total_h:.0f}in tall)")
