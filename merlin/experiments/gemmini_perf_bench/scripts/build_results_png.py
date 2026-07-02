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

# (figure stem, section caption) in reading order; hero first.
# Four sections: A middle-tier · B perf/fidelity · C static checks · D agentic authoring effort.
PANELS = [
    ("fig_arc_landscape", "A1 · Oracle landscape — the arc middle-tier fills the spike↔verilator gap"),
    ("fig_arc_bitexact", "A2 · Arc middle-tier — 20/20 bit-exact vs golden + per-capsule cycles"),
    ("fig_arc_speed", "A3 · Arc vs verilator/FireSim wall time — RTL-faithful, measured ~10⁵× (no SoC boot)"),
    ("fig_arc_latency", "A4 · Timing realism — arc cycles scale with modelled memory latency (1 capsule)"),
    ("fig_arc_hostcomm", "A5 · Host↔accelerator telemetry (RoCC control + DMA traffic)"),
    ("fig_cycles", "B1 · Cycle-accurate cycles — generated vs hand-tuned (verilator L3 + FireSim L5)"),
    ("fig_capability", "B2 · Functional correctness & op coverage (spike L2) — NOT timing"),
    ("fig_spike_not_timing", "B3 · Why spike ≠ performance (functional sim plateaus ~120 cyc)"),
    ("fig_iree_profile", "B4 · IREE profiled on FireSim L5 (its oracle) — 10–40× slower, 1–6% util"),
    ("fig_arc_checks", "C1 · RTL-derived static checks vs oracle — 0/242 FP, 65%→88% recall"),
    ("fig_arc_mutation", "C2 · Pre-screen catches RTL failures in ms vs verilator seconds (1 capsule)"),
    ("fig_agentic_trajectory", "D1 · Authoring trajectory — baseline vs merlin agent, activity over transcript"),
    ("fig_agentic_effort", "D2 · Authoring effort — baseline vs merlin agent (PILOT: N=3/N=1)"),
    ("fig_agentic_convergence", "D3 · Per-round convergence — capsules passing vs round (pilot)"),
    ("fig_agentic_coverage", "D4 · Capability coverage by op-class — the conv+movement gap (pilot)"),
    ("fig_agentic_per_capsule_effort", "D5 · Downstream efficiency — effort per pilot capsule passed"),
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
         "deterministic, RTL-grounded (CIRCT) · 20/20 arc bit-exact · 0/242 check FP · 24/24 cycle-accurate · "
         "measured ~10⁵× vs verilator · + agentic authoring-effort A/B (pilot)",
         ha="center", va="center", fontsize=10.5, color="#5A5A5A")
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
