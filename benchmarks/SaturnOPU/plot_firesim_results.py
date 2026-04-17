"""Generate benchmark plots for Saturn OPU (V128-D64) Int8 MatMul performance.

Reads data from CSV files and generates 4 separate PNG plots:
  1. performance_scaling.png  — Ops/Cycle vs Matrix Size (OPU vs RVV)
  2. speedup_vs_rvv.png       — Speedup bar chart
  3. utilization.png          — OPU hardware utilization
  4. optimization_journey.png — Performance progression at 1024x1024

Usage:
  uv run python benchmarks/SaturnOPU/plot_firesim_results.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).parent

# Hardware parameters
DLEN = 64
PEAK_FLOPS_PER_CYCLE = 2 * (DLEN // 8) ** 2  # 128

# Style constants
TITLE_SIZE = 16
LABEL_SIZE = 14
LEGEND_SIZE = 12
TICK_SIZE = 12
DPI = 300
COLOR_OPU = "#008080"
COLOR_RVV = "#ff8c00"
COLOR_SPEEDUP = "#00b050"
COLOR_UTIL = "#336699"
COLOR_PEAK = "#cc0000"


def load_results():
    """Load OPU and RVV results from CSV."""
    sizes = []
    opu_ops = []
    rvv_ops = []
    opu_cycles = []
    rvv_cycles = []

    with open(SCRIPT_DIR / "firesim_v128d64_results.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            size = int(row["size_m"])
            if row["variant"] == "OPU":
                sizes.append(size)
                opu_ops.append(float(row["ops_per_cycle"]))
                opu_cycles.append(int(row["avg_cycles"]))
            else:
                rvv_ops.append(float(row["ops_per_cycle"]))
                rvv_cycles.append(int(row["avg_cycles"]))

    return (
        np.array(sizes),
        np.array(opu_ops),
        np.array(rvv_ops),
        np.array(opu_cycles),
        np.array(rvv_cycles),
    )


def load_journey():
    """Load optimization journey data from CSV."""
    labels = []
    ops = []
    descriptions = []

    with open(SCRIPT_DIR / "optimization_journey.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row["label"])
            ops.append(float(row["ops_per_cycle"]))
            descriptions.append(row["description"])

    return labels, np.array(ops), descriptions


def plot_performance_scaling(sizes, opu_ops, rvv_ops):
    """Plot 1: Ops/Cycle vs Matrix Size."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x_labels = [str(s) for s in sizes]
    x = np.arange(len(sizes))

    ax.plot(
        x,
        opu_ops,
        color=COLOR_OPU,
        linewidth=3,
        marker="o",
        markersize=10,
        label="OPU (RVV + VOPACC)",
        zorder=5,
    )
    ax.plot(
        x,
        rvv_ops,
        color=COLOR_RVV,
        linewidth=3,
        marker="s",
        markersize=8,
        label="RVV Baseline",
        zorder=5,
    )
    ax.axhline(
        y=PEAK_FLOPS_PER_CYCLE,
        color=COLOR_PEAK,
        linestyle="--",
        linewidth=1.5,
        label=f"Peak ({PEAK_FLOPS_PER_CYCLE} FLOPs/cycle)",
    )

    # Data labels on OPU points
    for i, v in enumerate(opu_ops):
        ax.annotate(
            f"{v:.1f}",
            (x[i], v),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color=COLOR_OPU,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=TICK_SIZE)
    ax.set_xlabel("Matrix Size N (N\u00d7N\u00d7N matmul, i8\u00d7i8\u2192i32)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Ops / Cycle (2\u00d7M\u00d7N\u00d7K / cycles)", fontsize=LABEL_SIZE)
    ax.set_title(
        "Saturn OPU (V128-D64) Int8 MatMul Performance\n" "IREE Bare-Metal on FireSim",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    ax.legend(fontsize=LEGEND_SIZE, loc="upper left")
    ax.grid(True, linestyle="-", alpha=0.3)
    ax.set_ylim(0, PEAK_FLOPS_PER_CYCLE * 1.1)

    fig.tight_layout()
    out = SCRIPT_DIR / "performance_scaling.png"
    fig.savefig(out, dpi=DPI)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_speedup(sizes, opu_ops, rvv_ops):
    """Plot 2: Speedup over RVV baseline."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x_labels = [str(s) for s in sizes]
    speedup = opu_ops / rvv_ops

    bars = ax.bar(
        x_labels,
        speedup,
        color=COLOR_SPEEDUP,
        edgecolor="black",
        width=0.6,
        zorder=3,
    )

    for bar, s in zip(bars, speedup):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.8,
            f"{s:.1f}\u00d7",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xlabel("Matrix Size N (N\u00d7N\u00d7N matmul, i8\u00d7i8\u2192i32)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Speedup (OPU / RVV)", fontsize=LABEL_SIZE)
    ax.set_title(
        "OPU Speedup vs RVV Baseline\nSaturn V128-D64, IREE on FireSim", fontsize=TITLE_SIZE, fontweight="bold"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_ylim(0, max(speedup) * 1.15)
    ax.tick_params(labelsize=TICK_SIZE)

    fig.tight_layout()
    out = SCRIPT_DIR / "speedup_vs_rvv.png"
    fig.savefig(out, dpi=DPI)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_utilization(sizes, opu_ops):
    """Plot 3: OPU hardware utilization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x_labels = [str(s) for s in sizes]
    util = (opu_ops / PEAK_FLOPS_PER_CYCLE) * 100

    bars = ax.bar(
        x_labels,
        util,
        color=COLOR_UTIL,
        edgecolor="black",
        width=0.6,
        zorder=3,
    )
    ax.axhline(y=100, color=COLOR_PEAK, linestyle="--", linewidth=1.5, label="100%")

    for bar, u in zip(bars, util):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.0,
            f"{u:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xlabel("Matrix Size N (N\u00d7N\u00d7N matmul, i8\u00d7i8\u2192i32)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Utilization (%)", fontsize=LABEL_SIZE)
    ax.set_title(
        "OPU Hardware Utilization (V128-D64)\n" f"Peak = 2\u00d7(DLEN/8)\u00b2 = {PEAK_FLOPS_PER_CYCLE} FLOPs/cycle",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    ax.legend(fontsize=LEGEND_SIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_ylim(0, 115)
    ax.tick_params(labelsize=TICK_SIZE)

    fig.tight_layout()
    out = SCRIPT_DIR / "utilization.png"
    fig.savefig(out, dpi=DPI)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_optimization_journey(labels, ops):
    """Plot 4: Optimization journey at 1024x1024."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Color gradient: red → orange → yellow → green → teal → blue
    colors = ["#cc3333", "#e06030", "#ddaa00", "#55aa22", "#009966", "#0066aa"]

    bars = ax.bar(
        range(len(labels)),
        ops,
        color=colors,
        edgecolor="black",
        width=0.65,
        zorder=3,
    )

    for bar, v in zip(bars, ops):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.8,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10, rotation=20, ha="right")
    ax.set_ylabel("Ops / Cycle", fontsize=LABEL_SIZE)
    ax.set_title(
        "Optimization Journey (1024\u00d71024\u00d71024, i8)\n" "Saturn OPU V128-D64, IREE on FireSim",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_ylim(0, max(ops) * 1.15)
    ax.tick_params(labelsize=TICK_SIZE)

    fig.tight_layout()
    out = SCRIPT_DIR / "optimization_journey.png"
    fig.savefig(out, dpi=DPI)
    print(f"Saved: {out}")
    plt.close(fig)


def main():
    sizes, opu_ops, rvv_ops, _, _ = load_results()
    journey_labels, journey_ops, _ = load_journey()

    plot_performance_scaling(sizes, opu_ops, rvv_ops)
    plot_speedup(sizes, opu_ops, rvv_ops)
    plot_utilization(sizes, opu_ops)
    plot_optimization_journey(journey_labels, journey_ops)

    print("\nAll plots saved to:", SCRIPT_DIR)


if __name__ == "__main__":
    main()
