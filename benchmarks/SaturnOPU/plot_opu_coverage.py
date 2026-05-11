#!/usr/bin/env python3
"""Generate paper-quality plots of OPU VOPACC coverage per model.

Reads the CSV output from analyze_opu_coverage.py and generates:
1. Stacked bar chart: dispatch count by type with OPU highlighting
2. Compute weight chart: assembly size by type with OPU highlighting
3. Summary comparison across models

Usage:
    python plot_opu_coverage.py [--csv opu_coverage_results.csv] [--output-dir .]
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# Colors matching existing Saturn OPU plots
OPU_COLOR = "#2ecc71"  # Green for OPU-accelerated
NON_OPU_COLOR = "#95a5a6"  # Gray for non-OPU
TYPE_COLORS = {
    "matmul": "#3498db",
    "conv": "#e74c3c",
    "elementwise": "#f39c12",
    "reduction": "#9b59b6",
    "matvec": "#1abc9c",
    "memcpy": "#bdc3c7",
    "softmax": "#e67e22",
    "transpose": "#34495e",
    "other": "#7f8c8d",
}
OPU_HATCH = "///"


def load_results(csv_path):
    """Load analysis results from CSV."""
    results = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["lines"] = int(row["lines"])
            row["vopacc"] = int(row["vopacc"])
            results.append(row)
    return results


def aggregate_by_model(results):
    """Aggregate results per model per op_type."""
    models = {}
    for r in results:
        if r["op_type"] == "encoding":
            continue
        model = r["model"]
        if model not in models:
            models[model] = defaultdict(lambda: {"count": 0, "opu_count": 0, "lines": 0, "opu_lines": 0})
        d = models[model][r["op_type"]]
        d["count"] += 1
        d["lines"] += r["lines"]
        if r["opu_status"] != "none":
            d["opu_count"] += 1
            d["opu_lines"] += r["lines"]
    return models


def plot_dispatch_breakdown(models, output_dir):
    """Stacked bar chart: dispatch count by type with OPU vs non-OPU."""
    model_names = list(models.keys())
    all_types = set()
    for m in models.values():
        all_types.update(m.keys())
    # Sort types by total count across all models
    type_totals = defaultdict(int)
    for m in models.values():
        for t, d in m.items():
            type_totals[t] += d["count"]
    op_types = sorted(all_types, key=lambda t: -type_totals[t])

    fig, axes = plt.subplots(1, len(model_names), figsize=(4 * len(model_names), 6), sharey=False)
    if len(model_names) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):
        data = models[model_name]
        types_present = [t for t in op_types if t in data]

        opu_counts = [data[t]["opu_count"] for t in types_present]
        non_opu_counts = [data[t]["count"] - data[t]["opu_count"] for t in types_present]

        x = np.arange(len(types_present))
        width = 0.6

        ax.bar(x, opu_counts, width, label="OPU (VOPACC)", color=OPU_COLOR, edgecolor="white")
        ax.bar(x, non_opu_counts, width, bottom=opu_counts, label="Non-OPU", color=NON_OPU_COLOR, edgecolor="white")

        ax.set_xlabel("Operator Type")
        ax.set_ylabel("Dispatch Count")
        ax.set_title(model_name, fontweight="bold", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(types_present, rotation=45, ha="right")
        ax.legend(loc="upper right", fontsize=8)

        # Add count labels
        for i, (o, n) in enumerate(zip(opu_counts, non_opu_counts)):
            total = o + n
            if total > 0:
                ax.text(i, total + 0.5, str(total), ha="center", va="bottom", fontsize=9)
            if o > 0:
                ax.text(i, o / 2, str(o), ha="center", va="center", fontsize=8, color="white", fontweight="bold")

        # Add OPU % annotation
        total_compute = sum(d["count"] for d in data.values())
        total_opu = sum(d["opu_count"] for d in data.values())
        pct = (total_opu * 100 // total_compute) if total_compute > 0 else 0
        ax.text(
            0.98,
            0.98,
            f"OPU: {total_opu}/{total_compute} ({pct}%)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=OPU_COLOR, alpha=0.3),
        )

    plt.tight_layout()
    path = os.path.join(output_dir, "opu_coverage_dispatches.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_compute_weight(models, output_dir):
    """Bar chart: compute weight (assembly lines) by type with OPU."""
    model_names = list(models.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(model_names))
    width = 0.35

    opu_lines = []
    non_opu_lines = []
    for model_name in model_names:
        data = models[model_name]
        ol = sum(d["opu_lines"] for d in data.values())
        nl = sum(d["lines"] - d["opu_lines"] for d in data.values())
        opu_lines.append(ol)
        non_opu_lines.append(nl)

    ax.bar(x, opu_lines, width * 2, label="OPU-accelerated compute", color=OPU_COLOR)
    ax.bar(x, non_opu_lines, width * 2, bottom=opu_lines, label="Non-OPU compute", color=NON_OPU_COLOR)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Compute Weight (assembly lines)", fontsize=12)
    ax.set_title("OPU Compute Coverage by Model", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=12)
    ax.legend(fontsize=10)

    # Add percentage labels
    for i, (o, n) in enumerate(zip(opu_lines, non_opu_lines)):
        total = o + n
        pct = (o * 100 // total) if total > 0 else 0
        ax.text(i, total + total * 0.02, f"{pct}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "opu_coverage_compute_weight.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="benchmarks/SaturnOPU/opu_coverage_results.csv")
    parser.add_argument("--output-dir", default="benchmarks/SaturnOPU/")
    args = parser.parse_args()

    results = load_results(args.csv)
    models = aggregate_by_model(results)

    plot_dispatch_breakdown(models, args.output_dir)
    plot_compute_weight(models, args.output_dir)
