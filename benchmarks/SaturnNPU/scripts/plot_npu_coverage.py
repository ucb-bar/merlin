#!/usr/bin/env python3
"""Generate useful plots for SmolVLA NPU kernel decomposition.

Only generates plots that are actually informative:
  1. Pareto coverage — which kernels to implement first
  2. Per-PyTorch-layer op decomposition — what each layer becomes
  3. Cross-level flow diagram — how ops transform through compilation

Usage:
    python scripts/plot_npu_coverage.py \
        benchmarks/SaturnNPU/smolvla_graph_manifest.json \
        --layer-trace benchmarks/SaturnNPU/layer_decomposition.json \
        --output-dir benchmarks/SaturnNPU/plots/
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> dict | list:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plot 1: Pareto coverage
# ---------------------------------------------------------------------------


def plot_pareto(manifest: dict, output_dir: Path) -> None:
    """Which kernels to implement first by FLOP impact."""
    pareto = manifest.get("coverage", {}).get("pareto", [])
    pareto = [p for p in pareto if p["flops"] > 0 and p["kernel_type"] != "other"]

    fig, ax = plt.subplots(figsize=(12, 6))

    names = [p["kernel_type"].replace("_", "\n") for p in pareto]
    individual = [p["flops_pct"] for p in pareto]
    cumulative = [p["cumulative_pct"] for p in pareto]
    instances = [p["instance_count"] for p in pareto]

    x = range(len(pareto))
    ax.bar(x, individual, color="#3498db", alpha=0.85, edgecolor="white")
    ax.plot(x, cumulative, "o-", color="#e74c3c", linewidth=2, markersize=5)

    ax.set_xlabel("Kernel Type", fontsize=10)
    ax.set_ylabel("% of Total Compute", fontsize=10)
    ax.set_title("Implement Kernel X -> Cover Y% of Total Model Compute", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7, rotation=45, ha="right")

    for i, p in enumerate(pareto):
        if p["flops_pct"] >= 0.3:
            ax.text(
                i,
                p["flops_pct"] + 0.5,
                f"{p['flops_pct']:.1f}%\n({instances[i]})",
                ha="center",
                va="bottom",
                fontsize=6,
            )
        ax.text(
            i,
            cumulative[i] + 0.5,
            f"{cumulative[i]:.1f}%",
            ha="center",
            va="bottom",
            fontsize=6,
            color="#e74c3c",
            fontweight="bold",
        )

    for m in [50, 90, 99]:
        ax.axhline(y=m, color="gray", linestyle="--", alpha=0.3, linewidth=0.7)

    ax.legend(["Cumulative %", "Individual %"], fontsize=9)
    ax.set_ylim(0, 108)
    plt.tight_layout()
    fig.savefig(output_dir / "pareto_coverage.png", dpi=150)
    plt.close(fig)
    print("  pareto_coverage.png")


# ---------------------------------------------------------------------------
# Plot 2: Per-PyTorch-layer op decomposition
# ---------------------------------------------------------------------------


def plot_layer_decomposition(trace_data: dict, output_dir: Path) -> None:
    """Stacked bar: what ops each PyTorch layer type decomposes into."""
    input_layers = trace_data.get("input_level_layers", [])
    if not input_layers:
        print("  No layer trace data, skipping.")
        return

    layer_types = defaultdict(lambda: Counter())
    layer_type_counts = Counter()
    for layer in input_layers:
        lt = layer["layer_type"]
        layer_type_counts[lt] += 1
        for op in layer["compute_ops"]:
            layer_types[lt][op["op"]] += 1

    lt_names = sorted(layer_types.keys())

    # Get top ops by total count
    op_totals = Counter()
    for counts in layer_types.values():
        op_totals.update(counts)
    top_ops = [op for op, _ in op_totals.most_common(12)]

    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.arange(len(lt_names))
    width = 0.5
    bottom = np.zeros(len(lt_names))

    colors = plt.cm.Set3(np.linspace(0, 1, len(top_ops)))

    for i, op_type in enumerate(top_ops):
        vals = np.array([layer_types[lt].get(op_type, 0) / max(layer_type_counts[lt], 1) for lt in lt_names])
        ax.bar(
            x,
            vals,
            width,
            bottom=bottom,
            label=op_type.replace("_", " "),
            color=colors[i],
            edgecolor="white",
            linewidth=0.5,
        )
        for j, v in enumerate(vals):
            if v >= 1.5:
                ax.text(x[j], bottom[j] + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=7, fontweight="bold")
        bottom += vals

    lt_labels = {
        "siglip_encoder": f"SigLIP Encoder\n({layer_type_counts['siglip_encoder']}x)",
        "gemma_decoder": f"Gemma Decoder\n({layer_type_counts['gemma_decoder']}x)",
    }
    ax.set_xticks(x)
    ax.set_xticklabels([lt_labels.get(lt, lt) for lt in lt_names], fontsize=10)
    ax.set_ylabel("Avg Ops per Layer", fontsize=10)
    ax.set_title("How Each PyTorch Layer Decomposes into MLIR Ops", fontsize=12)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_dir / "layer_op_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  layer_op_decomposition.png")


# ---------------------------------------------------------------------------
# Plot 3: Cross-level flow
# ---------------------------------------------------------------------------


def plot_cross_level_flow(manifest: dict, output_dir: Path) -> None:
    """Flow diagram: PyTorch ops -> Linalg/Input -> Global-Opt."""
    flows = [
        (
            "torch.aten.linear (382)",
            "quantized_matmul_fp8\n+ dequant chain",
            "quantized_matmul_fp8 (379)\ndequant HOISTED",
        ),
        ("torch.aten.sdpa (36)", "iree_linalg_ext.attention (36)", "iree_linalg_ext.attention (36)"),
        (
            "torch.aten.layer_norm (75)",
            "rms_norm + reduce_sum\n+ div + mul (13 ops/norm)",
            "rms_norm (139)\nreduce (214) + div (217)",
        ),
        ("torch.aten.matmul/mm (114)", "batch_matmul (46)\n+ matmul (67)", "batch_matmul (46)\n+ matmul (67)"),
        ("torch.aten.softmax (24)", "softmax_exp (23)\n+ max_reduce (29)", "linalg.softmax (23)"),
        ("torch.aten.gelu (36)", "gelu_tanh (36)", "gelu_tanh (36)"),
        ("torch.aten.silu (33)", "silu (32)", "silu (32)"),
        ("MX fp8 dequant (446)", "dequant_bitshift (231)\nnan_detect, select, ...", "HOISTED to initializers"),
        ("linalg.transpose (774)", "linalg.transpose (774)", "ELIMINATED (fused)"),
    ]

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(-0.3, 3.0)
    ax.set_ylim(-1, len(flows) + 0.5)
    ax.axis("off")

    cols = [0, 1.1, 2.2]
    headers = ["PyTorch / Torch-MLIR", "Linalg / Input", "Global-Opt (vanilla)"]

    for cx, h in zip(cols, headers):
        ax.text(
            cx,
            len(flows) + 0.1,
            h,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1", edgecolor="#bdc3c7"),
        )

    for i, (pytorch, input_op, gopt) in enumerate(flows):
        y = len(flows) - 1 - i
        is_eliminated = "HOISTED" in gopt or "ELIMINATED" in gopt
        color = (
            "#e74c3c"
            if is_eliminated
            else "#3498db"
            if "matmul" in pytorch.lower() or "linear" in pytorch.lower()
            else "#2ecc71"
            if "attention" in pytorch.lower() or "sdpa" in pytorch.lower()
            else "#9b59b6"
        )
        alpha = 0.25 if is_eliminated else 0.6

        for cx, text in [(cols[0], pytorch), (cols[1], input_op), (cols[2], gopt)]:
            ax.text(
                cx,
                y,
                text,
                ha="center",
                va="center",
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=alpha, edgecolor="gray", linewidth=0.5),
            )

        for j in range(len(cols) - 1):
            ax.annotate(
                "",
                xy=(cols[j + 1] - 0.32, y),
                xytext=(cols[j] + 0.32, y),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1, alpha=0.5),
            )

    legend = [
        mpatches.Patch(facecolor="#3498db", alpha=0.6, label="Compute (matmul/linear)"),
        mpatches.Patch(facecolor="#2ecc71", alpha=0.6, label="Attention"),
        mpatches.Patch(facecolor="#9b59b6", alpha=0.6, label="Activation / Norm / Other"),
        mpatches.Patch(facecolor="#e74c3c", alpha=0.25, label="Eliminated / Hoisted"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9)
    ax.set_title("SmolVLA: How PyTorch Ops Transform Through MLIR Compilation", fontsize=13, fontweight="bold", pad=15)

    plt.tight_layout()
    fig.savefig(output_dir / "cross_level_flow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  cross_level_flow.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--layer-trace", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/SaturnNPU/plots"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(args.manifest)
    trace_path = args.layer_trace or Path("benchmarks/SaturnNPU/layer_decomposition.json")
    trace_data = load_json(trace_path) if trace_path.exists() else {}

    print("Generating plots...")
    plot_pareto(manifest, args.output_dir)
    plot_layer_decomposition(trace_data, args.output_dir)
    plot_cross_level_flow(manifest, args.output_dir)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
