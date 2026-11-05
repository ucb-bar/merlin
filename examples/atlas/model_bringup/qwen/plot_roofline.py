"""Plot the scheduled Qwen measurements against the memory/compute roofline."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.results.read_text())["results"]
    labels = {
        "attention": "Attention",
        "swiglu": "SwiGLU",
        **{f"projection_k{k}": f"K={k}" for k in (128, 256, 512)},
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")
    colors = plt.get_cmap("tab10")
    for ax, memory, title in zip(
        axes,
        ("curve", "fast-fixed"),
        ("Default VCS memory curve", "Fast fixed memory (sensitivity test)"),
    ):
        selected = [
            r for r in rows if r["variant"] == "scheduled" and r["memory"] == memory
        ]
        peak = selected[0]["peak_flops_per_cycle"]
        # Finite-run DRAM charges count N-1 beat intervals. Use the large-run
        # backend rate for the plotted line, not those slightly inflated rates.
        if memory == "fast-fixed":
            bandwidth = 32.0
        else:
            from npu_model.hardware.memory_backend import CurveMemoryBackend
            from npu_model.configs.hardware.default import DefaultHardwareConfig

            config = DefaultHardwareConfig()
            bandwidth = (
                CurveMemoryBackend(**config.dma_memory_params).peak_bandwidth
                * config.dma_beat_bytes
            )
        intensity = np.geomspace(8, 8192, 400)
        ax.loglog(
            intensity,
            np.minimum(bandwidth * intensity, peak),
            color="black",
            lw=2,
            label="DRAM / two-MXU roof",
        )
        for index, row in enumerate(selected):
            ax.scatter(
                row["arithmetic_intensity"],
                row["attained_flops_per_cycle"],
                color=colors(index),
                s=60,
                label=labels[row["workload"]],
                zorder=3,
            )
        ax.set(
            xlim=(8, 8192),
            ylim=(8, 7000),
            xlabel="Arithmetic intensity (MXU FLOP / DRAM byte)",
            title=title,
        )
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("Attained MXU FLOP / cycle")
    fig.suptitle(
        "Qwen kernels on Atlas: scheduled programs\nFP8/BF16 diagnostic arithmetic; VPU FLOPs excluded"
    )
    fig.savefig(args.out, dpi=180)
    vector = args.out.with_suffix(".svg")
    fig.savefig(vector)
    vector.write_text(
        "\n".join(line.rstrip() for line in vector.read_text().splitlines()) + "\n"
    )


if __name__ == "__main__":
    main()
