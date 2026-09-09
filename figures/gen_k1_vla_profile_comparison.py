#!/usr/bin/env python3
"""Generate the K1 VLA latency + whole-model profile figure from its frozen JSON receipt."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
DATA = HERE / "k1_vla_profile_comparison_20260907.json"
OUT_PDF = HERE / "k1_vla_profile_comparison.pdf"
OUT_PNG = HERE / "k1_vla_profile_comparison.png"

COLORS = {
    "layout_copy": "#2F75B5",
    "contraction": "#F1C232",
    "elementwise": "#C77DBB",
    "reduction_softmax": "#2E8B57",
    "other": "#B8B8B8",
    "hybrid": "#D55E00",
    "blocked": "#C7352A",
    "ink": "#222222",
    "grid": "#D9D9D9",
}
LABELS = {
    "layout_copy": "Layout / copy",
    "contraction": "Contraction",
    "elementwise": "Elementwise",
    "reduction_softmax": "Reduction / softmax",
    "other": "Other / unattributed",
}
CATEGORIES = list(LABELS)


def _latency_label(ms: float) -> str:
    return f"{ms / 1000:.2f} s" if ms >= 1000 else f"{ms:.1f} ms"


def _validate(data: dict) -> None:
    if [row["model"] for row in data["models"]] != ["BitVLA", "OpenVLA", "RDT2"]:
        raise ValueError("unexpected model set/order")
    for row in data["models"]:
        if abs(sum(row["categories_ms"].values()) - row["profile_wall_ms"]) > 0.01:
            raise ValueError(f"profile categories do not sum to wall for {row['model']}")
        if row["cosine"] < 0.9999:
            raise ValueError(f"ungated result for {row['model']}")
        if row["hybrid_ms"] is not None and row["hybrid_cosine"] < 0.9999:
            raise ValueError(f"ungated hybrid result for {row['model']}")


def main() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    _validate(data)
    rows = data["models"]
    names = [row["model"] + ("*" if not row["profile_trusted"] else "") for row in rows]
    warm = np.array([row["warm_ms"] for row in rows], dtype=float)
    hybrid = np.array([row["hybrid_ms"] or np.nan for row in rows], dtype=float)
    shares = {
        category: np.array([
            row["categories_ms"][category] / row["profile_wall_ms"] for row in rows
        ])
        for category in CATEGORIES
    }

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
    with mpl.rc_context(rc):
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.15), gridspec_kw={"wspace": 0.30})
        x = np.arange(len(rows))

        ax = axes[0]
        width = 0.34
        ax.bar(x - width / 2, warm, width=width, color="#5B9BD5", edgecolor=COLORS["ink"],
               linewidth=0.8, label="Merlin generated")
        ax.bar(x + width / 2, hybrid, width=width, color=COLORS["hybrid"],
               edgecolor=COLORS["ink"], linewidth=0.8, label="Merlin + XNNPACK GEMM")
        ax.set_yscale("log")
        ax.set_ylim(20, 2.0e4)
        ax.set_xticks(x, names)
        ax.set_ylabel("Warm inference latency (ms, log scale)")
        ax.grid(axis="y", which="major", color=COLORS["grid"], linewidth=0.8, zorder=0)
        for i, value in enumerate(warm):
            ax.text(i - width / 2, value * 1.16, _latency_label(value), ha="center", va="bottom", fontsize=9,
                    fontweight="bold")
            if np.isfinite(hybrid[i]):
                ax.text(i + width / 2, hybrid[i] * 1.16, _latency_label(hybrid[i]), ha="center",
                        va="bottom", fontsize=9, fontweight="bold")
            else:
                ax.text(i + width / 2, -0.11, "× blocked", transform=ax.get_xaxis_transform(),
                        ha="center", va="center", fontsize=8.5, fontweight="bold",
                        color=COLORS["blocked"], clip_on=False)
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        ax.set_title("1   Warm latency: generated vs kernel swap", loc="left", fontsize=14,
                     fontweight="bold")

        ax = axes[1]
        bottom = np.zeros(len(rows))
        for category in CATEGORIES:
            values = shares[category]
            ax.bar(x, values, width=0.62, bottom=bottom, color=COLORS[category],
                   edgecolor="white", linewidth=0.6)
            for i, value in enumerate(values):
                if value >= 0.075:
                    ax.text(i, bottom[i] + value / 2, f"{100 * value:.1f}%", ha="center",
                            va="center", fontsize=9, fontweight="bold",
                            color="white" if category == "layout_copy" else COLORS["ink"])
            bottom += values
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.linspace(0, 1, 6), [f"{int(v)}%" for v in np.linspace(0, 100, 6)])
        ax.set_xticks(x, names)
        ax.set_ylabel("Share of warm runtime")
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8, zorder=0)
        ax.set_title("2   Where the wall goes", loc="left", fontsize=14, fontweight="bold")
        handles = [Patch(facecolor=COLORS[c], edgecolor="none", label=LABELS[c]) for c in CATEGORIES]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=5,
                   frameon=False, columnspacing=1.35, handlelength=1.5)
        fig.text(0.5, 0.012,
                 "SpaceMiT K1 · 1 core · FP32 reduced-depth deterministic captures · "
                 "3 launches, 2 warmup + 5 timed · perturbation-gated operator profiles",
                 ha="center", va="bottom", fontsize=8.8, color="#555555")
        fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.25)
        fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight", pad_inches=0.06)
        fig.savefig(OUT_PNG, format="png", bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)
    print(f"saved {OUT_PDF}")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
