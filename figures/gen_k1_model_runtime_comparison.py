#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "k1_model_runtime_comparison_20260907.json"
OUT_PDF = HERE / "k1_model_runtime_comparison.pdf"
OUT_PNG = HERE / "k1_model_runtime_comparison.png"

COLORS = {
    "executorch": "#D55E00",
    "merlin": "#2F75B5",
    "hybrid": "#F1C232",
    "blocked": "#C7352A",
    "success": "#2E8B57",
    "ink": "#222222",
    "grid": "#D9D9D9",
}


def _label_ms(value_ms: float, lower_bound: bool = False) -> str:
    prefix = ">" if lower_bound else ""
    if value_ms >= 1000:
        return f"{prefix}{value_ms / 1000:.1f} s"
    if value_ms >= 100:
        return f"{prefix}{value_ms:.0f} ms"
    return f"{prefix}{value_ms:.1f} ms"


def _validate(data: dict) -> None:
    expected = ["LSTMNetVIT", "SmallLLaMA", "TinyLLaMA", "SmolVLA"]
    actual = [row["model"] for row in data["models"]]
    if actual != expected:
        raise ValueError(f"unexpected model order: {actual}")
    for row in data["models"]:
        if not row["executorch_ms"] or not row["merlin_ms"]:
            raise ValueError(f"missing required measured/bounded value for {row['model']}")
        if row["merlin_xnnpack_ms"] is not None:
            raise ValueError("qd8 hybrid must remain absent until the board backend is validated")


def _blocked_slots(ax: plt.Axes, xpos: np.ndarray) -> None:
    """Put unavailable cells in an axes-relative strip, never at a fake data value."""
    for x0 in xpos:
        ax.text(
            x0,
            -0.165,
            "× N/A",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="center",
            color=COLORS["blocked"],
            fontsize=8.5,
            fontweight="bold",
            clip_on=False,
        )


def main() -> None:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    _validate(data)
    rows = data["models"]
    model_names = ["LSTMNetVIT", "SmallLLaMA†", "TinyLLaMA", "SmolVLA‡"]
    executorch = np.array([row["executorch_ms"] for row in rows], dtype=float)
    merlin = np.array([row["merlin_ms"] for row in rows], dtype=float)
    lower_bound = np.array([row["merlin_kind"] == "lower_bound" for row in rows])
    slowdown = merlin / executorch

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
    with mpl.rc_context(rc):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12.8, 5.15),
            gridspec_kw={"width_ratios": [1.18, 1.0], "wspace": 0.28},
        )
        x = np.arange(len(rows), dtype=float)
        width = 0.23
        offsets = {"executorch": -width, "merlin": 0.0, "hybrid": width}

        # (a) Absolute latency. Log scale keeps 1.9 ms through >120 s legible.
        ax = axes[0]
        ax.bar(
            x + offsets["executorch"],
            executorch,
            width,
            color=COLORS["executorch"],
            edgecolor=COLORS["ink"],
            linewidth=0.8,
            zorder=3,
        )
        merlin_bars = ax.bar(
            x + offsets["merlin"],
            merlin,
            width,
            color=COLORS["merlin"],
            edgecolor=COLORS["ink"],
            linewidth=0.8,
            zorder=3,
        )
        merlin_bars[-1].set_hatch("///")
        ax.set_yscale("log")
        ax.set_ylim(0.7, 3.0e5)
        ax.set_ylabel("Warm inference latency (ms, log scale)")
        ax.set_xticks(x, model_names)
        ax.grid(axis="y", which="major", color=COLORS["grid"], linewidth=0.8, zorder=0)
        _blocked_slots(ax, x + offsets["hybrid"])
        for i, value in enumerate(executorch):
            ax.text(
                x[i] + offsets["executorch"],
                value * 1.20,
                _label_ms(value),
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=8.5,
                color=COLORS["ink"],
            )
        for i, value in enumerate(merlin):
            ax.text(
                x[i] + offsets["merlin"],
                value * 1.20,
                _label_ms(value, bool(lower_bound[i])),
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=8.5,
                fontweight="bold" if lower_bound[i] else "normal",
                color=COLORS["ink"],
            )
        ax.annotate(
            "",
            xy=(x[-1] + offsets["merlin"] + 0.075, 2.25e5),
            xytext=(x[-1] + offsets["merlin"] + 0.075, merlin[-1]),
            arrowprops={"arrowstyle": "-|>", "color": COLORS["ink"], "lw": 1.4},
        )
        ax.text(
            0.01,
            1.04,
            "1",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="white",
            fontsize=13,
            fontweight="bold",
            bbox={"boxstyle": "circle,pad=0.30", "fc": COLORS["success"], "ec": "none"},
        )
        ax.text(
            0.065,
            1.04,
            "Absolute latency",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=14,
            fontweight="bold",
        )

        # (b) Normalized latency makes the remaining compiler gap immediately readable.
        ax = axes[1]
        ax.axhline(1.0, color=COLORS["success"], linestyle="--", linewidth=1.4, zorder=1)
        ax.bar(
            x + offsets["executorch"],
            np.ones_like(executorch),
            width,
            color=COLORS["executorch"],
            edgecolor=COLORS["ink"],
            linewidth=0.8,
            zorder=3,
        )
        ratio_bars = ax.bar(
            x + offsets["merlin"],
            slowdown,
            width,
            color=COLORS["merlin"],
            edgecolor=COLORS["ink"],
            linewidth=0.8,
            zorder=3,
        )
        ratio_bars[-1].set_hatch("///")
        ax.set_ylim(0, 3.75)
        ax.set_ylabel("Latency normalized to ExecuTorch")
        ax.set_xticks(x, model_names)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8, zorder=0)
        _blocked_slots(ax, x + offsets["hybrid"])
        for i, value in enumerate(slowdown):
            prefix = ">" if lower_bound[i] else ""
            ax.text(
                x[i] + offsets["merlin"],
                value + 0.08,
                f"{prefix}{value:.2f}x",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        ax.annotate(
            "",
            xy=(x[-1] + offsets["merlin"] + 0.075, 3.55),
            xytext=(x[-1] + offsets["merlin"] + 0.075, slowdown[-1]),
            arrowprops={"arrowstyle": "-|>", "color": COLORS["ink"], "lw": 1.4},
        )
        ax.text(
            0.01,
            1.04,
            "2",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="white",
            fontsize=13,
            fontweight="bold",
            bbox={"boxstyle": "circle,pad=0.30", "fc": COLORS["success"], "ec": "none"},
        )
        ax.text(
            0.065,
            1.04,
            "Gap to ExecuTorch",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=14,
            fontweight="bold",
        )

        legend_handles = [
            Patch(facecolor=COLORS["executorch"], edgecolor=COLORS["ink"], label="ExecuTorch + XNNPACK"),
            Patch(facecolor=COLORS["merlin"], edgecolor=COLORS["ink"], label="Merlin generated"),
            Line2D([], [], marker="x", linestyle="None", markersize=9, markeredgewidth=2.2,
                   color=COLORS["blocked"], label="Merlin + XNNPACK qd8: blocked"),
            Patch(facecolor=COLORS["merlin"], edgecolor=COLORS["ink"], hatch="///",
                  label="timeout lower bound"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=4,
            frameon=False,
            handlelength=1.8,
            columnspacing=1.6,
        )
        fig.text(
            0.5,
            0.09,
            "SpaceMiT K1 · 1 core · diagnostic, not a paired certification set · "
            "† cross-run pair · ‡ timeout/reference-qualified",
            ha="center",
            va="center",
            fontsize=9.2,
            color="#555555",
        )
        fig.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.29)
        fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight", pad_inches=0.06)
        fig.savefig(OUT_PNG, format="png", bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)
    print(f"saved {OUT_PDF}")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
