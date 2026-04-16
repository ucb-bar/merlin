#!/usr/bin/env python3
"""Plot the Saturn OPU optimization journey."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from palette import JOURNEY_COLORS, JOURNEY_FIGSIZE, apply_paper_style

BENCH_DIR = Path(__file__).resolve().parent


def load_rows(path: Path) -> list[dict[str, str]]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [row for row in rows if row["label"] != "Initial mmt4d+OPU"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=BENCH_DIR / "optimization_journey.csv")
    parser.add_argument("--out-dir", type=Path, default=BENCH_DIR / "figures")
    parser.add_argument("--config-title", default="1024x1024x1024, i8")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style()

    rows = load_rows(args.csv)
    labels = [row["label"] for row in rows]
    values = [float(row["ops_per_cycle"]) for row in rows]

    fig, ax = plt.subplots(figsize=JOURNEY_FIGSIZE)
    bars = ax.bar(
        range(len(values)),
        values,
        width=0.72,
        color=JOURNEY_COLORS[: len(values)],
        edgecolor="black",
        linewidth=0.5,
    )
    ymax = max(values) if values else 1.0
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ymax * 0.018,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            fontweight="bold",
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Ops / cycle")
    ax.set_title(f"Optimization journey ({args.config_title})")
    ax.set_ylim(0, ymax * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = args.out_dir / f"optimization_journey.{ext}"
        fig.savefig(out)
        print(f"  -> {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
