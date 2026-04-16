"""Horizontal bar chart: OPU vs RVV speedup per model.

Reads /tmp/sweep_iters.csv (produced by run_final_sweep.sh --phase=clean)
and computes RVV_avg / OPU_avg per model. Geomean annotated on the plot.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from palette import COLUMN_WIDTH_IN, JOURNEY_COLORS, apply_paper_style

BENCH_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iters-csv",
        type=Path,
        default=Path("/tmp/sweep_iters.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=BENCH_DIR / "figures",
    )
    parser.add_argument(
        "--out-name",
        default="opu_vs_rvv_speedup",
    )
    args = parser.parse_args()

    by: dict[str, dict[str, int]] = {}
    with open(args.iters_csv) as f:
        for row in csv.DictReader(f):
            v = row.get("variant", "")
            avg = row.get("avg", "")
            if v in ("opu", "rvv") and avg.isdigit():
                by.setdefault(row["model"], {})[v] = int(avg)

    pairs = [(model, d["opu"], d["rvv"], d["rvv"] / d["opu"]) for model, d in by.items() if "opu" in d and "rvv" in d]
    pairs.sort(key=lambda x: x[3], reverse=True)
    if not pairs:
        raise SystemExit(f"no (opu, rvv) pairs in {args.iters_csv}")

    apply_paper_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 0.35 * len(pairs) + 0.7))
    y = list(range(len(pairs)))[::-1]
    speedups = [p[3] for p in pairs]
    ax.barh(y, speedups, color=JOURNEY_COLORS[4], alpha=0.85, edgecolor="white", linewidth=0.4)
    for yi, (model, o, r, s) in zip(y, pairs):
        ax.text(s + 0.02 * max(speedups), yi, f"{s:.2f}×", va="center", ha="left", fontsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels([p[0] for p in pairs])
    ax.axvline(1.0, color="#888", linewidth=0.6, linestyle="--", alpha=0.6, zorder=0)
    ax.set_xlabel("Speedup (RVV cycles / OPU cycles)")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    # Geomean annotation.
    gmean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    ax.set_title(f"{len(pairs)} models; geomean speedup = {gmean:.2f}×", fontsize=7, pad=6)
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"{args.out_name}.{ext}")
    plt.close(fig)
    print(f"wrote {args.out_dir / (args.out_name + '.png')}")
    print(f"      geomean = {gmean:.3f}x across {len(pairs)} models")


if __name__ == "__main__":
    main()
