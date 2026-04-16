#!/usr/bin/env python3
"""Plot per-model OPU-vs-other cycle breakdown from the CSV emitted by
parse_dispatch_cycles.py.

Groups dispatches into three buckets using the `opu_path` column from
model_dispatch_decomposition.csv:
  - "opu"           : opu_path in {"opu", "opu_matmul", "opu_fused", ...}
  - "rvv"           : opu_path in {"rvv", "rvv_reduction", ...}
  - "other"         : everything else (data_movement, empty, etc.)

Output: two figures under benchmarks/SaturnOPU/figures/:
  - opu_fraction_stacked.png   — per (model, variant) stacked bar
  - opu_fraction_per_dispatch.png — per-dispatch top-N breakdown
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OPU_PATHS = {"opu", "opu_matmul", "opu_fused", "opu_bme"}
RVV_PATHS = {"rvv", "rvv_reduction", "rvv_fallback"}


def bucket(opu_path: str) -> str:
    if not opu_path:
        return "other"
    if opu_path in OPU_PATHS:
        return "opu"
    if opu_path in RVV_PATHS:
        return "rvv"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "cycles_csv",
        type=Path,
        nargs="?",
        default=Path("/tmp/firesim_dispatch_cycles.csv"),
    )
    ap.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    totals = defaultdict(lambda: defaultdict(int))
    with args.cycles_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["model"], row["variant"])
            totals[key][bucket(row["is_opu_path"])] += int(row["total_cycles"] or 0)

    keys = sorted(totals)
    labels = [f"{m}_{v}" for m, v in keys]
    opu = [totals[k]["opu"] for k in keys]
    rvv = [totals[k]["rvv"] for k in keys]
    oth = [totals[k]["other"] for k in keys]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), 4))
    ax.bar(labels, opu, label="OPU", color="#3c76d1")
    ax.bar(labels, rvv, bottom=opu, label="RVV", color="#d17a3c")
    ax.bar(
        labels,
        oth,
        bottom=[a + b for a, b in zip(opu, rvv)],
        label="other",
        color="#888888",
    )
    ax.set_ylabel("cycles (summed across workgroups)")
    ax.set_title("per-dispatch cycle totals grouped by execution path")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    p = args.out_dir / "opu_fraction_stacked.png"
    plt.savefig(p, dpi=140)
    print(f"wrote {p}")
    plt.close()

    # Normalized fraction view.
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), 4))
    totals_s = [o + r + x for o, r, x in zip(opu, rvv, oth)]
    fopu = [o / t if t else 0.0 for o, t in zip(opu, totals_s)]
    frvv = [r / t if t else 0.0 for r, t in zip(rvv, totals_s)]
    foth = [x / t if t else 0.0 for x, t in zip(oth, totals_s)]
    ax.bar(labels, fopu, label="OPU", color="#3c76d1")
    ax.bar(labels, frvv, bottom=fopu, label="RVV", color="#d17a3c")
    ax.bar(
        labels,
        foth,
        bottom=[a + b for a, b in zip(fopu, frvv)],
        label="other",
        color="#888888",
    )
    ax.set_ylabel("fraction of total runtime")
    ax.set_ylim(0, 1)
    ax.set_title("OPU vs RVV vs other — fraction of total dispatch cycles")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    p = args.out_dir / "opu_fraction_normalized.png"
    plt.savefig(p, dpi=140)
    print(f"wrote {p}")
    plt.close()


if __name__ == "__main__":
    main()
