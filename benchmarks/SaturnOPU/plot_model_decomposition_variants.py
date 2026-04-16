#!/usr/bin/env python3
"""Generate alternative layouts of the per-model dispatch decomposition.

Produces three sibling figures to the canonical `per_model_decomposition*.png`,
each of which complements the dispatch-share breakdown with an OPU
coverage metric weighted by something more meaningful than dispatch
count: parameters (by layer type) and/or analytical compute (MACs).

Layouts:
  A. `..._with_params.png`       — per model: dispatch-share bar on top,
                                   thin param/compute coverage bar below.
  B. `..._side_by_side.png`      — two panels, left dispatch-share,
                                   right param/compute coverage.
  C. `..._parameters_only.png`   — compact one-row-per-model coverage bar.

All three figures are marked "layer-level estimate" for the coverage
metric so readers don't read them as measured execution-time coverage
(which is the domain of `sweep_cycles.csv`).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

from palette import (  # noqa: E402
    COLUMN_WIDTH_IN,
    DECOMPOSITION_BASE_HEIGHT,
    DECOMPOSITION_ROW_HEIGHT,
    OPU_SEGMENTS,
    SEGMENT_ALPHA,
    SEGMENT_COLORS,
    SEGMENT_LABELS,
    SEGMENT_ORDER,
    apply_paper_style,
)

MODEL_PARAMS = {
    "mlp": "320",
    "mlp_wide": "2.2K",
    "opu_bench_large_mlp": "1.4M",
    "opu_bench_vit_small": "399K",
    "opu_bench_vit": "19.2M",
    "opu_bench_hybrid": "475K",
    "opu_bench_convnet": "697K",
    "dronet": "312K",
    "yolov8_nano": "3.2M",
    "tinyllama": "1.1B",
}

OPU_COLOR = "#117A8B"  # dark teal — matches AOT 32×32
NONOPU_COLOR = "#c9ced4"  # neutral light grey


# ----------------------------- data loading --------------------------------


def load_rows(summary_csv: Path, include: list[str] | None):
    rows = list(csv.DictReader(summary_csv.open()))
    if include:
        keep = set(include)
        missing = keep - {r["model_key"] for r in rows}
        if missing:
            raise SystemExit(f"unknown model_keys: {sorted(missing)}")
        rows = [r for r in rows if r["model_key"] in keep]
    return rows


def apply_renames(rows, rename_arg: str | None):
    if not rename_arg:
        return
    for pair in rename_arg.split(","):
        if ":" not in pair:
            continue
        k, v = pair.split(":", 1)
        for r in rows:
            if r["model_key"] == k.strip():
                r["model"] = v.strip()


def sort_by_speedup(rows, iters_csv: Path):
    if not iters_csv.exists():
        return
    last = {}
    for r in csv.DictReader(iters_csv.open()):
        avg = (r.get("avg") or "").strip()
        if avg.isdigit():
            last[(r["model"], r["variant"])] = int(avg)
    sp = {}
    for (m, v), val in list(last.items()):
        if v != "opu":
            continue
        rv = last.get((m, "rvv"))
        if rv and val:
            sp[m] = rv / val
    rows.sort(key=lambda r: sp.get(r["model_key"], 0.0), reverse=True)


def load_dispatch_counts(dispatch_csv: Path):
    counts: dict[str, Counter] = {}
    for r in csv.DictReader(dispatch_csv.open()):
        if r["include_in_model"] != "1":
            continue
        counts.setdefault(r["model_key"], Counter())[r["segment"]] += 1
    return counts


def fmt_pct(v: float) -> str:
    if 99.995 <= v < 100.0:
        return "<100%"
    if v >= 99.0:
        return f"{v:.2f}%"
    return f"{v:.1f}%"


# ----------------------------- drawing helpers ------------------------------


def draw_dispatch_bar(ax, y, row, dispatch_counts, height=0.68):
    """Existing stacked dispatch-share breakdown."""
    counts = dispatch_counts.get(row["model_key"], Counter())
    total = sum(counts.values())
    if total <= 0:
        return
    left = 0.0
    for segment in SEGMENT_ORDER:
        raw = counts[segment]
        frac = raw / total
        if frac <= 0:
            continue
        ax.barh(
            y,
            frac,
            left=left,
            height=height,
            color=SEGMENT_COLORS[segment],
            alpha=SEGMENT_ALPHA.get(segment, 1.0),
            edgecolor="white",
            linewidth=0.25,
        )
        if frac >= 0.12:
            alpha = SEGMENT_ALPHA.get(segment, 1.0)
            dark = (segment in OPU_SEGMENTS and alpha >= 0.7) or segment == "rvv_matmul"
            ax.text(
                left + frac / 2,
                y,
                f"{100*frac:.0f}%",
                ha="center",
                va="center",
                fontsize=5.5,
                color=("white" if dark else "black"),
            )
        left += frac


def draw_coverage_bar(ax, y, pct: float, height=0.32, color=OPU_COLOR, rest_color=NONOPU_COLOR):
    """Two-tone bar: OPU coverage vs rest."""
    f = pct / 100.0
    ax.barh(y, f, left=0.0, height=height, color=color, edgecolor="white", linewidth=0.25)
    ax.barh(y, 1.0 - f, left=f, height=height, color=rest_color, edgecolor="white", linewidth=0.25)
    if f >= 0.12:
        ax.text(f / 2, y, f"{pct:.0f}%", ha="center", va="center", fontsize=5.5, color="white")


def visible_segments(rows, dispatch_counts):
    return [s for s in SEGMENT_ORDER if any(dispatch_counts.get(r["model_key"], Counter())[s] > 0 for r in rows)]


def add_dispatch_legend(ax, rows, dispatch_counts):
    segs = visible_segments(rows, dispatch_counts)
    # Dedupe by label — multiple segments can share a label (e.g.
    # encoding_16x16_tile and inline_vopacc both produce a 16×16 VOPACC).
    handles, labels, seen = [], [], set()
    for s in segs:
        lbl = SEGMENT_LABELS[s]
        if lbl in seen:
            continue
        seen.add(lbl)
        handles.append(plt.Rectangle((0, 0), 1, 1, color=SEGMENT_COLORS[s], alpha=SEGMENT_ALPHA.get(s, 1.0)))
        labels.append(lbl)
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=3,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.65,
    )


# ----------------------------- layouts --------------------------------------


def layout_A_stacked(rows, dispatch_counts, metric: str, out_path: Path):
    """Option A: two bars per row — dispatch share on top, coverage below."""
    n = len(rows)
    height = DECOMPOSITION_BASE_HEIGHT + 0.40 * n + 0.2
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, height))
    ys_dispatch = [2 * (n - 1 - i) + 0.35 for i in range(n)]
    ys_coverage = [2 * (n - 1 - i) - 0.35 for i in range(n)]

    for i, row in enumerate(rows):
        draw_dispatch_bar(ax, ys_dispatch[i], row, dispatch_counts, height=0.68)
        pct_s = row.get(f"opu_{metric}_pct", "")
        if pct_s:
            draw_coverage_bar(ax, ys_coverage[i], float(pct_s), height=0.34)

        # Right-side annotation
        opu_d = int(row.get("opu_dispatches") or 0)
        total_d = int(row.get("dispatches") or 0)
        d_pct = 100 * opu_d / total_d if total_d else 0
        params = MODEL_PARAMS.get(row["model_key"], "")
        cov_pct = float(pct_s) if pct_s else 0
        cov_label = "params" if metric == "param" else "FLOPs"
        ax.text(
            1.015,
            ys_dispatch[i] - 0.35,
            f"{fmt_pct(d_pct)} OPU dispatches\n"
            f"{fmt_pct(cov_pct)} OPU {cov_label} (est.)\n"
            f"{opu_d}/{total_d} dispatches, {params} params",
            va="center",
            ha="left",
            fontsize=5.1,
            linespacing=0.95,
        )

    # y ticks at middle of each pair
    ax.set_yticks([2 * (n - 1 - i) for i in range(n)])
    ax.set_yticklabels([r["model"] for r in rows])
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(f"Upper: dispatch share   •   Lower: OPU {cov_label} share (layer-level estimate)")
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    add_dispatch_legend(ax, rows, dispatch_counts)
    fig.tight_layout()
    fig.subplots_adjust(left=0.20, right=0.71, bottom=0.26)
    for ext in ("pdf", "png"):
        fig.savefig(out_path.with_suffix(f".{ext}"))
        print(f"  -> {out_path.with_suffix('.'+ext)}")
    plt.close(fig)


def layout_B_side_by_side(rows, dispatch_counts, metric: str, out_path: Path):
    """Option B: two panels — dispatch share left, coverage right."""
    n = len(rows)
    height = DECOMPOSITION_BASE_HEIGHT + DECOMPOSITION_ROW_HEIGHT * n
    fig, (ax_d, ax_c) = plt.subplots(
        1,
        2,
        figsize=(2 * COLUMN_WIDTH_IN, height),
        gridspec_kw={"width_ratios": [2, 1], "wspace": 0.05},
        sharey=True,
    )
    ys = list(range(n))[::-1]

    # Left: dispatch share
    for y, row in zip(ys, rows):
        draw_dispatch_bar(ax_d, y, row, dispatch_counts)
        opu_d = int(row.get("opu_dispatches") or 0)
        total_d = int(row.get("dispatches") or 0)
        d_pct = 100 * opu_d / total_d if total_d else 0
        ax_d.text(
            1.015, y, f"{fmt_pct(d_pct)}\n{opu_d}/{total_d}", va="center", ha="left", fontsize=5.1, linespacing=0.95
        )
    ax_d.set_yticks(ys)
    ax_d.set_yticklabels([r["model"] for r in rows])
    ax_d.set_xlim(0.0, 1.0)
    ax_d.set_xticks([0.0, 0.5, 1.0])
    ax_d.set_xticklabels(["0", "50%", "100%"])
    ax_d.set_xlabel("Dispatch share")
    ax_d.grid(axis="x", linestyle="--", alpha=0.35)

    # Right: coverage
    for y, row in zip(ys, rows):
        pct_s = row.get(f"opu_{metric}_pct", "")
        if pct_s:
            draw_coverage_bar(ax_c, y, float(pct_s), height=0.55)
            params = MODEL_PARAMS.get(row["model_key"], "")
            ax_c.text(
                1.015, y, f"{fmt_pct(float(pct_s))}\n{params}", va="center", ha="left", fontsize=5.1, linespacing=0.95
            )
    ax_c.set_xlim(0.0, 1.0)
    ax_c.set_xticks([0.0, 0.5, 1.0])
    ax_c.set_xticklabels(["0", "50%", "100%"])
    cov_label = "parameters" if metric == "param" else "FLOPs"
    ax_c.set_xlabel(f"OPU {cov_label} (estimate)")
    ax_c.grid(axis="x", linestyle="--", alpha=0.35)

    add_dispatch_legend(ax_d, rows, dispatch_counts)
    fig.tight_layout()
    fig.subplots_adjust(left=0.11, right=0.93, bottom=0.30)
    for ext in ("pdf", "png"):
        fig.savefig(out_path.with_suffix(f".{ext}"))
        print(f"  -> {out_path.with_suffix('.'+ext)}")
    plt.close(fig)


def layout_C_coverage_only(rows, metric: str, out_path: Path):
    """Option C: compact single-bar-per-model OPU-coverage figure."""
    n = len(rows)
    height = DECOMPOSITION_BASE_HEIGHT + 0.28 * n
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, height))
    ys = list(range(n))[::-1]
    for y, row in zip(ys, rows):
        pct_s = row.get(f"opu_{metric}_pct", "")
        if not pct_s:
            continue
        pct = float(pct_s)
        draw_coverage_bar(ax, y, pct, height=0.6)
        params = MODEL_PARAMS.get(row["model_key"], "")
        ax.text(1.015, y, f"{fmt_pct(pct)}  •  {params} params", va="center", ha="left", fontsize=5.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["model"] for r in rows])
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    cov_label = "parameters" if metric == "param" else "FLOPs"
    ax.set_xlabel(f"Fraction of {cov_label} on OPU-backed layers (layer-type estimate)")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.subplots_adjust(left=0.20, right=0.76)
    for ext in ("pdf", "png"):
        fig.savefig(out_path.with_suffix(f".{ext}"))
        print(f"  -> {out_path.with_suffix('.'+ext)}")
    plt.close(fig)


# ----------------------------- main -----------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--summary", type=Path, default=BENCH_DIR / "per_model_summary.csv")
    p.add_argument("--dispatch", type=Path, default=BENCH_DIR / "model_dispatch_decomposition.csv")
    p.add_argument("--iters-csv", type=Path, default=Path("/tmp/sweep_iters.csv"))
    p.add_argument("--out-dir", type=Path, default=BENCH_DIR / "figures")
    p.add_argument("--include-models", default=None, help="Comma-separated model_keys to include")
    p.add_argument("--rename", default=None, help="Comma-separated key:display overrides")
    p.add_argument(
        "--metric",
        choices=["param", "compute"],
        default="compute",
        help="Coverage metric: 'param' (weights) or 'compute' (MACs)",
    )
    p.add_argument("--layout", choices=["A", "B", "C", "all"], default="all")
    args = p.parse_args()

    include = [s.strip() for s in args.include_models.split(",")] if args.include_models else None
    rows = load_rows(args.summary, include)
    apply_renames(rows, args.rename)
    sort_by_speedup(rows, args.iters_csv)

    dispatch_counts = load_dispatch_counts(args.dispatch)
    apply_paper_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    suffix = args.metric  # "param" or "compute"
    if args.layout in ("A", "all"):
        layout_A_stacked(rows, dispatch_counts, args.metric, args.out_dir / f"per_model_decomposition_with_{suffix}")
    if args.layout in ("B", "all"):
        layout_B_side_by_side(
            rows, dispatch_counts, args.metric, args.out_dir / f"per_model_decomposition_side_by_side_{suffix}"
        )
    if args.layout in ("C", "all"):
        layout_C_coverage_only(rows, args.metric, args.out_dir / f"per_model_{suffix}_only")


if __name__ == "__main__":
    main()
