#!/usr/bin/env python3
"""Plot per-model Saturn OPU dispatch decomposition."""

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


def load_summary(path: Path) -> list[dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def load_dispatch_counts(path: Path) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["include_in_model"] != "1":
                continue
            counts.setdefault(row["model_key"], Counter())[row["segment"]] += 1
    return counts


def fmt_pct(value: float) -> str:
    if 99.995 <= value < 100.0:
        return "<100%"
    if value >= 99.0:
        return f"{value:.2f}%"
    return f"{value:.1f}%"


def load_runtime_cycles(path: Path) -> dict[str, dict[tuple[str, int], int]]:
    """Load per-model per-(symbol, ordinal) measured cycles from the profile
    sweep output. Returns {model_key: {(symbol, ordinal): total_cycles}}."""
    out: dict[str, dict[tuple[str, int], int]] = {}
    if not path.exists():
        return out
    with open(path) as f:
        for row in csv.DictReader(f):
            model = row["model"]
            # Only profile-phase rows (variant = 'opu_prof') contribute.
            if row.get("variant") != "opu_prof":
                continue
            try:
                ordinal = int(row["ordinal"])
                cyc = int(row["total_cycles"])
            except (ValueError, KeyError):
                continue
            sym = row.get("symbol", "")
            out.setdefault(model, {})[(sym, ordinal)] = cyc
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=BENCH_DIR / "per_model_summary.csv")
    parser.add_argument("--dispatch", type=Path, default=BENCH_DIR / "model_dispatch_decomposition.csv")
    parser.add_argument("--out-dir", type=Path, default=BENCH_DIR / "figures")
    parser.add_argument("--figsize-w", type=float, default=COLUMN_WIDTH_IN)
    parser.add_argument("--figsize-h", type=float)
    parser.add_argument(
        "--compute-share",
        choices=["static", "runtime"],
        default="static",
        help="'static' (default) uses the analytical compute_pct from per_model_summary.csv. "
        "'runtime' replaces it with measured OPU-cycle fraction from --runtime-csv.",
    )
    parser.add_argument(
        "--runtime-csv",
        type=Path,
        default=Path("/tmp/sweep_cycles.csv"),
        help="Path to the profile-phase cycle dump (only used with --compute-share=runtime).",
    )
    parser.add_argument(
        "--out-name",
        default="per_model_decomposition",
        help="Output basename (without extension) under --out-dir.",
    )
    parser.add_argument(
        "--include-models",
        default=None,
        help=("Comma-separated list of model_key values to keep " "(in summary-CSV order). Default: all rows."),
    )
    parser.add_argument(
        "--rename",
        default=None,
        help=("Comma-separated key:display_name overrides for the y-tick " "labels, e.g. 'opu_bench_vit_small:ViT'."),
    )
    parser.add_argument(
        "--iters-csv",
        type=Path,
        default=Path("/tmp/sweep_iters.csv"),
        help=(
            "Path to the FireSim sweep iter CSV. Used to compute the "
            "per-model OPU-vs-RVV speedup displayed on the right side "
            "of each bar. Falls back silently if missing."
        ),
    )
    args = parser.parse_args()

    MODEL_PARAMS: dict[str, str] = {
        "mlp": "320",
        "mlp_wide": "2.2K",
        "mlp_fast": "2.2M",
        "opu_bench_large_mlp": "1.4M",
        "opu_bench_vit_small": "399K",
        "opu_bench_vit": "75.6M",
        "opu_bench_hybrid": "475K",
        "opu_bench_convnet": "697K",
        "dronet": "312K",
        "yolov8_nano": "3.2M",
        "tinyllama": "1.1B",
    }

    rows = load_summary(args.summary)
    if not rows:
        raise SystemExit(f"empty summary: {args.summary}")

    # --- Model whitelist (keep summary-CSV order) -----------------------
    if args.include_models:
        keep = {k.strip() for k in args.include_models.split(",") if k.strip()}
        missing = keep - {r["model_key"] for r in rows}
        if missing:
            raise SystemExit(f"--include-models: unknown model_keys: {sorted(missing)}")
        rows = [r for r in rows if r["model_key"] in keep]
        if not rows:
            raise SystemExit("--include-models filtered out every row")

    # --- Order by speedup (descending) if sweep data available -----------
    if args.iters_csv.exists():
        last_avg: dict[tuple[str, str], int] = {}
        with open(args.iters_csv) as f:
            for r in csv.DictReader(f):
                avg = (r.get("avg") or "").strip()
                if avg.isdigit():
                    last_avg[(r["model"], r["variant"])] = int(avg)
        speedups: dict[str, float] = {}
        for (model, variant), v in list(last_avg.items()):
            if variant != "opu":
                continue
            rv = last_avg.get((model, "rvv"))
            if rv and v:
                speedups[model] = rv / v
        rows.sort(key=lambda r: speedups.get(r["model_key"], 0.0), reverse=True)

    # --- Display-name overrides -----------------------------------------
    if args.rename:
        for pair in args.rename.split(","):
            if ":" not in pair:
                raise SystemExit(f"--rename entries must be key:name (got '{pair}')")
            key, new_name = pair.split(":", 1)
            key, new_name = key.strip(), new_name.strip()
            matched = False
            for r in rows:
                if r["model_key"] == key:
                    r["model"] = new_name
                    matched = True
                    break
            if not matched:
                raise SystemExit(f"--rename: model_key '{key}' not in the kept rows")

    # --- Load measured OPU-vs-RVV speedups from the sweep CSV ----------
    # For each model_key, keep the LAST row per variant with a numeric
    # avg (the freshest measurement). speedup = rvv_avg / opu_avg.
    speedups: dict[str, float] = {}
    if args.iters_csv.exists():
        last_avg: dict[tuple[str, str], int] = {}
        with open(args.iters_csv) as f:
            for r in csv.DictReader(f):
                avg = (r.get("avg") or "").strip()
                if not avg.isdigit():
                    continue
                last_avg[(r["model"], r["variant"])] = int(avg)
        for (model, variant), v in list(last_avg.items()):
            if variant != "opu":
                continue
            rv = last_avg.get((model, "rvv"))
            if rv and v:
                speedups[model] = rv / v

    dispatch_counts = load_dispatch_counts(args.dispatch)

    # Runtime-cycle compute share: need the per-dispatch CSV to know which
    # symbol belongs to which segment, then aggregate cycles by segment.
    runtime_share: dict[str, float] = {}
    if args.compute_share == "runtime":
        cycles_by_model = load_runtime_cycles(args.runtime_csv)
        # Build symbol → segment lookup from the dispatch decomposition CSV.
        sym_segment: dict[tuple[str, str], str] = {}
        with open(args.dispatch) as f:
            for r in csv.DictReader(f):
                sym_segment[(r["model_key"], r["symbol"])] = r.get("segment", "")
        for model, cyc_map in cycles_by_model.items():
            total = 0
            opu = 0
            for (sym, _ord), c in cyc_map.items():
                total += c
                seg = sym_segment.get((model, sym), "")
                if seg in OPU_SEGMENTS:
                    opu += c
            if total:
                runtime_share[model] = 100.0 * opu / total

    apply_paper_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    height = args.figsize_h or (DECOMPOSITION_BASE_HEIGHT + DECOMPOSITION_ROW_HEIGHT * len(rows))
    fig, ax = plt.subplots(figsize=(args.figsize_w, height))
    y_positions = list(range(len(rows)))[::-1]

    for y, row in zip(y_positions, rows):
        model_key = row["model_key"]
        counts = dispatch_counts.get(model_key, Counter())
        # Include runtime-support helpers in the denominator: for tiny
        # models the linked-in library code dominates the binary, and
        # hiding that is misleading. Rendered in its own hatched segment.
        rt = int(row.get("runtime_support") or 0)
        if rt > 0:
            counts = Counter(counts)  # don't mutate shared dict
            counts["runtime_support"] = rt
        total = sum(counts.values())
        if total <= 0:
            raise SystemExit(f"no dispatch rows for {model_key} in {args.dispatch}")
        # Aggregate consecutive SEGMENT_ORDER entries that share a label
        # into a single draw call, so segments like quantize/dequantize/
        # requantize (all labeled "Quant / Dequant") render as one
        # contiguous block instead of three white-bordered sub-slices.
        grouped: list[tuple[str, float]] = []  # (canonical_segment, fraction)
        for segment in SEGMENT_ORDER:
            raw = counts[segment]
            if raw <= 0:
                continue
            frac = raw / total
            if grouped and SEGMENT_LABELS.get(segment) == SEGMENT_LABELS.get(grouped[-1][0]):
                prev_seg, prev_frac = grouped[-1]
                grouped[-1] = (prev_seg, prev_frac + frac)
            else:
                grouped.append((segment, frac))

        left = 0.0
        for segment, frac in grouped:
            ax.barh(
                y,
                frac,
                left=left,
                height=0.68,
                color=SEGMENT_COLORS[segment],
                alpha=SEGMENT_ALPHA.get(segment, 1.0),
                edgecolor="white",
                linewidth=0.25,
            )
            # OPU segments always get a label (even thin ones like
            # MLP's 9.1%) since the OPU share is the key number the
            # reader is looking for. Non-OPU segments only label above
            # 12% to keep the bar uncluttered.
            show_label = frac >= 0.12 or segment in OPU_SEGMENTS
            if show_label:
                alpha = SEGMENT_ALPHA.get(segment, 1.0)
                dark_bar = (segment in OPU_SEGMENTS and alpha >= 0.7) or segment == "rvv_matmul"
                text_color = "white" if dark_bar else "black"
                # One decimal when the slice is very thin so "9.1%"
                # reads as itself rather than a rounded "9%".
                fmt = f"{100*frac:.1f}%" if frac < 0.10 else f"{100*frac:.0f}%"
                ax.text(left + frac / 2, y, fmt, ha="center", va="center", fontsize=5.5, color=text_color)
            left += frac

        # Right-side annotation: just the raw work-unit count and the
        # model's parameter count. The OPU % appears inline in the
        # colored bar, so no need to repeat it here.
        dispatches = sum(counts.values())
        opu_dispatches = sum(counts[s] for s in counts if s in OPU_SEGMENTS)
        params_str = MODEL_PARAMS.get(model_key, "")
        ann = f"{opu_dispatches}/{dispatches} work units"
        if params_str:
            ann += f"\n{params_str} params"
        ax.text(
            1.015,
            y,
            ann,
            va="center",
            ha="left",
            fontsize=5.4,
            linespacing=0.95,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([row["model"] for row in rows])
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Work-unit share (fused dispatches + linked runtime support)")
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    visible_segments = [
        segment
        for segment in SEGMENT_ORDER
        if any(
            dispatch_counts.get(row["model_key"], Counter())[segment] > 0
            or (segment == "runtime_support" and int(row.get("runtime_support") or 0) > 0)
            for row in rows
        )
    ]
    # Dedupe: multiple segments can share a label (e.g. encoding_16x16_tile
    # and inline_vopacc both emit a 16×16 outer-product VOPACC — we label
    # them identically so the legend shows one entry).
    handles, labels, seen = [], [], set()
    for segment in visible_segments:
        lbl = SEGMENT_LABELS[segment]
        if lbl in seen:
            continue
        seen.add(lbl)
        handles.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                color=SEGMENT_COLORS[segment],
                alpha=SEGMENT_ALPHA.get(segment, 1.0),
            )
        )
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

    fig.tight_layout()
    fig.subplots_adjust(left=0.20, right=0.73, bottom=0.31)
    for ext in ("pdf", "png"):
        out = args.out_dir / f"{args.out_name}.{ext}"
        fig.savefig(out)
        print(f"  -> {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
