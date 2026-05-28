#!/usr/bin/env python3
"""Render the planned-vs-observed Gantt for a scheduler-runner trace.

Model- and target-agnostic. Auto-discovers job names and cluster targets
from the trace + schedule, and assigns palette colors automatically. The
job-coloring function accepts an override dict so callers can pin specific
jobs to specific colors for cross-plot consistency.

Builds on tmp/analysis/plot_dispatch_trace.py with two extensions:

  1. Reads the schedule.json alongside the trace CSV so kernel-variant
     tags from compose_multi_schedule.py (kernel_variant: llm/tile/
     layer/megakernel/...) flow through to a hatching overlay on each
     dispatch interval.

  2. Auto-discovers any number of jobs + targets (any model family, any
     CPU/GPU/NPU cluster naming convention).

Usage:
    python tools/perf/plot_planned_vs_observed.py \\
        --trace-csv build/.../trace.csv \\
        --schedule build/.../multi_schedule.json \\
        --out plots/planned_vs_observed.png \\
        --title "<board>: <model summary> with kernel-variant overlay"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

# Hatching pattern per kernel_variant; rendered on top of the job-color fill.
HATCH_BY_VARIANT = {
    "llm": "//",
    "tile": "..",
    "layer": "xx",
    "megakernel": "++",
    "compiled": "",  # default; no hatch.
    "skipped": "//",  # PR 7: skipped chunks render hatched-grey on planned band
    "": "",
}


def load_inputs(trace_csv: Path, schedule_json: Path | None) -> pd.DataFrame:
    df = pd.read_csv(trace_csv)
    df["dispatch_key"] = df["dispatch_key"].astype(str)
    df["job_name"] = df["job_name"].fillna("").astype(str)
    df["target"] = df["target"].astype(str)

    # Join in kernel_variant from the schedule, keyed on the FULL replica
    # dispatch_key (e.g. "mlp3_dispatch_15"). Skipped chunks are
    # auto-tagged kernel_variant="skipped" so the plot can hatched-grey
    # them.
    df["kernel_variant"] = ""
    df["skipped"] = False
    if schedule_json is not None and schedule_json.exists():
        sched = json.loads(schedule_json.read_text())
        sched_dispatches = sched.get("dispatches", {})

        def pick(row):
            cand = f"{row.job_name}_{row.dispatch_key}"
            entry = sched_dispatches.get(cand) or sched_dispatches.get(row.dispatch_key)
            if entry is None:
                return ("", False)
            variant = entry.get("kernel_variant", "")
            skipped = bool(entry.get("skipped", False))
            if skipped and not variant:
                variant = "skipped"
            return (variant, skipped)

        joined = df.apply(pick, axis=1, result_type="expand")
        df["kernel_variant"] = joined[0]
        df["skipped"] = joined[1]
    return df


def build_job_colors(
    job_names: list[str],
    color_map_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    """Assign a tab20 color to each job name. Model-agnostic.

    `color_map_overrides` lets the caller pin specific jobs to specific
    colors (e.g. for cross-plot consistency). Anything not in the override
    map gets the next palette slot, with `mlp<N>` jobs grouped sequentially
    starting at slot 4 (an old visual convention that's still useful when
    the workload contains a numbered MLP family).
    """
    cmap = plt.get_cmap("tab20")
    colors: dict[str, object] = dict(color_map_overrides or {})
    mlps = sorted([j for j in job_names if re.fullmatch(r"mlp\d+", j)], key=lambda s: int(s[3:]))
    for i, j in enumerate(mlps):
        colors.setdefault(j, cmap((4 + i) % cmap.N))
    for j in job_names:
        if j not in colors:
            colors[j] = cmap(len(colors) % cmap.N)
    return colors


def short_label(dispatch_key: str) -> str:
    m = re.search(r"_dispatch_(.+)$", dispatch_key)
    return m.group(1) if m else dispatch_key


def assign_lanes(intervals: list[dict]) -> int:
    intervals.sort(key=lambda x: (x["start_ms"], x["end_ms"], x["dispatch_key"]))
    lane_ends: list[float] = []
    for iv in intervals:
        placed = False
        for li, le in enumerate(lane_ends):
            if iv["start_ms"] >= le - 1e-9:
                iv["lane"] = li
                lane_ends[li] = iv["end_ms"]
                placed = True
                break
        if not placed:
            iv["lane"] = len(lane_ends)
            lane_ends.append(iv["end_ms"])
    return max(1, len(lane_ends))


def make_intervals(df: pd.DataFrame, mode: str, target: str) -> list[dict]:
    if mode == "observed":
        starts = df["start_us"] / 1000.0
        durs = df["run_us"] / 1000.0
    else:
        starts = df["planned_start_us"] / 1000.0
        durs = df["planned_duration_us"] / 1000.0
    out = []
    for row, s, d in zip(df.itertuples(index=False), starts, durs):
        skipped = bool(getattr(row, "skipped", False))
        # Skipped chunks have no observed window — drop them from the
        # observed band entirely. The planned band keeps them rendered
        # (hatched-grey via the kernel_variant overlay) so the user can
        # see what would have run had it not been skipped.
        if skipped and mode == "observed":
            continue
        out.append(
            {
                "dispatch_key": row.dispatch_key,
                "job_name": row.job_name,
                "target": target,
                "kernel_variant": getattr(row, "kernel_variant", "") or "",
                "skipped": skipped,
                "start_ms": float(s),
                "dur_ms": float(d),
                "end_ms": float(s + d),
                "label": short_label(row.dispatch_key),
            }
        )
    return out


def draw_intervals(ax, intervals, band_y0, band_h, num_lanes, job_colors, planned, label_threshold_ms, xlim):
    if not intervals or num_lanes <= 0:
        return
    lane_h = band_h / num_lanes
    for iv in intervals:
        y = band_y0 + iv["lane"] * lane_h
        color = job_colors.get(iv["job_name"], "#cccccc")
        hatch = HATCH_BY_VARIANT.get(iv["kernel_variant"], "")
        rect = mpatches.Rectangle(
            (iv["start_ms"], y),
            max(iv["dur_ms"], 0.0),
            lane_h,
            facecolor=color,
            edgecolor=("black" if not planned else color),
            linewidth=0.6 if not planned else 0.9,
            linestyle=("-" if not planned else "--"),
            alpha=(0.85 if not planned else 0.30),
            hatch=(hatch if not planned else None),
            zorder=4 if not planned else 2,
            clip_on=True,
        )
        ax.add_patch(rect)
        cx = iv["start_ms"] + iv["dur_ms"] / 2.0
        if (xlim is None or (xlim[0] <= cx <= xlim[1])) and iv["dur_ms"] >= label_threshold_ms:
            ax.text(
                cx,
                y + lane_h / 2.0,
                iv["label"],
                ha="center",
                va="center",
                fontsize=7 if planned else 8,
                color=("black" if not planned else "#444"),
                clip_on=True,
            )


def plot(df: pd.DataFrame, out_path: Path, title: str, xlim=None, deadline_ms: float | None = None) -> None:
    PREFERRED_ORDER = ["CPU_P", "CPU_E", "QNN_GPU", "QNN_HTA"]
    targets_in_trace = list(df["target"].unique())
    ordered = [t for t in PREFERRED_ORDER if t in targets_in_trace]
    ordered += [t for t in sorted(targets_in_trace) if t not in PREFERRED_ORDER]
    rows = [(t, m) for t in ordered for m in ("observed", "planned")]
    bands = []
    for tgt, mode in rows:
        sub = df[df["target"] == tgt]
        ivs = make_intervals(sub, mode, tgt)
        if xlim is not None:
            x0, x1 = xlim
            ivs = [iv for iv in ivs if iv["end_ms"] >= x0 and iv["start_ms"] <= x1]
        nlanes = assign_lanes(ivs)
        bands.append((tgt, mode, ivs, nlanes))

    # Auto-scale figure size based on lane count (height) and time extent
    # (width). Caps prevent matplotlib from refusing to render at extreme
    # aspect ratios on very long traces.
    lane_h_in = 0.40
    fig_h = 2.0 + sum(lane_h_in * max(1, n) for _, _, _, n in bands)
    x_max = max((iv["end_ms"] for _, _, ivs, _ in bands for iv in ivs), default=1.0)
    if xlim is not None:
        x_max = xlim[1] - xlim[0]
    fig_w = max(14.0, min(48.0, 6.0 + x_max * 0.18))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    job_names = sorted(df["job_name"].unique())
    job_colors = build_job_colors(job_names)

    BAND_GAP = 0.4
    yticks = []
    yticklabels = []
    cur_y = 0.0
    for tgt, mode, ivs, nlanes in bands:
        band_h = max(0.6, 0.55 * max(1, nlanes))
        ax.axhspan(cur_y, cur_y + band_h, color="#808080", alpha=0.06, zorder=0)
        draw_intervals(
            ax,
            ivs,
            cur_y,
            band_h,
            nlanes,
            job_colors,
            planned=(mode == "planned"),
            label_threshold_ms=(0.4 if mode == "observed" else 1.2),
            xlim=xlim,
        )
        yticks.append(cur_y + band_h / 2.0)
        yticklabels.append(f"{tgt} {mode}")
        cur_y += band_h + BAND_GAP

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.invert_yaxis()
    ax.set_xlabel("Time (ms)")
    ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        x_max = max((iv["end_ms"] for _, _, ivs, _ in bands for iv in ivs), default=1.0)
        ax.set_xlim(-0.5, x_max * 1.02)
    ax.set_ylim(cur_y - BAND_GAP, -0.2)

    # Deadline overlay (PR 7 of the rosy-sundae plan): a vertical red
    # dashed line at deadline_ms with a faded red shade for time past
    # the deadline. Helps the user see at a glance whether the dominant
    # job met its real-time constraint.
    if deadline_ms is not None and deadline_ms > 0:
        x_lo, x_hi = ax.get_xlim()
        y_lo, y_hi = ax.get_ylim()
        # Faded shade past the deadline.
        if x_hi > deadline_ms:
            ax.axvspan(deadline_ms, x_hi, color="#cc0000", alpha=0.06, zorder=0)
        # Vertical dashed line at the deadline.
        ax.axvline(deadline_ms, color="#cc0000", linestyle="--", linewidth=1.5, alpha=0.85, zorder=5)
        ax.text(
            deadline_ms,
            y_hi + 0.05,
            f"deadline {deadline_ms:.0f} ms",
            ha="left",
            va="bottom",
            color="#cc0000",
            fontsize=10,
            fontweight="bold",
        )

    # Legends.
    win_handles = [
        mpatches.Patch(facecolor="#e8e8e8", edgecolor="#888", linestyle="--", label="planned"),
        mpatches.Patch(facecolor="#cccccc", edgecolor="black", linestyle="-", label="observed"),
    ]
    leg1 = ax.legend(handles=win_handles, title="Window type", loc="upper left", fontsize=10, title_fontsize=10)
    ax.add_artist(leg1)
    job_handles = [mpatches.Patch(facecolor=job_colors[j], edgecolor="black", label=j) for j in job_names]
    variants = sorted({iv["kernel_variant"] for _, _, ivs, _ in bands for iv in ivs if iv["kernel_variant"]})
    var_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", hatch=HATCH_BY_VARIANT.get(v, ""), label=v)
        for v in variants
    ]
    handles = job_handles + (
        []
        if not var_handles
        else [mpatches.Patch(facecolor="none", edgecolor="none", label="— kernel variant —")] + var_handles
    )
    ax.legend(handles=handles, title="Jobs / kernels", loc="upper right", fontsize=9, title_fontsize=10)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--trace-csv", required=True, type=Path)
    p.add_argument(
        "--schedule",
        type=Path,
        default=None,
        help="Optional schedule.json. When set, kernel_variant " "tags overlay as hatching on each interval.",
    )
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--title", default="Cluster schedule by target: planned vs observed")
    p.add_argument("--zoom-ms", type=float, default=None)
    p.add_argument(
        "--deadline-ms",
        type=float,
        default=None,
        help="Robotics-deadline overlay. Renders a red dashed "
        "vertical line at the deadline + a faded shade "
        "past it. Surfaces whether the dominant job met "
        "its hard real-time constraint.",
    )
    args = p.parse_args()
    df = load_inputs(args.trace_csv, args.schedule)
    plot(df, args.out, args.title, deadline_ms=args.deadline_ms)
    print(f"wrote {args.out}")
    if args.zoom_ms is not None:
        zoom_out = args.out.with_name(args.out.stem + "_zoom" + args.out.suffix)
        plot(df, zoom_out, args.title + " (zoom)", xlim=(-0.5, args.zoom_ms), deadline_ms=args.deadline_ms)
        print(f"wrote {zoom_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
