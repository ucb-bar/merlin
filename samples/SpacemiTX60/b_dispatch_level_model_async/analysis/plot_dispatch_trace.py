#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class Interval:
    dispatch_key: str
    job_name: str
    target: str
    start_ms: float
    dur_ms: float
    end_ms: float
    lane: int = 0
    label: str = ""


def dispatch_short_label(dispatch_key: str) -> str:
    m = re.search(r"_dispatch_(.+)$", dispatch_key)
    return m.group(1) if m else dispatch_key


def build_job_colors(job_names: list[str]) -> dict[str, object]:
    cmap = plt.get_cmap("tab20")
    colors: dict[str, object] = {}

    if "dronet" in job_names:
        colors["dronet"] = cmap(0)

    mlps = sorted(
        [j for j in job_names if re.fullmatch(r"mlp\d+", j)],
        key=lambda s: int(s[3:]),
    )
    start_idx = 4
    for i, job in enumerate(mlps):
        colors[job] = cmap((start_idx + i) % cmap.N)

    for job in job_names:
        if job not in colors:
            colors[job] = cmap(len(colors) % cmap.N)

    return colors


def assign_lanes(intervals: list[Interval], epsilon_ms: float = 1e-9) -> int:
    if not intervals:
        return 1

    intervals.sort(key=lambda x: (x.start_ms, x.end_ms, x.dispatch_key))
    lane_ends: list[float] = []

    for interval in intervals:
        placed = False
        for lane_idx, lane_end in enumerate(lane_ends):
            if interval.start_ms >= lane_end - epsilon_ms:
                interval.lane = lane_idx
                lane_ends[lane_idx] = interval.end_ms
                placed = True
                break
        if not placed:
            interval.lane = len(lane_ends)
            lane_ends.append(interval.end_ms)

    return max(1, len(lane_ends))


def make_intervals(df: pd.DataFrame, mode: str, target: str) -> list[Interval]:
    if mode == "observed":
        starts = df["start_us"] / 1000.0
        durs = df["run_us"] / 1000.0
    elif mode == "planned":
        starts = df["planned_start_us"] / 1000.0
        durs = df["planned_duration_us"] / 1000.0
    else:
        raise ValueError(mode)

    out: list[Interval] = []
    for row, start_ms, dur_ms in zip(df.itertuples(index=False), starts, durs):
        out.append(
            Interval(
                dispatch_key=row.dispatch_key,
                job_name=row.job_name,
                target=target,
                start_ms=float(start_ms),
                dur_ms=float(dur_ms),
                end_ms=float(start_ms + dur_ms),
                label=dispatch_short_label(row.dispatch_key),
            )
        )
    return out


def add_band_background(ax, y0: float, y1: float) -> None:
    ax.axhspan(y0, y1, color="#808080", alpha=0.08, zorder=0)


def draw_intervals(
    ax,
    intervals: list[Interval],
    band_y0: float,
    band_h: float,
    total_lanes: int,
    job_colors: dict[str, object],
    *,
    planned: bool,
    min_tick_ms: float,
    planned_label_ms: float,
    observed_label_ms: float,
) -> None:
    if not intervals:
        return

    lane_gap = min(0.06, band_h * 0.12)
    inner_pad = min(0.10, band_h * 0.18)
    usable_h = max(0.1, band_h - 2 * inner_pad)
    lane_h = max(0.05, (usable_h - max(0, total_lanes - 1) * lane_gap) / total_lanes)

    for interval in intervals:
        lane_y = band_y0 + inner_pad + interval.lane * (lane_h + lane_gap)
        color = job_colors[interval.job_name]

        if planned and interval.dur_ms < min_tick_ms:
            x = interval.start_ms + 0.5 * interval.dur_ms
            ax.vlines(
                x,
                lane_y,
                lane_y + lane_h,
                color=color,
                linewidth=1.4,
                linestyles=(0, (2, 2)),
                alpha=0.95,
                zorder=3,
            )
            continue

        rect = mpatches.Rectangle(
            (interval.start_ms, lane_y),
            max(interval.dur_ms, 0.0),
            lane_h,
            facecolor=color,
            edgecolor="black" if not planned else color,
            linewidth=0.8 if not planned else 1.0,
            linestyle="--" if planned else "-",
            alpha=0.28 if planned else 0.82,
            zorder=2 if planned else 4,
        )
        ax.add_patch(rect)

        min_label_ms = planned_label_ms if planned else observed_label_ms
        if interval.dur_ms < min_label_ms:
            continue

        fontsize = 8 if not planned else 7
        rotation = 0
        if interval.dur_ms < 0.8:
            fontsize = 6
            rotation = 90 if not planned else 0

        ax.text(
            interval.start_ms + interval.dur_ms / 2.0,
            lane_y + lane_h / 2.0,
            interval.label,
            ha="center",
            va="center",
            fontsize=fontsize,
            rotation=rotation,
            color="black",
            clip_on=True,
            zorder=5,
        )


def plot_cluster_schedule(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    xlim: tuple[float, float] | None = None,
) -> None:
    job_order = sorted(df["job_name"].unique(), key=lambda s: (s != "dronet", s))
    job_colors = build_job_colors(job_order)

    band_names = [
        ("CPU_E", "observed"),
        ("CPU_E", "planned"),
        ("CPU_P", "observed"),
        ("CPU_P", "planned"),
    ]

    intervals_by_band: dict[tuple[str, str], list[Interval]] = {}
    lanes_by_band: dict[tuple[str, str], int] = {}

    for target, mode in band_names:
        band_df = df[df["target"] == target].copy()
        intervals = make_intervals(band_df, mode, target)
        lanes = assign_lanes(intervals)
        intervals_by_band[(target, mode)] = intervals
        lanes_by_band[(target, mode)] = lanes

    band_h = 1.0
    band_gap = 0.38
    total_h = len(band_names) * band_h + (len(band_names) - 1) * band_gap

    fig_h = 10 if xlim is None else 8.5
    fig_w = 24 if xlim is None else 18
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    y_positions: dict[tuple[str, str], float] = {}
    current_y = total_h - band_h
    for band in band_names:
        y_positions[band] = current_y
        current_y -= band_h + band_gap

    for target, mode in band_names:
        y0 = y_positions[(target, mode)]
        y1 = y0 + band_h
        add_band_background(ax, y0, y1)
        draw_intervals(
            ax,
            intervals_by_band[(target, mode)],
            y0,
            band_h,
            lanes_by_band[(target, mode)],
            job_colors,
            planned=(mode == "planned"),
            min_tick_ms=0.18,
            planned_label_ms=1.25,
            observed_label_ms=0.38,
        )

    yticks = []
    ylabels = []
    for target, mode in band_names:
        yticks.append(y_positions[(target, mode)] + band_h / 2.0)
        ylabels.append(f"{target} {mode}")

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=17, fontweight="bold")
    ax.set_xlabel("Time (ms)", fontsize=16)
    ax.set_title(title, fontsize=24, pad=18)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        max_end = max(
            float((df["complete_us"] / 1000.0).max()),
            float(((df["planned_start_us"] + df["planned_duration_us"]) / 1000.0).max()),
        )
        ax.set_xlim(-2.0, max_end * 1.04)

    ax.set_ylim(-0.25, total_h + 0.25)

    window_handles = [
        mpatches.Patch(
            facecolor="white",
            edgecolor="gray",
            linestyle="--",
            linewidth=1.2,
            label="planned",
        ),
        mpatches.Patch(
            facecolor="white",
            edgecolor="black",
            linestyle="-",
            linewidth=1.2,
            label="observed",
        ),
    ]
    legend1 = ax.legend(
        handles=window_handles,
        title="Window type",
        loc="upper left",
        fontsize=11,
        title_fontsize=11,
        frameon=True,
    )
    ax.add_artist(legend1)

    job_handles = [mpatches.Patch(facecolor=job_colors[j], edgecolor="black", label=j) for j in job_order]
    ax.legend(
        handles=job_handles,
        title="Jobs",
        loc="upper right",
        fontsize=10,
        title_fontsize=11,
        frameon=True,
        ncol=1,
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_inputs(trace_csv: Path, schedule_json: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(trace_csv)

    required = {
        "dispatch_key",
        "job_name",
        "target",
        "planned_start_us",
        "planned_duration_us",
        "eligible_us",
        "complete_us",
        "residency_us",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in trace CSV: {sorted(missing)}")

    if schedule_json and schedule_json.exists():
        sched = pd.read_json(schedule_json)
        if "dispatches" in sched:
            pass

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace-csv",
        type=Path,
        default=Path("/scratch2/agustin/merlin/tmp/analysis/run_out/run_trace.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis/plots"),
    )
    parser.add_argument(
        "--zoom-ms",
        type=float,
        default=None,
        help="Optional x-axis max in ms for a zoomed plot.",
    )
    args = parser.parse_args()

    df = load_inputs(args.trace_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    full_out = args.out_dir / "cluster_schedule_by_target_planned_vs_observed.png"
    plot_cluster_schedule(
        df,
        full_out,
        title="Cluster schedule by target: planned vs observed",
        xlim=None,
    )

    print(f"Wrote:\n  {full_out}")

    if args.zoom_ms is not None:
        zoom_out = args.out_dir / "cluster_schedule_by_target_planned_vs_observed_zoom.png"
        plot_cluster_schedule(
            df,
            zoom_out,
            title="Cluster schedule by target: planned vs observed (zoom)",
            xlim=(-0.5, args.zoom_ms),
        )
        print(f"  {zoom_out}")


if __name__ == "__main__":
    main()
