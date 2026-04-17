#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class Interval:
    name: str
    target: str
    start_ms: float
    dur_ms: float
    end_ms: float
    lane: int = 0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def require_columns(rows: list[dict[str, str]], required: set[str], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows found in {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")


def successful_sweep_rows(path: Path) -> list[dict[str, str]]:
    rows = read_rows(path)
    require_columns(
        rows,
        {
            "exit_code",
            "csv_parse_ok",
            "mlp_hz",
            "dronet_sensor_hz",
            "dronet_inflight",
            "mlp_inflight",
            "core_mask",
            "d_lat_avg_ms",
            "m_lat_avg_ms",
            "dronet_total",
            "mlp_total",
        },
        path,
    )
    return [row for row in rows if row["exit_code"] == "0" and row["csv_parse_ok"] == "1"]


def select_config(
    rows: list[dict[str, str]],
    *,
    mlp_hz: float,
    dronet_sensor_hz: float,
    dronet_inflight: int,
    mlp_inflight: int,
    core_mask: str,
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if to_float(row["mlp_hz"]) == mlp_hz
        and to_float(row["dronet_sensor_hz"]) == dronet_sensor_hz
        and int(row["dronet_inflight"]) == dronet_inflight
        and int(row["mlp_inflight"]) == mlp_inflight
        and row["core_mask"].lower() == core_mask.lower()
    ]


def aggregate_config(rows: list[dict[str, str]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot aggregate an empty row set")
    metric_names = [
        "mlp_hz",
        "dronet_sensor_hz",
        "duration_s",
        "warmup_s",
        "dronet_total",
        "mlp_total",
        "mlp_misses",
        "d_lat_avg_ms",
        "d_lat_p50_ms",
        "d_lat_p99_ms",
        "m_lat_avg_ms",
        "m_lat_p50_ms",
        "m_lat_p99_ms",
        "m_jitter_avg_ms",
        "m_jitter_p99_ms",
        "m_fresh",
        "m_stale",
    ]
    return {name: mean([to_float(row.get(name)) for row in rows]) for name in metric_names}


def reference_mlp_roots(path: Path) -> list[tuple[str, float]]:
    rows = read_rows(path)
    require_columns(rows, {"job_name", "dispatch_id", "target", "planned_start_us"}, path)
    roots = [
        (row["job_name"], to_float(row["planned_start_us"]) / 1000.0)
        for row in rows
        if row["target"] == "CPU_E" and row["job_name"].startswith("mlp") and row["dispatch_id"] == "0"
    ]
    roots.sort(key=lambda item: item[1])
    if not roots:
        raise ValueError(f"No CPU_E MLP root dispatches found in {path}")
    return roots


def assign_lanes(intervals: list[Interval], epsilon_ms: float = 1e-9) -> int:
    intervals.sort(key=lambda item: (item.start_ms, item.end_ms, item.name))
    lane_ends: list[float] = []
    for interval in intervals:
        for lane_idx, lane_end in enumerate(lane_ends):
            if interval.start_ms >= lane_end - epsilon_ms:
                interval.lane = lane_idx
                lane_ends[lane_idx] = interval.end_ms
                break
        else:
            interval.lane = len(lane_ends)
            lane_ends.append(interval.end_ms)
    return max(1, len(lane_ends))


def intervals_from_rate(
    *,
    name_prefix: str,
    target: str,
    period_ms: float,
    dur_ms: float,
    xlim_ms: float,
) -> list[Interval]:
    if period_ms <= 0 or dur_ms <= 0:
        return []
    intervals = []
    idx = 0
    start_ms = 0.0
    while start_ms <= xlim_ms:
        name = name_prefix if name_prefix == "dronet" else f"{name_prefix}{idx}"
        intervals.append(
            Interval(
                name=name,
                target=target,
                start_ms=start_ms,
                dur_ms=dur_ms,
                end_ms=start_ms + dur_ms,
            )
        )
        idx += 1
        start_ms = idx * period_ms
    return intervals


def intervals_from_reference_roots(
    roots: list[tuple[str, float]],
    *,
    dur_ms: float,
    target: str = "CPU_E",
) -> list[Interval]:
    return [
        Interval(
            name=name,
            target=target,
            start_ms=start_ms,
            dur_ms=dur_ms,
            end_ms=start_ms + dur_ms,
        )
        for name, start_ms in roots
    ]


def draw_intervals(ax, intervals: list[Interval], y0: float, band_h: float, color: object) -> None:
    lane_count = assign_lanes(intervals)
    lane_gap = min(0.06, band_h * 0.12)
    inner_pad = min(0.10, band_h * 0.18)
    usable_h = max(0.1, band_h - 2 * inner_pad)
    lane_h = max(0.05, (usable_h - max(0, lane_count - 1) * lane_gap) / lane_count)

    for interval in intervals:
        lane_y = y0 + inner_pad + interval.lane * (lane_h + lane_gap)
        rect = mpatches.Rectangle(
            (interval.start_ms, lane_y),
            interval.dur_ms,
            lane_h,
            facecolor=color,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.82,
            zorder=4,
        )
        ax.add_patch(rect)
        if interval.dur_ms >= 0.75:
            ax.text(
                interval.start_ms + interval.dur_ms / 2.0,
                lane_y + lane_h / 2.0,
                interval.name,
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                clip_on=True,
                zorder=5,
            )


def plot_timeline(
    *,
    cpu_e_intervals: list[Interval],
    cpu_p_intervals: list[Interval],
    title: str,
    subtitle: str,
    out_path: Path,
    xlim_ms: float,
) -> None:
    band_h = 1.0
    band_gap = 0.38
    total_h = 2 * band_h + band_gap
    y_positions = {"CPU_E observed": band_h + band_gap, "CPU_P observed": 0.0}

    cmap = plt.get_cmap("tab20")
    colors = {"dronet": cmap(0), "mlp": cmap(4)}

    fig, ax = plt.subplots(figsize=(18, 5.2))
    for y0 in y_positions.values():
        ax.axhspan(y0, y0 + band_h, color="#808080", alpha=0.08, zorder=0)

    draw_intervals(ax, cpu_e_intervals, y_positions["CPU_E observed"], band_h, colors["mlp"])
    draw_intervals(ax, cpu_p_intervals, y_positions["CPU_P observed"], band_h, colors["dronet"])

    ax.set_yticks([y_positions["CPU_E observed"] + band_h / 2.0, y_positions["CPU_P observed"] + band_h / 2.0])
    ax.set_yticklabels(["CPU_E observed", "CPU_P observed"], fontsize=15, fontweight="bold")
    ax.set_xlabel("Time (ms)", fontsize=14)
    ax.set_title(title, fontsize=20, pad=24)
    ax.text(0.5, 1.02, subtitle, ha="center", va="bottom", transform=ax.transAxes, fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_xlim(-0.5, xlim_ms)
    ax.set_ylim(-0.25, total_h + 0.25)

    handles = [
        mpatches.Patch(facecolor=colors["dronet"], edgecolor="black", label="dronet"),
        mpatches.Patch(facecolor=colors["mlp"], edgecolor="black", label="mlp"),
    ]
    ax.legend(handles=handles, title="Jobs", loc="upper right", fontsize=10, title_fontsize=11, frameon=True)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(out_path: Path, aggregate: dict[str, float], selected_rows: list[dict[str, str]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        fieldnames = ["selected_rows", *aggregate.keys()]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"selected_rows": len(selected_rows), **aggregate})


def main() -> int:
    root = repo_root_from_script()
    default_base = root / "benchmarks/SpacemiTX60/baseline_dual_model_async"

    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-csv", type=Path, default=default_base / "results/sweep.csv")
    parser.add_argument(
        "--reference-trace",
        type=Path,
        default=root / "samples/SpacemiTX60/dispatch_scheduler/analysis/reference_trace.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=default_base / "results/plots")
    parser.add_argument("--xlim-ms", type=float, default=None)
    args = parser.parse_args()

    rows = successful_sweep_rows(args.sweep_csv)
    selected = select_config(
        rows,
        mlp_hz=40.0,
        dronet_sensor_hz=15.0,
        dronet_inflight=1,
        mlp_inflight=1,
        core_mask="0x1",
    )
    if not selected:
        raise ValueError("No successful archived rows found for MLP 40 Hz, dronet 15 Hz, inflight 1, core_mask 0x1")

    aggregate = aggregate_config(selected)
    roots = reference_mlp_roots(args.reference_trace)
    ref_xlim_ms = max(start_ms for _, start_ms in roots) + max(aggregate["m_lat_avg_ms"], 1.0) + 2.0
    xlim_ms = args.xlim_ms if args.xlim_ms is not None else ref_xlim_ms

    mlp_period_ms = 1000.0 / aggregate["mlp_hz"]
    dronet_period_ms = 1000.0 / aggregate["dronet_sensor_hz"]
    archived_cpu_e = intervals_from_rate(
        name_prefix="mlp",
        target="CPU_E",
        period_ms=mlp_period_ms,
        dur_ms=aggregate["m_lat_avg_ms"],
        xlim_ms=xlim_ms,
    )
    archived_cpu_p = intervals_from_rate(
        name_prefix="dronet",
        target="CPU_P",
        period_ms=dronet_period_ms,
        dur_ms=aggregate["d_lat_avg_ms"],
        xlim_ms=xlim_ms,
    )
    plot_timeline(
        cpu_e_intervals=archived_cpu_e,
        cpu_p_intervals=archived_cpu_p,
        title="Baseline dual-model timeline, archived-rate reconstruction",
        subtitle=("MLP 40 Hz, dronet 15 Hz, 1 core, inflight 1; block widths use archived aggregate average latency"),
        out_path=args.out_dir / "baseline_timeline_archived_mlp40hz_dronet15hz.png",
        xlim_ms=xlim_ms,
    )

    reconstructed_cpu_e = intervals_from_reference_roots(roots, dur_ms=aggregate["m_lat_avg_ms"])
    reconstructed_cpu_p = intervals_from_rate(
        name_prefix="dronet",
        target="CPU_P",
        period_ms=dronet_period_ms,
        dur_ms=aggregate["d_lat_avg_ms"],
        xlim_ms=xlim_ms,
    )
    plot_timeline(
        cpu_e_intervals=reconstructed_cpu_e,
        cpu_p_intervals=reconstructed_cpu_p,
        title="Baseline dual-model timeline, 500 Hz MLP reconstruction",
        subtitle=(
            "CPU_E MLP cadence from dispatch_scheduler reference_trace; "
            "block widths use archived baseline aggregate average latency"
        ),
        out_path=args.out_dir / "baseline_timeline_reconstructed_mlp500hz.png",
        xlim_ms=xlim_ms,
    )

    write_summary(args.out_dir / "baseline_timeline_reconstruction_summary.csv", aggregate, selected)
    print("Wrote:")
    print(f"  {args.out_dir / 'baseline_timeline_archived_mlp40hz_dronet15hz.png'}")
    print(f"  {args.out_dir / 'baseline_timeline_reconstructed_mlp500hz.png'}")
    print(f"  {args.out_dir / 'baseline_timeline_reconstruction_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
