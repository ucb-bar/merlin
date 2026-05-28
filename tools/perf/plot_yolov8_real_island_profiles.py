#!/usr/bin/env python3
"""Plot QNN island measurements (CPU / QNN_GPU / QNN_HTA) for any model.

Model-agnostic — was first used for YOLOv8 dequant + conv islands, but
the inputs are not yolov8-specific. Consumes only measured artifacts:

* CPU strict flow-runner trace (`invoke_us` per dispatch).
* QNN GPU profiled manifest for captured dequant islands.
* QNN HTA real-island CSV for captured conv islands.

Targets are fixed as {CPU, QNN_GPU, QNN_HTA} (this is a QNN-specific tool).
Pass `--target-colors <yaml>` to override the default palette.

It does not synthesize timings. Missing target cells remain blank in the CSV.
The produced schedule is a measured-data visualization schedule, not a claim
that the current runtime can execute any island split end-to-end.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

TARGET_COLORS = {
    "CPU": "#8c8c8c",
    "QNN_GPU": "#2f80ed",
    "QNN_HTA": "#f2994a",
}


def _dispatch_id(name: str) -> int:
    match = re.search(r"dispatch_(\d+)", name)
    if not match:
        raise ValueError(f"not a dispatch name: {name}")
    return int(match.group(1))


def _sort_key(name: str) -> tuple[int, int, str]:
    call = re.search(r"_call_(\d+)", name)
    call_id = int(call.group(1)) if call else _dispatch_id(name)
    return (call_id, _dispatch_id(name), name)


def _dispatch_name(name: str) -> str:
    return f"dispatch_{_dispatch_id(name)}"


def _load_cpu_trace(path: pathlib.Path) -> dict[str, float]:
    df = pd.read_csv(path)
    out: dict[str, float] = {}
    for row in df.itertuples(index=False):
        out[str(row.op)] = float(row.invoke_us) / 1000.0
    return out


def _load_hta_csv(path: pathlib.Path) -> tuple[dict[str, float], dict[str, float]]:
    mean: dict[str, float] = {}
    setup: dict[str, float] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            dispatch = row.get("source_dispatch", "")
            if not dispatch or not row.get("mean_us"):
                continue
            mean[dispatch] = float(row["mean_us"]) / 1000.0
            setup[dispatch] = float(row.get("setup_us") or 0.0) / 1000.0
    return mean, setup


def _load_gpu_manifest(path: pathlib.Path) -> tuple[dict[str, float], dict[str, float]]:
    payload = json.loads(path.read_text())
    mean: dict[str, float] = {}
    setup: dict[str, float] = {}
    for canonical, row in payload.get("dispatches", {}).items():
        cell = row.get("qnn_gpu") or {}
        profile = cell.get("profile") or {}
        if "mean_us" not in profile:
            continue
        dispatch = _dispatch_name(canonical)
        mean[dispatch] = float(profile["mean_us"]) / 1000.0
        setup[dispatch] = float(profile.get("setup_us") or 0.0) / 1000.0
    return mean, setup


def _load_graph(
    matrix_path: pathlib.Path,
) -> tuple[dict[str, list[str]], dict[str, str], dict[str, str]]:
    payload = json.loads(matrix_path.read_text())
    deps: dict[str, list[str]] = {}
    summary: dict[str, str] = {}
    canonical: dict[str, str] = {}
    for dispatch, row in payload.get("dispatch_graph", {}).items():
        deps[dispatch] = [str(dep) for dep in row.get("dependencies", [])]
        summary[dispatch] = str(row.get("op_summary") or "")
        canonical[dispatch] = str(row.get("canonical_dispatch") or dispatch)
    return deps, summary, canonical


def _choose_target(row: dict[str, Any]) -> tuple[str, float]:
    if row.get("hta_ms") is not None:
        return "QNN_HTA", float(row["hta_ms"])
    if row.get("gpu_ms") is not None:
        return "QNN_GPU", float(row["gpu_ms"])
    return "CPU", float(row["cpu_ms"])


def _build_rows(
    *,
    cpu_ms: dict[str, float],
    gpu_ms: dict[str, float],
    gpu_setup_ms: dict[str, float],
    hta_ms: dict[str, float],
    hta_setup_ms: dict[str, float],
    summaries: dict[str, str],
    canonical: dict[str, str],
) -> list[dict[str, Any]]:
    dispatches = sorted(cpu_ms, key=_sort_key)
    rows: list[dict[str, Any]] = []
    for dispatch in dispatches:
        canonical_dispatch = canonical.get(dispatch, _dispatch_name(dispatch))
        row: dict[str, Any] = {
            "dispatch": dispatch,
            "canonical_dispatch": canonical_dispatch,
            "id": _dispatch_id(dispatch),
            "op_summary": summaries.get(dispatch, ""),
            "cpu_ms": cpu_ms[dispatch],
            "gpu_ms": gpu_ms.get(dispatch, gpu_ms.get(canonical_dispatch)),
            "hta_ms": hta_ms.get(dispatch, hta_ms.get(canonical_dispatch)),
            "gpu_setup_ms": gpu_setup_ms.get(dispatch, gpu_setup_ms.get(canonical_dispatch)),
            "hta_setup_ms": hta_setup_ms.get(dispatch, hta_setup_ms.get(canonical_dispatch)),
        }
        target, chosen_ms = _choose_target(row)
        row["chosen_target"] = target
        row["chosen_ms"] = chosen_ms
        row["note"] = (
            "real HTA conv island, not full-dispatch replacement"
            if target == "QNN_HTA"
            else "real QNN GPU dequant dispatch"
            if target == "QNN_GPU"
            else "real CPU strict-flow dispatch"
        )
        rows.append(row)
    return rows


def _schedule_rows(
    rows: list[dict[str, Any]],
    deps: dict[str, list[str]],
    transfer_ms: float,
) -> list[dict[str, Any]]:
    row_by_dispatch = {row["dispatch"]: row for row in rows}
    finish: dict[str, float] = {}
    target_free = {target: 0.0 for target in TARGET_COLORS}
    scheduled: list[dict[str, Any]] = []
    for row in rows:
        dispatch = row["dispatch"]
        target = str(row["chosen_target"])
        dep_finish = 0.0
        boundary_ms = 0.0
        for dep in deps.get(dispatch, []):
            if dep not in finish:
                continue
            dep_finish = max(dep_finish, finish[dep])
            dep_target = row_by_dispatch.get(dep, {}).get("chosen_target")
            if dep_target and dep_target != target:
                boundary_ms = max(boundary_ms, transfer_ms)
        start_ms = max(target_free[target], dep_finish + boundary_ms)
        end_ms = start_ms + float(row["chosen_ms"])
        target_free[target] = end_ms
        finish[dispatch] = end_ms
        scheduled.append(
            {
                **row,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "boundary_transfer_ms": boundary_ms,
            }
        )
    return scheduled


def _plot_timeline(rows: list[dict[str, Any]], out: pathlib.Path) -> None:
    targets = ["CPU", "QNN_GPU", "QNN_HTA"]
    y = {target: i for i, target in enumerate(targets)}
    fig, ax = plt.subplots(figsize=(16, 4.5))
    for row in rows:
        target = str(row["chosen_target"])
        start = float(row["start_ms"])
        dur = float(row["chosen_ms"])
        ax.barh(
            y[target],
            dur,
            left=start,
            height=0.72,
            color=TARGET_COLORS[target],
            edgecolor="black",
            linewidth=0.9,
        )
        ax.text(
            start + dur / 2.0,
            y[target],
            str(row["id"]),
            ha="center",
            va="center",
            fontsize=8,
            color="black",
            clip_on=True,
        )
    ax.set_yticks([y[target] for target in targets], labels=targets)
    ax.set_xlabel("Time (ms)")
    ax.set_title("YOLOv8 measured island schedule prefix (real board timings)")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_coverage(rows: list[dict[str, Any]], out: pathlib.Path) -> None:
    ids = [int(row["id"]) for row in rows]
    fig, ax = plt.subplots(figsize=(18, 5))
    width = 0.25
    specs = [
        ("cpu_ms", "CPU", -width),
        ("gpu_ms", "QNN_GPU", 0.0),
        ("hta_ms", "QNN_HTA", width),
    ]
    for key, label, dx in specs:
        xs = [i + dx for i in range(len(rows))]
        vals = [row.get(key) for row in rows]
        ax.bar(
            xs,
            [v if v is not None else 0.0 for v in vals],
            width=width,
            label=label,
            color=TARGET_COLORS[label],
            edgecolor="black",
            linewidth=0.7,
            alpha=0.92,
        )
    ax.set_xticks(range(len(rows)), labels=[str(i) for i in ids], rotation=0)
    ax.set_xlabel("Dispatch id")
    ax.set_ylabel("Measured invoke time (ms)")
    ax.set_title("YOLOv8 real measured coverage by target")
    ax.legend()
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-trace", type=pathlib.Path, required=True)
    parser.add_argument("--gpu-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--hta-csv", type=pathlib.Path, required=True)
    parser.add_argument("--matrix", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--transfer-ms",
        type=float,
        default=0.0,
        help="Optional measured boundary transfer cost to add "
        "when adjacent scheduled ops change target. "
        "Default 0.0 because no cross-target transfer "
        "measurement was produced by this script.",
    )
    args = parser.parse_args()

    cpu_ms = _load_cpu_trace(args.cpu_trace)
    gpu_ms, gpu_setup_ms = _load_gpu_manifest(args.gpu_manifest)
    hta_ms, hta_setup_ms = _load_hta_csv(args.hta_csv)
    deps, summaries, canonical = _load_graph(args.matrix)
    rows = _build_rows(
        cpu_ms=cpu_ms,
        gpu_ms=gpu_ms,
        gpu_setup_ms=gpu_setup_ms,
        hta_ms=hta_ms,
        hta_setup_ms=hta_setup_ms,
        summaries=summaries,
        canonical=canonical,
    )
    scheduled = _schedule_rows(rows, deps, args.transfer_ms)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.out_dir / "real_profile_table.csv"
    pd.DataFrame(scheduled).to_csv(table_path, index=False)
    schedule_path = args.out_dir / "measured_visual_schedule.json"
    schedule_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "note": __doc__,
                    "transfer_ms": args.transfer_ms,
                    "targets": list(TARGET_COLORS),
                },
                "dispatches": {row["dispatch"]: row for row in scheduled},
            },
            indent=2,
        )
    )
    _plot_timeline(scheduled, args.out_dir / "timeline_ms_labeled.png")
    _plot_coverage(scheduled, args.out_dir / "coverage_ms_labeled.png")

    counts: dict[str, int] = {}
    for row in scheduled:
        counts[str(row["chosen_target"])] = counts.get(str(row["chosen_target"]), 0) + 1
    print(
        json.dumps(
            {
                "rows": len(scheduled),
                "chosen_counts": counts,
                "table": str(table_path),
                "schedule": str(schedule_path),
                "timeline": str(args.out_dir / "timeline_ms_labeled.png"),
                "coverage": str(args.out_dir / "coverage_ms_labeled.png"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
