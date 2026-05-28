#!/usr/bin/env python3
"""Export measured yolov8 profile data into scheduler inputs and plots.

This tool is intentionally data-driven: it merges real board measurements
from Merlin's per-dispatch profiler with the existing XPU-RT workload DAG.
It does not synthesize timings. Missing target cells are marked infeasible.

Outputs:
  - workload.json, processing_times.json, transfer_times.json for XPU-RT
  - schedule_fastest.json: a deterministic fastest-available preview
  - plots/*.png and summary.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
from collections import Counter
from collections.abc import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MACHINES = ["CPU", "GPU", "HTA"]
TARGET_TO_MACHINE = {
    "cpu": "CPU",
    "qnn_gpu": "GPU",
    "qnn_hta": "HTA",
}
INF_US = 1.0e9


def _load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def _load_xpu_manifest(path: pathlib.Path) -> dict:
    if path.is_file():
        return _load_json(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    dispatches = {}
    for shape_path in sorted(path.glob("dispatch_*.shapes.json")):
        row = _load_json(shape_path)
        name = row.get("name") or shape_path.name.removesuffix(".shapes.json")
        dispatches[name] = row
    if not dispatches:
        raise ValueError(f"no dispatch_*.shapes.json files found in {path}")
    return {"dispatches": dispatches, "source": str(path)}


def _dispatch_name_from_any(name: str) -> str | None:
    """Normalize Merlin/XPU dispatch keys to XPU-RT's dispatch_N form."""
    if name.startswith("dispatch_"):
        parts = name.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            return f"dispatch_{int(parts[1])}"
    # HTA island names are hta_conv_island_dispatch_N_dispatch_0. The first
    # dispatch number is the original yolov8 dispatch; the suffix is the
    # per-island dump's local dispatch id.
    if name.startswith("hta_conv_island_dispatch_"):
        rest = name.removeprefix("hta_conv_island_dispatch_")
        first = rest.split("_", 1)[0]
        if first.isdigit():
            return f"dispatch_{int(first)}"
    matches = re.findall(r"dispatch_(\d+)", name)
    if not matches:
        return None
    return f"dispatch_{int(matches[-1])}"


def _profile_mean_us(profile: dict | None) -> float | None:
    if not profile:
        return None
    for key in ("mean_us", "mean_time_us"):
        value = profile.get(key)
        if isinstance(value, int | float) and value > 0:
            return float(value)
    return None


def _profile_setup_us(profile: dict | None) -> float | None:
    if not profile:
        return None
    value = profile.get("setup_us")
    if isinstance(value, int | float) and value >= 0:
        return float(value)
    return None


def _read_cpu_profile(path: pathlib.Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    data = _load_json(path)
    out: dict[str, dict] = {}
    for key, row in data.get("dispatches", {}).items():
        dispatch = _dispatch_name_from_any(key)
        if dispatch is None:
            continue
        cpu_cell = row.get("cpu") if isinstance(row.get("cpu"), dict) else None
        cpu_profile = cpu_cell.get("profile") if cpu_cell else None
        mean = _profile_mean_us(cpu_profile) or _profile_mean_us(row)
        setup = _profile_setup_us(cpu_profile) or _profile_setup_us(row)
        if mean is None:
            profiles = row.get("profiles", {})
            if profiles:
                mean = _profile_mean_us(next(iter(profiles.values())))
        if mean is not None:
            out[dispatch] = {
                "mean_us": mean,
                "setup_us": setup or 0.0,
                "kind": "whole_dispatch",
                "source_key": key,
            }
    return out


def _read_merlin_profile(path: pathlib.Path | None, target: str, kind: str) -> dict[str, dict]:
    if path is None:
        return {}
    data = _load_json(path)
    out: dict[str, dict] = {}
    for key, row in data.get("dispatches", {}).items():
        dispatch = _dispatch_name_from_any(key)
        if dispatch is None:
            continue
        cell = row.get(target, {})
        profile = cell.get("profile") if isinstance(cell, dict) else None
        mean = _profile_mean_us(profile)
        if mean is None:
            continue
        out[dispatch] = {
            "mean_us": mean,
            "setup_us": _profile_setup_us(profile) or 0.0,
            "kind": kind,
            "source_key": key,
            "func": cell.get("func"),
            "binding_byte_sizes": list(cell.get("binding_byte_sizes", [])),
        }
    return out


_TENSOR_RE = re.compile(r"tensor<([^>]+)>")


def _tensor_nbytes(type_text: str) -> int | None:
    match = _TENSOR_RE.search(type_text)
    if not match:
        return None
    parts = match.group(1).split("x")
    if len(parts) < 2:
        return None
    elem = parts[-1]
    dims: list[int] = []
    for part in parts[:-1]:
        if not part.isdigit():
            return None
        dims.append(int(part))
    elem_bytes = 1
    if elem in {"f16", "i16", "ui16"}:
        elem_bytes = 2
    elif elem in {"f32", "i32", "ui32"}:
        elem_bytes = 4
    elif elem in {"f64", "i64", "ui64"}:
        elem_bytes = 8
    elif elem in {"i8", "ui8"}:
        elem_bytes = 1
    else:
        return None
    n = elem_bytes
    for dim in dims:
        n *= dim
    return n


def _dispatch_output_bytes(manifest: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, row in manifest.get("dispatches", {}).items():
        dispatch = _dispatch_name_from_any(key)
        if dispatch is None:
            continue
        sizes = [_tensor_nbytes(t) for t in row.get("outputs", [])]
        sizes = [s for s in sizes if s is not None]
        if sizes:
            out[dispatch] = max(sizes)
    return out


def _sample_transfer(samples: dict, src: str, dst: str, nbytes: int) -> float:
    if src == dst:
        return 0.0
    key = f"{src.lower()}->{dst.lower()}".replace("gpu", "qnn_gpu").replace("hta", "qnn_hta")
    points = samples.get(key, [])
    if not points:
        return 0.0
    points = sorted((int(b), float(us)) for b, us in points)
    if nbytes <= points[0][0]:
        return points[0][1]
    for (b0, t0), (b1, t1) in zip(points, points[1:], strict=False):
        if b0 <= nbytes <= b1:
            if b1 == b0:
                return t1
            frac = (nbytes - b0) / (b1 - b0)
            return t0 + frac * (t1 - t0)
    b0, t0 = points[-2]
    b1, t1 = points[-1]
    slope = (t1 - t0) / (b1 - b0) if b1 != b0 else 0.0
    return max(0.0, t1 + slope * (nbytes - b1))


def _build_transfer_matrix(samples: dict, nbytes: int) -> list[list[float]]:
    matrix: list[list[float]] = []
    for src in MACHINES:
        row: list[float] = []
        for dst in MACHINES:
            row.append(_sample_transfer(samples, src, dst, nbytes))
        matrix.append(row)
    return matrix


def _topological_order(workload: dict) -> list[str]:
    pending = set(workload["dispatches"])
    order: list[str] = []
    while pending:
        progressed = False
        for name in sorted(pending, key=lambda n: workload["dispatches"][n]["id"]):
            deps = workload["dispatches"][name].get("dependencies", [])
            if any(dep in pending for dep in deps if dep in workload["dispatches"]):
                continue
            order.append(name)
            pending.remove(name)
            progressed = True
            break
        if not progressed:
            missing = sorted(pending)[:10]
            raise RuntimeError(f"dependency cycle or missing predecessor around {missing}")
    return order


def _fastest_available_assignments(processing: dict[str, list[float]]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for name, times in processing.items():
        machine_idx = min(range(len(times)), key=lambda i: times[i])
        if times[machine_idx] >= INF_US:
            raise RuntimeError(f"{name} has no feasible measured target")
        assignments[name] = MACHINES[machine_idx]
    return assignments


def _minimum_overhead_diverse_assignments(
    processing: dict[str, list[float]],
    base: dict[str, str],
) -> dict[str, str]:
    assignments = dict(base)
    used = set(assignments.values())
    for machine_idx, machine in enumerate(MACHINES):
        if machine in used:
            continue
        best: tuple[float, str] | None = None
        for name, times in processing.items():
            if times[machine_idx] >= INF_US:
                continue
            cur_idx = MACHINES.index(assignments[name])
            penalty = times[machine_idx] - times[cur_idx]
            if best is None or penalty < best[0]:
                best = (penalty, name)
        if best is not None:
            assignments[best[1]] = machine
            used.add(machine)
    return assignments


def _schedule_from_assignments(
    workload: dict,
    processing: dict[str, list[float]],
    assignments: dict[str, str],
    transfer_matrix: list[list[float]],
    policy: str,
) -> dict:
    starts: dict[str, float] = {}
    ends: dict[str, float] = {}
    machine_free = {m: 0.0 for m in MACHINES}
    ops: list[dict] = []
    for name in _topological_order(workload):
        times = processing[name]
        machine = assignments[name]
        machine_idx = MACHINES.index(machine)
        duration = times[machine_idx]
        if duration >= INF_US:
            raise RuntimeError(f"{name} assignment to {machine} is infeasible")
        deps = workload["dispatches"][name].get("dependencies", [])
        ready = 0.0
        for dep in deps:
            if dep not in ends:
                continue
            pred_idx = MACHINES.index(assignments[dep])
            ready = max(ready, ends[dep] + transfer_matrix[pred_idx][machine_idx])
        start = max(ready, machine_free[machine])
        end = start + duration
        starts[name] = start
        ends[name] = end
        machine_free[machine] = end
        ops.append(
            {
                "name": name,
                "machine": machine,
                "start_us": start,
                "duration_us": duration,
                "end_us": end,
                "dependencies": deps,
            }
        )
    return {
        "machines": MACHINES,
        "policy": policy,
        "ops": ops,
        "counts_per_machine": dict(Counter(assignments.values())),
        "makespan_us": max(ends.values(), default=0.0),
    }


def _fastest_available_schedule(
    workload: dict,
    processing: dict[str, list[float]],
    transfer_matrix: list[list[float]],
) -> dict:
    return _schedule_from_assignments(
        workload,
        processing,
        _fastest_available_assignments(processing),
        transfer_matrix,
        "fastest_available_topological_preview",
    )


def _showcase_schedule(
    workload: dict,
    processing: dict[str, list[float]],
    transfer_matrix: list[list[float]],
) -> dict:
    base = _fastest_available_assignments(processing)
    assignments = _minimum_overhead_diverse_assignments(processing, base)
    return _schedule_from_assignments(
        workload,
        processing,
        assignments,
        transfer_matrix,
        "minimum_overhead_all_measured_targets_preview",
    )


def _save_bar(
    path: pathlib.Path,
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = ["#3c6e71", "#d9a441", "#7b2d26", "#6b705c", "#4a4e69"]
    ax.bar(labels, values, color=colors[: len(labels)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _dispatch_label(name: str) -> str:
    match = re.search(r"dispatch_(\d+)", name)
    return match.group(1) if match else name


def _write_timeline(path: pathlib.Path, schedule: dict, title: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 6.2))
    y = {m: i for i, m in enumerate(MACHINES)}
    colors = {"CPU": "#3c6e71", "GPU": "#d9a441", "HTA": "#7b2d26"}
    max_end_ms = max((op["end_us"] / 1000.0 for op in schedule["ops"]), default=0.0)
    for op in schedule["ops"]:
        left_ms = op["start_us"] / 1000.0
        width_ms = op["duration_us"] / 1000.0
        ax.barh(
            y[op["machine"]],
            width_ms,
            left=left_ms,
            height=0.58,
            color=colors[op["machine"]],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.45,
        )
        if width_ms >= max(0.55, max_end_ms * 0.0075):
            ax.text(
                left_ms + width_ms / 2.0,
                y[op["machine"]],
                _dispatch_label(op["name"]),
                ha="center",
                va="center",
                fontsize=6.5,
                color="black",
                clip_on=True,
            )
    ax.set_yticks(range(len(MACHINES)), MACHINES)
    ax.set_xlabel("time (ms)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[m], edgecolor="black") for m in MACHINES]
    ax.legend(
        handles,
        MACHINES,
        title="Color = target",
        loc="upper right",
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_plots(
    out_dir: pathlib.Path,
    summary: dict,
    processing: dict[str, list[float]],
    schedule: dict,
    showcase: dict,
    transfer_matrix: list[list[float]],
) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    _save_bar(
        plot_dir / "coverage_status.png",
        ["CPU whole", "GPU whole", "HTA islands"],
        [
            summary["measured_counts"]["CPU"],
            summary["measured_counts"]["GPU"],
            summary["measured_counts"]["HTA"],
        ],
        "Measured yolov8 dispatch coverage on QRB5165",
        "dispatches",
    )
    _save_bar(
        plot_dir / "fastest_preview_placements.png",
        MACHINES,
        [schedule["counts_per_machine"].get(m, 0) for m in MACHINES],
        "Fastest measured target preview",
        "dispatches placed",
    )
    _save_bar(
        plot_dir / "showcase_preview_placements.png",
        MACHINES,
        [showcase["counts_per_machine"].get(m, 0) for m in MACHINES],
        "Minimum-overhead all-target preview",
        "dispatches placed",
    )

    paired = []
    for name, times in processing.items():
        if times[0] < INF_US and times[1] < INF_US:
            paired.append((name, times[0], times[1]))
    if paired:
        fig, ax = plt.subplots(figsize=(6, 5))
        cpu = [p[1] / 1000.0 for p in paired]
        gpu = [p[2] / 1000.0 for p in paired]
        ax.scatter(cpu, gpu, color="#d9a441", edgecolor="#222", alpha=0.8)
        lo = max(1.0, min(min(cpu), min(gpu)))
        hi = max(max(cpu), max(gpu))
        ax.plot([lo, hi], [lo, hi], color="#444", linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("CPU mean (ms)")
        ax.set_ylabel("QNN GPU mean (ms)")
        ax.set_title("Whole-dispatch CPU vs QNN GPU measured cells")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "cpu_vs_gpu_scatter.png", dpi=160)
        plt.close(fig)

    hta_rows = [(name, times[2]) for name, times in processing.items() if times[2] < INF_US]
    if hta_rows:
        hta_rows = sorted(hta_rows, key=lambda x: x[1], reverse=True)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.bar(
            [x[0] for x in hta_rows],
            [x[1] / 1000.0 for x in hta_rows],
            color="#7b2d26",
            edgecolor="black",
            linewidth=0.7,
        )
        ax.set_title("Measured QNN HTA conv islands")
        ax.set_ylabel("mean (ms)")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "hta_conv_islands.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(np.array(transfer_matrix), cmap="YlGnBu")
    ax.set_xticks(range(len(MACHINES)), MACHINES)
    ax.set_yticks(range(len(MACHINES)), MACHINES)
    for i in range(len(MACHINES)):
        for j in range(len(MACHINES)):
            ax.text(
                j,
                i,
                f"{transfer_matrix[i][j] / 1000.0:.3f}",
                ha="center",
                va="center",
            )
    ax.set_title("Measured transfer matrix (ms)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(plot_dir / "transfer_heatmap.png", dpi=160)
    plt.close(fig)
    _write_timeline(
        plot_dir / "fastest_preview_timeline.png",
        schedule,
        "Fastest measured preview (color = target, labels = dispatch id)",
    )
    _write_timeline(
        plot_dir / "showcase_preview_timeline.png",
        showcase,
        "All-target measured preview (color = target, labels = dispatch id)",
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=pathlib.Path, required=True)
    parser.add_argument(
        "--xpu-manifest",
        type=pathlib.Path,
        required=True,
        help="JSON manifest, or an XPU-RT breakdown dir with dispatch_*.shapes.json",
    )
    parser.add_argument("--cpu-profile", type=pathlib.Path, required=True)
    parser.add_argument("--gpu-profile", type=pathlib.Path, required=True)
    parser.add_argument("--hta-island-profile", type=pathlib.Path, required=True)
    parser.add_argument("--transfers", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--transfer-size-bytes", type=int, default=None)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    workload_in = _load_json(args.workload)
    xpu_manifest = _load_xpu_manifest(args.xpu_manifest)
    cpu = _read_cpu_profile(args.cpu_profile)
    gpu = _read_merlin_profile(args.gpu_profile, "qnn_gpu", "whole_dispatch")
    hta = _read_merlin_profile(args.hta_island_profile, "qnn_hta", "conv_island")
    transfer_samples = _load_json(args.transfers).get("samples_per_pair", {})
    output_bytes = _dispatch_output_bytes(xpu_manifest)
    max_output_bytes = max(output_bytes.values(), default=0)
    transfer_size = args.transfer_size_bytes or max_output_bytes
    transfer_matrix = _build_transfer_matrix(transfer_samples, transfer_size)

    workload_out = {"dispatches": {}, "machines": MACHINES}
    processing: dict[str, list[float]] = {}
    support: dict[str, dict] = {}

    for name, row in sorted(
        workload_in.get("dispatches", {}).items(),
        key=lambda kv: int(kv[1].get("id", 0)),
    ):
        if name not in cpu:
            # No measured CPU cell means this dispatch cannot currently be
            # scheduled from real data. Keep it out rather than inventing data.
            continue
        times = [
            cpu[name]["mean_us"],
            gpu.get(name, {}).get("mean_us", INF_US),
            hta.get(name, {}).get("mean_us", INF_US),
        ]
        infeasible = [MACHINES[i] for i, value in enumerate(times) if value >= INF_US]
        edge_bytes = output_bytes.get(name, transfer_size)
        cost_by_pred = {}
        for src in MACHINES:
            for dst in MACHINES:
                cost_by_pred[f"{src}->{dst}"] = _sample_transfer(transfer_samples, src, dst, edge_bytes)
        workload_out["dispatches"][name] = {
            "id": row.get("id", len(workload_out["dispatches"])),
            "dependencies": list(row.get("dependencies", [])),
            "op_summary": row.get("op_summary", ""),
            "infeasible_machines": infeasible,
            "cost_by_pred": cost_by_pred,
            "output_bytes": edge_bytes,
            "target_support": {
                "CPU": cpu.get(name, {}).get("kind"),
                "GPU": gpu.get(name, {}).get("kind"),
                "HTA": hta.get(name, {}).get("kind"),
            },
        }
        processing[name] = times
        support[name] = {
            "CPU": cpu.get(name),
            "GPU": gpu.get(name),
            "HTA": hta.get(name),
        }

    setup = {}
    for label, rows in [("CPU", cpu), ("GPU", gpu), ("HTA", hta)]:
        values = [r.get("setup_us", 0.0) for r in rows.values() if r.get("setup_us")]
        setup[label] = statistics.median(values) if values else 0.0
    workload_out["setup_us_by_machine"] = setup
    workload_out["transfer_volume_bytes"] = transfer_size
    workload_out["transfer_source"] = str(args.transfers)

    schedule = _fastest_available_schedule(workload_out, processing, transfer_matrix)
    showcase = _showcase_schedule(workload_out, processing, transfer_matrix)
    measured_counts = {
        "CPU": sum(1 for x in processing.values() if x[0] < INF_US),
        "GPU": sum(1 for x in processing.values() if x[1] < INF_US),
        "HTA": sum(1 for x in processing.values() if x[2] < INF_US),
    }
    summary = {
        "workload_dispatches": len(workload_in.get("dispatches", {})),
        "scheduler_dispatches_with_real_cpu": len(processing),
        "measured_counts": measured_counts,
        "paired_cpu_gpu_dispatches": sum(1 for x in processing.values() if x[0] < INF_US and x[1] < INF_US),
        "hta_island_dispatches": sorted(name for name, x in processing.items() if x[2] < INF_US),
        "setup_us_by_machine_median": setup,
        "transfer_size_bytes": transfer_size,
        "fastest_preview_counts": schedule["counts_per_machine"],
        "fastest_preview_makespan_us": schedule["makespan_us"],
        "showcase_preview_counts": showcase["counts_per_machine"],
        "showcase_preview_makespan_us": showcase["makespan_us"],
        "notes": [
            "GPU rows are whole-dispatch QNN GPU VMFB measurements.",
            "HTA rows are measured channel-last int8 conv islands; they require a boundary split pass before full-network execution.",
            "Missing cells are infeasible, not filled with synthetic timings.",
        ],
    }

    (args.out_dir / "workload.json").write_text(json.dumps(workload_out, indent=2))
    (args.out_dir / "processing_times.json").write_text(json.dumps(processing, indent=2))
    (args.out_dir / "transfer_times.json").write_text(
        json.dumps(
            {
                "machines": MACHINES,
                "matrix": transfer_matrix,
                "edge_volume_bytes": transfer_size,
                "source": str(args.transfers),
            },
            indent=2,
        )
    )
    (args.out_dir / "target_support.json").write_text(json.dumps(support, indent=2))
    (args.out_dir / "schedule_fastest.json").write_text(json.dumps(schedule, indent=2))
    (args.out_dir / "schedule_showcase_all_targets.json").write_text(json.dumps(showcase, indent=2))
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_plots(args.out_dir, summary, processing, schedule, showcase, transfer_matrix)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
