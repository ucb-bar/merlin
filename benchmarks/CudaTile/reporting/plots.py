"""Plot generation from benchmark results."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def generate_plots(json_path: Path, output_dir: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return []

    with open(json_path) as f:
        report = json.load(f)
    results = report["results"]

    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    generated.extend(_plot_latency_bars(results, output_dir, plt, np))
    generated.extend(_plot_scaling_lines(results, output_dir, plt, np))
    generated.extend(_plot_compile_time(results, output_dir, plt, np))
    generated.extend(_plot_first_run(results, output_dir, plt, np))

    return generated


def _plot_latency_bars(results, output_dir, plt, np) -> list[Path]:
    by_workload = defaultdict(list)
    for r in results:
        by_workload[r["workload"]].append(r)

    paths = []
    for workload_name, entries in by_workload.items():
        sizes = []
        backend_data = defaultdict(list)
        seen_sizes = set()

        for e in entries:
            size = e["size"]
            backend = e["backend"]
            p50 = e.get("subsequent_p50_ms", 0)
            if size not in seen_sizes:
                sizes.append(size)
                seen_sizes.add(size)
            backend_data[backend].append((size, p50))

        if not sizes:
            continue

        backends = sorted(backend_data.keys())
        x = np.arange(len(sizes))
        width = 0.8 / max(len(backends), 1)

        fig, ax = plt.subplots(figsize=(max(8, len(sizes) * 1.5), 5))
        for i, backend in enumerate(backends):
            vals = {s: v for s, v in backend_data[backend]}
            heights = [vals.get(s, 0) for s in sizes]
            ax.bar(x + i * width, heights, width, label=backend, alpha=0.85)

        ax.set_xlabel("Size")
        ax.set_ylabel("p50 Latency (ms)")
        ax.set_title(f"{workload_name} — Latency by Backend")
        ax.set_xticks(x + width * (len(backends) - 1) / 2)
        ax.set_xticklabels(sizes, rotation=45, ha="right")
        ax.legend()
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        path = output_dir / f"latency_{workload_name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    return paths


def _plot_scaling_lines(results, output_dir, plt, np) -> list[Path]:
    by_workload = defaultdict(lambda: defaultdict(list))
    for r in results:
        p50 = r.get("subsequent_p50_ms", 0)
        if p50 <= 0:
            continue
        params = r.get("size_params", {})
        total_elements = 1
        for v in params.values():
            if isinstance(v, (int, float)):
                total_elements *= v
        by_workload[r["workload"]][r["backend"]].append(
            (total_elements, p50)
        )

    paths = []
    for workload_name, backend_data in by_workload.items():
        if not backend_data:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        for backend, points in sorted(backend_data.items()):
            points.sort()
            xs, ys = zip(*points)
            ax.plot(xs, ys, "o-", label=backend, markersize=5)

        ax.set_xlabel("Problem Size (total elements)")
        ax.set_ylabel("p50 Latency (ms)")
        ax.set_title(f"{workload_name} — Scaling")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()

        path = output_dir / f"scaling_{workload_name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    return paths


def _plot_compile_time(results, output_dir, plt, np) -> list[Path]:
    by_workload = defaultdict(list)
    for r in results:
        ct = r.get("compile_time_s", 0)
        if ct > 0:
            by_workload[r["workload"]].append(r)

    if not by_workload:
        return []

    all_workloads = list(by_workload.keys())
    backends = sorted({r["backend"] for r in results})
    x = np.arange(len(all_workloads))
    width = 0.8 / max(len(backends), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(all_workloads) * 2), 5))
    for i, backend in enumerate(backends):
        heights = []
        for wl in all_workloads:
            entries = [r for r in by_workload[wl] if r["backend"] == backend]
            avg_ct = sum(r["compile_time_s"] for r in entries) / max(len(entries), 1)
            heights.append(avg_ct)
        ax.bar(x + i * width, heights, width, label=backend, alpha=0.85)

    ax.set_xlabel("Workload")
    ax.set_ylabel("Avg Compile Time (s)")
    ax.set_title("Compilation Time by Backend")
    ax.set_xticks(x + width * (len(backends) - 1) / 2)
    ax.set_xticklabels(all_workloads, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = output_dir / "compile_times.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]


def _plot_first_run(results, output_dir, plt, np) -> list[Path]:
    by_workload = defaultdict(list)
    for r in results:
        fr = r.get("first_run_ms", 0)
        if fr > 0:
            by_workload[r["workload"]].append(r)

    if not by_workload:
        return []

    all_workloads = list(by_workload.keys())
    backends = sorted({r["backend"] for r in results})
    x = np.arange(len(all_workloads))
    width = 0.8 / max(len(backends), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(all_workloads) * 2), 5))
    for i, backend in enumerate(backends):
        heights = []
        for wl in all_workloads:
            entries = [r for r in by_workload[wl] if r["backend"] == backend]
            avg_fr = sum(r["first_run_ms"] for r in entries) / max(len(entries), 1)
            heights.append(avg_fr)
        ax.bar(x + i * width, heights, width, label=backend, alpha=0.85)

    ax.set_xlabel("Workload")
    ax.set_ylabel("Avg First-Run Latency (ms)")
    ax.set_title("First-Run Latency by Backend (includes module load + JIT)")
    ax.set_xticks(x + width * (len(backends) - 1) / 2)
    ax.set_xticklabels(all_workloads, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = output_dir / "first_run_latency.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]
