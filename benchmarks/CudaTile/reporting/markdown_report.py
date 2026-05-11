"""Markdown report generation with comparison tables."""

from __future__ import annotations

from pathlib import Path


def generate_markdown_report(
    results: list[dict],
    output_path: Path,
    baseline_backend: str = "torch-compile",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# GPU Benchmark Report\n\n")

        by_workload: dict[str, list[dict]] = {}
        for r in results:
            key = f"{r['workload']}_{r['size']}"
            by_workload.setdefault(key, []).append(r)

        all_backends = sorted(
            {r["backend"] for r in results}
        )

        f.write("## Summary\n\n")
        f.write(f"| Workload | Size |")
        for b in all_backends:
            f.write(f" {b} (ms) |")
        if baseline_backend in all_backends:
            for b in all_backends:
                if b != baseline_backend:
                    f.write(f" {b} ratio |")
        f.write("\n")

        f.write("|---|---|")
        for _ in all_backends:
            f.write("---|")
        if baseline_backend in all_backends:
            for b in all_backends:
                if b != baseline_backend:
                    f.write("---|")
        f.write("\n")

        for key, group in by_workload.items():
            backend_data = {r["backend"]: r for r in group}
            workload_name = group[0]["workload"]
            size_desc = group[0]["size"]

            f.write(f"| {workload_name} | {size_desc} |")
            for b in all_backends:
                d = backend_data.get(b)
                if d and d.get("subsequent_p50_ms", 0) > 0:
                    f.write(f" {d['subsequent_p50_ms']:.3f} |")
                elif d and d.get("error"):
                    f.write(f" FAIL |")
                else:
                    f.write(f" - |")

            if baseline_backend in all_backends:
                baseline = backend_data.get(baseline_backend, {})
                baseline_p50 = baseline.get("subsequent_p50_ms", 0)
                for b in all_backends:
                    if b == baseline_backend:
                        continue
                    d = backend_data.get(b)
                    if d and d.get("subsequent_p50_ms", 0) > 0 and baseline_p50 > 0:
                        ratio = d["subsequent_p50_ms"] / baseline_p50
                        marker = "" if ratio <= 1.2 else " !"
                        f.write(f" {ratio:.2f}x{marker} |")
                    else:
                        f.write(f" - |")
            f.write("\n")

        f.write("\n## Compilation Times\n\n")
        f.write("| Workload | Size |")
        for b in all_backends:
            f.write(f" {b} (s) |")
        f.write("\n")
        f.write("|---|---|")
        for _ in all_backends:
            f.write("---|")
        f.write("\n")

        for key, group in by_workload.items():
            backend_data = {r["backend"]: r for r in group}
            workload_name = group[0]["workload"]
            size_desc = group[0]["size"]
            f.write(f"| {workload_name} | {size_desc} |")
            for b in all_backends:
                d = backend_data.get(b)
                if d and d.get("compile_time_s", 0) > 0:
                    f.write(f" {d['compile_time_s']:.2f} |")
                else:
                    f.write(f" - |")
            f.write("\n")

        f.write("\n## First-Run Latency\n\n")
        f.write("| Workload | Size |")
        for b in all_backends:
            f.write(f" {b} (ms) |")
        f.write("\n")
        f.write("|---|---|")
        for _ in all_backends:
            f.write("---|")
        f.write("\n")

        for key, group in by_workload.items():
            backend_data = {r["backend"]: r for r in group}
            workload_name = group[0]["workload"]
            size_desc = group[0]["size"]
            f.write(f"| {workload_name} | {size_desc} |")
            for b in all_backends:
                d = backend_data.get(b)
                if d and d.get("first_run_ms", 0) > 0:
                    f.write(f" {d['first_run_ms']:.1f} |")
                else:
                    f.write(f" - |")
            f.write("\n")
