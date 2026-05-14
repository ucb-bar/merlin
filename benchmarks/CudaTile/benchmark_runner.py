#!/usr/bin/env python3
# Copyright 2026 UCB-BAR
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Unified GPU benchmark runner.
#
# Compares compilation paths: IREE CUDA, IREE cuda_new, cuda_tile (old HAL),
# cuda_tile (new HAL), and torch.compile (inductor/Triton).
#
# Usage:
#   python benchmark_runner.py --level 1 --backends iree-cuda,torch-compile \
#       --build-dir /path/to/build
#
#   # Legacy compat: --phase N maps to --level N --backends iree-cuda,ctl-old-hal
#   python benchmark_runner.py --phase 1 --build-dir /path/to/build

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def _early_gpu_setup():
    """Set CUDA_VISIBLE_DEVICES before any torch imports to avoid dynamo crashes."""
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
            break


_early_gpu_setup()

from config import BenchConfig
from backends.base import BackendID, CompileResult, RunResult
from backends.iree_backend import IREEBackend
from backends.torch_compile_backend import TorchCompileBackend
from reporting.json_report import save_json_report
from reporting.markdown_report import generate_markdown_report
from reporting.history import save_to_history, compare_with_history
from workloads import ALL_WORKLOADS, WORKLOADS_BY_LEVEL

_BACKEND_MAP = {
    "iree-cuda": lambda cfg: IREEBackend(BackendID.IREE_CUDA, cfg),
    "iree-cuda-new": lambda cfg: IREEBackend(BackendID.IREE_CUDA_NEW, cfg),
    "ctl-old-hal": lambda cfg: IREEBackend(BackendID.CTL_OLD_HAL, cfg),
    "ctl-new-hal": lambda cfg: IREEBackend(BackendID.CTL_NEW_HAL, cfg),
    "torch-compile": lambda cfg: TorchCompileBackend(cfg),
}


def run_benchmarks(config: BenchConfig, backend_names: list[str], levels: list[int], name_filter: str | None = None):
    backends = []
    for name in backend_names:
        factory = _BACKEND_MAP.get(name)
        if not factory:
            print(f"Unknown backend: {name}. Available: {list(_BACKEND_MAP.keys())}")
            sys.exit(1)
        backend = factory(config)
        if not backend.check_available():
            print(f"Backend {name} is not available, skipping.")
            continue
        backends.append(backend)

    if not backends:
        print("No available backends.")
        sys.exit(1)

    workloads = []
    for level in sorted(levels):
        workloads.extend(WORKLOADS_BY_LEVEL.get(level, []))

    if name_filter:
        import fnmatch
        workloads = [w for w in workloads if fnmatch.fnmatch(w.name, name_filter)]

    if not workloads:
        print(f"No workloads for levels {levels}")
        sys.exit(1)

    all_results = []
    total = len(workloads) * len(backends)
    done = 0

    for workload in workloads:
        for size in workload.sizes:
            desc = size.get("desc", "default")

            for backend in backends:
                done += 1
                tag = f"[{done}/{total}] {workload.name}/{desc} on {backend.name}"
                print(f"\n{tag}")

                result_entry = {
                    "workload": workload.name,
                    "level": workload.level,
                    "size": desc,
                    "size_params": {k: v for k, v in size.items() if k != "desc"},
                    "backend": backend.name,
                }

                # Compile.
                compile_result = backend.compile(workload, size)
                result_entry["compile_time_s"] = compile_result.compile_time_s
                result_entry["artifact_bytes"] = compile_result.artifact_bytes

                if not compile_result.success:
                    result_entry["error"] = compile_result.error
                    print(f"  COMPILE FAIL: {compile_result.error[:100]}")
                    all_results.append(result_entry)
                    continue

                print(f"  compiled in {compile_result.compile_time_s:.2f}s ({compile_result.artifact_bytes} bytes)")

                # Run.
                run_result = backend.run(compile_result, workload, size)
                result_entry["first_run_ms"] = run_result.first_run_ms
                result_entry["subsequent_p50_ms"] = run_result.subsequent_p50_ms
                result_entry["subsequent_p99_ms"] = run_result.subsequent_p99_ms
                result_entry["subsequent_min_ms"] = run_result.subsequent_min_ms
                result_entry["subsequent_max_ms"] = run_result.subsequent_max_ms
                result_entry["throughput_ops_s"] = run_result.throughput_ops_s

                if run_result.error:
                    result_entry["error"] = run_result.error

                if run_result.subsequent_p50_ms > 0:
                    print(
                        f"  first: {run_result.first_run_ms:.1f}ms, "
                        f"p50: {run_result.subsequent_p50_ms:.3f}ms, "
                        f"p99: {run_result.subsequent_p99_ms:.3f}ms, "
                        f"throughput: {run_result.throughput_ops_s:.0f} ops/s"
                    )
                elif run_result.error:
                    print(f"  RUN ISSUE: {run_result.error[:100]}")

                all_results.append(result_entry)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="GPU benchmark runner")
    parser.add_argument(
        "--level",
        type=int,
        nargs="+",
        default=None,
        help="Workload levels to run (1=primitives, 2=composites, 3=small NNs, 4=CNNs, 5=transformers)",
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="iree-cuda,ctl-old-hal",
        help="Comma-separated backends (default: iree-cuda,ctl-old-hal)",
    )
    parser.add_argument(
        "--build-dir",
        type=str,
        required=True,
        help="Path to the IREE build directory",
    )
    parser.add_argument("--sm-arch", type=str, default="sm_86")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/<timestamp>)",
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=None, help="GPU index (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument(
        "--ctl-tiles", type=str, default=None,
        help="Tile flags for cuda_tile backends, e.g. '64,64,16' for tileM,tileN,tileK",
    )
    parser.add_argument("--filter", type=str, default=None, help="Glob filter on workload names")
    parser.add_argument(
        "--phase",
        type=int,
        default=None,
        help="Legacy compat: --phase N = --level N --backends iree-cuda,ctl-old-hal",
    )
    parser.add_argument(
        "--all-phases",
        action="store_true",
        help="Legacy compat: run phases 1..N",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Save results to history and check for regressions",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots (requires matplotlib)",
    )
    args = parser.parse_args()

    # Legacy compat.
    if args.phase is not None:
        levels = list(range(1, args.phase + 1)) if args.all_phases else [args.phase]
        backend_names = ["iree-cuda", "ctl-old-hal"]
    elif args.level is not None:
        levels = args.level
        backend_names = [b.strip() for b in args.backends.split(",")]
    else:
        parser.error("Either --level or --phase is required")
        return

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"results/run_{ts}")

    ctl_extra = []
    if args.ctl_tiles:
        parts = args.ctl_tiles.split(",")
        if len(parts) == 3:
            ctl_extra = [
                f"--iree-cuda-tile-tile-m={parts[0]}",
                f"--iree-cuda-tile-tile-n={parts[1]}",
                f"--iree-cuda-tile-tile-k={parts[2]}",
            ]

    config = BenchConfig(
        build_dir=Path(args.build_dir),
        sm_arch=args.sm_arch,
        output_dir=output_dir,
        warmup=args.warmup,
        iterations=args.iterations,
        gpu_index=args.gpu,
        ctl_extra_flags=ctl_extra,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")
    print(f"Backends: {backend_names}")
    print(f"Levels: {levels}")
    print(f"Iterations: {config.iterations}, Warmup: {config.warmup}")

    results = run_benchmarks(config, backend_names, levels, args.filter)

    if not results:
        print("No results collected.")
        return

    # Save reports.
    json_path = output_dir / "benchmark_results.json"
    save_json_report(results, json_path, config.sm_arch, config.build_dir)
    print(f"\nJSON report: {json_path}")

    md_path = output_dir / "benchmark_report.md"
    generate_markdown_report(results, md_path)
    print(f"Markdown report: {md_path}")

    if args.plot:
        from reporting.plots import generate_plots

        plot_paths = generate_plots(json_path, output_dir / "plots")
        if plot_paths:
            print(f"Generated {len(plot_paths)} plots in {output_dir / 'plots'}")

    if args.history:
        from reporting.json_report import _get_metadata

        metadata = _get_metadata(config.sm_arch, config.build_dir)
        hist_path = save_to_history(results, metadata)
        print(f"Saved to history: {hist_path}")

        regressions = compare_with_history(results)
        if regressions:
            print(f"\nWARNING: {len(regressions)} regression(s) detected:")
            for reg in regressions:
                print(
                    f"  {reg['workload']}/{reg['size']} on {reg['backend']}: "
                    f"{reg['previous_p50_ms']:.3f}ms -> {reg['current_p50_ms']:.3f}ms "
                    f"(+{reg['change_pct']:.1f}%)"
                )
        else:
            print("No regressions detected vs previous run.")

    # Print summary table.
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    by_backend: dict[str, list] = {}
    for r in results:
        by_backend.setdefault(r["backend"], []).append(r)

    for backend, bres in by_backend.items():
        successes = sum(1 for r in bres if r.get("subsequent_p50_ms", 0) > 0)
        failures = len(bres) - successes
        avg_p50 = 0
        if successes > 0:
            avg_p50 = sum(r["subsequent_p50_ms"] for r in bres if r.get("subsequent_p50_ms", 0) > 0) / successes
        print(f"  {backend}: {successes} ok, {failures} fail, avg p50={avg_p50:.3f}ms")


if __name__ == "__main__":
    main()
