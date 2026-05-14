#!/usr/bin/env python3
# Copyright 2026 UCB-BAR
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Autotune cuda_tile matmul tile sizes (tileM, tileN, tileK).
#
# Usage:
#   uv run python autotune.py --build-dir build/host-merlin-release --gpu 4
#   uv run python autotune.py --build-dir build/host-merlin-release \
#       --tile-m 32,64,128,256 --tile-n 32,64,128,256 --tile-k 8,16,32,64

import argparse
import itertools
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def _early_gpu_setup():
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
            break


_early_gpu_setup()

from backends.base import BackendID, CompileResult, RunResult
from backends.iree_backend import IREEBackend
from config import BenchConfig
from workloads.primitives import MatmulF32


@dataclass(frozen=True)
class TileConfig:
    tile_m: int
    tile_n: int
    tile_k: int

    @property
    def flags(self) -> list[str]:
        return [
            f"--iree-cuda-tile-tile-m={self.tile_m}",
            f"--iree-cuda-tile-tile-n={self.tile_n}",
            f"--iree-cuda-tile-tile-k={self.tile_k}",
        ]

    @property
    def label(self) -> str:
        return f"{self.tile_m}x{self.tile_n}x{self.tile_k}"

    def to_dict(self) -> dict:
        return {"tile_m": self.tile_m, "tile_n": self.tile_n, "tile_k": self.tile_k}


@dataclass
class TuneResult:
    size_desc: str
    size_params: dict
    tile_config: TileConfig
    compile_time_s: float = 0.0
    compile_success: bool = False
    p50_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    error: str = ""


class TunableIREEBackend(IREEBackend):
    def __init__(self, backend_id: BackendID, config: BenchConfig,
                 extra_compile_flags: list[str] | None = None):
        super().__init__(backend_id, config)
        self._extra_compile_flags = extra_compile_flags or []

    def compile_from_mlir(self, mlir_path: Path, vmfb_path: Path,
                          input_type: str | None = None) -> CompileResult:
        return self._compile_mlir(
            mlir_path, vmfb_path, export_time=0.0,
            input_type=input_type,
            extra_flags=self._extra_compile_flags,
        )


def build_search_space(tile_m_vals, tile_n_vals, tile_k_vals) -> list[TileConfig]:
    return [
        TileConfig(m, n, k)
        for m, n, k in itertools.product(tile_m_vals, tile_n_vals, tile_k_vals)
    ]


def export_mlir_once(workload, size, output_dir: Path) -> Path:
    desc = size.get("desc", "default")
    mlir_path = output_dir / f"{workload.name}_{desc}.mlir"
    if mlir_path.exists():
        return mlir_path

    mlir_path.parent.mkdir(parents=True, exist_ok=True)
    torch_result = workload.torch_module(size)
    if torch_result is None:
        raise RuntimeError(f"workload {workload.name} has no torch_module")

    import torch
    import iree.turbine.aot as aot

    module, example_inputs = torch_result
    module = module.cpu().eval()
    cpu_inputs = [
        x.cpu() if isinstance(x, torch.Tensor) else x for x in example_inputs
    ]
    exported = aot.export(module, *cpu_inputs)
    exported.save_mlir(str(mlir_path))
    return mlir_path


def run_autotune(
    config: BenchConfig,
    backend_id: BackendID,
    workload,
    search_space: list[TileConfig],
) -> dict[str, list[TuneResult]]:
    results_by_size: dict[str, list[TuneResult]] = {}
    total = len(workload.sizes) * len(search_space)
    done = 0

    mlir_cache_dir = config.output_dir / "mlir_cache"

    for size in workload.sizes:
        desc = size.get("desc", "default")
        size_params = {k: v for k, v in size.items() if k != "desc"}
        results_by_size[desc] = []

        print(f"\n{'='*60}")
        print(f"Size: {desc} ({size_params})")
        print(f"{'='*60}")

        t0 = time.perf_counter()
        mlir_path = export_mlir_once(workload, size, mlir_cache_dir)
        export_time = time.perf_counter() - t0
        print(f"  MLIR exported in {export_time:.2f}s (cached for all tile configs)")

        for tile_cfg in search_space:
            done += 1
            tag = f"[{done}/{total}] tiles={tile_cfg.label}"

            vmfb_dir = config.output_dir / f"tile_{tile_cfg.label}" / workload.name
            vmfb_dir.mkdir(parents=True, exist_ok=True)
            vmfb_path = vmfb_dir / f"{workload.name}_{desc}.vmfb"

            backend = TunableIREEBackend(
                backend_id, config,
                extra_compile_flags=tile_cfg.flags,
            )

            compile_result = backend.compile_from_mlir(
                mlir_path, vmfb_path, input_type="torch"
            )

            tr = TuneResult(
                size_desc=desc,
                size_params=size_params,
                tile_config=tile_cfg,
                compile_time_s=compile_result.compile_time_s,
                compile_success=compile_result.success,
            )

            if not compile_result.success:
                tr.error = compile_result.error
                print(f"  {tag}  COMPILE FAIL: {compile_result.error[:80]}")
                results_by_size[desc].append(tr)
                continue

            run_result = backend.run(compile_result, workload, size)
            if run_result.success and run_result.subsequent_p50_ms > 0:
                tr.p50_ms = run_result.subsequent_p50_ms
                tr.p99_ms = run_result.subsequent_p99_ms
                tr.min_ms = run_result.subsequent_min_ms
                print(f"  {tag}  p50={tr.p50_ms:.3f}ms  p99={tr.p99_ms:.3f}ms  compile={tr.compile_time_s:.1f}s")
            else:
                tr.error = run_result.error or "no timing data"
                print(f"  {tag}  RUN FAIL: {tr.error[:80]}")

            results_by_size[desc].append(tr)

    return results_by_size


def select_best_configs(results_by_size: dict[str, list[TuneResult]]) -> dict[str, dict]:
    best = {}
    for size_desc, tune_results in results_by_size.items():
        valid = [tr for tr in tune_results if tr.p50_ms > 0]
        if not valid:
            best[size_desc] = {"error": "no valid runs"}
            continue
        winner = min(valid, key=lambda tr: tr.p50_ms)
        best[size_desc] = {
            "size_params": winner.size_params,
            "tile_config": winner.tile_config.to_dict(),
            "tile_label": winner.tile_config.label,
            "p50_ms": winner.p50_ms,
            "p99_ms": winner.p99_ms,
            "compile_time_s": winner.compile_time_s,
            "flags": " ".join(winner.tile_config.flags),
        }
    return best


def save_report(
    results_by_size: dict[str, list[TuneResult]],
    best_configs: dict[str, dict],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    best_path = output_dir / "best_configs.json"
    with open(best_path, "w") as f:
        json.dump(best_configs, f, indent=2)

    full_data = {}
    for size_desc, tune_results in results_by_size.items():
        full_data[size_desc] = [
            {
                "tile_config": tr.tile_config.to_dict(),
                "tile_label": tr.tile_config.label,
                "compile_success": tr.compile_success,
                "compile_time_s": tr.compile_time_s,
                "p50_ms": tr.p50_ms,
                "p99_ms": tr.p99_ms,
                "min_ms": tr.min_ms,
                "error": tr.error,
            }
            for tr in tune_results
        ]
    full_path = output_dir / "full_sweep.json"
    with open(full_path, "w") as f:
        json.dump(full_data, f, indent=2)

    print(f"\n{'='*70}")
    print("AUTOTUNE RESULTS — BEST CONFIG PER SIZE")
    print(f"{'='*70}")
    print(f"{'Size':<10} {'Best Tiles':<16} {'p50 (ms)':<12} {'Compile (s)':<12} Flags")
    print("-" * 90)
    for size_desc in sorted(best_configs, key=lambda x: int(x) if x.isdigit() else 0):
        info = best_configs[size_desc]
        if "error" in info:
            print(f"{size_desc:<10} {'FAILED':<16}")
        else:
            print(
                f"{size_desc:<10} {info['tile_label']:<16} "
                f"{info['p50_ms']:<12.3f} {info['compile_time_s']:<12.1f} "
                f"{info['flags']}"
            )

    print(f"\n{'='*70}")
    print("FULL RANKING BY SIZE")
    print(f"{'='*70}")
    for size_desc in sorted(results_by_size, key=lambda x: int(x) if x.isdigit() else 0):
        tune_results = results_by_size[size_desc]
        print(f"\n--- {size_desc} ---")
        valid = [tr for tr in tune_results if tr.p50_ms > 0]
        valid.sort(key=lambda tr: tr.p50_ms)
        for rank, tr in enumerate(valid, 1):
            marker = " <-- BEST" if rank == 1 else ""
            print(
                f"  #{rank} tiles={tr.tile_config.label:<16} "
                f"p50={tr.p50_ms:.3f}ms  p99={tr.p99_ms:.3f}ms  "
                f"compile={tr.compile_time_s:.1f}s{marker}"
            )
        failed = [tr for tr in tune_results if not tr.compile_success]
        if failed:
            print(f"  ({len(failed)} configs failed to compile)")

    return best_path, full_path


def main():
    parser = argparse.ArgumentParser(description="Autotune cuda_tile matmul tile sizes")
    parser.add_argument("--build-dir", type=str, required=True)
    parser.add_argument("--sm-arch", type=str, default="sm_86")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output dir (default: results/autotune_<timestamp>)",
    )
    parser.add_argument("--tile-m", type=str, default="64,128",
                        help="Comma-separated tileM values (default: 64,128)")
    parser.add_argument("--tile-n", type=str, default="64,128",
                        help="Comma-separated tileN values (default: 64,128)")
    parser.add_argument("--tile-k", type=str, default="16,32,64",
                        help="Comma-separated tileK values (default: 16,32,64)")
    parser.add_argument(
        "--backend", type=str, default="ctl-new-hal",
        choices=["ctl-old-hal", "ctl-new-hal"],
        help="cuda_tile backend variant (default: ctl-new-hal)",
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument(
        "--sizes", type=str, default=None,
        help="Comma-separated matmul size descs to test (default: all)",
    )
    parser.add_argument(
        "--apply", type=str, default=None,
        help="Path to best_configs.json to verify (no sweep)",
    )
    args = parser.parse_args()

    tile_m_vals = [int(x) for x in args.tile_m.split(",")]
    tile_n_vals = [int(x) for x in args.tile_n.split(",")]
    tile_k_vals = [int(x) for x in args.tile_k.split(",")]

    backend_id = (
        BackendID.CTL_NEW_HAL if args.backend == "ctl-new-hal"
        else BackendID.CTL_OLD_HAL
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"results/autotune_{ts}")

    config = BenchConfig(
        build_dir=Path(args.build_dir),
        sm_arch=args.sm_arch,
        output_dir=output_dir,
        warmup=args.warmup,
        iterations=args.iterations,
        gpu_index=args.gpu,
    )

    workload = MatmulF32()
    if args.sizes:
        allowed = set(args.sizes.split(","))
        workload.sizes = [s for s in workload.sizes if s.get("desc") in allowed]

    if args.apply:
        with open(args.apply) as f:
            best = json.load(f)
        print(f"Verifying {len(best)} configs from {args.apply}")
        search_space = []
        verify_sizes = []
        for size_desc, info in best.items():
            if "error" in info:
                continue
            tc = info["tile_config"]
            search_space.append(TileConfig(tc["tile_m"], tc["tile_n"], tc["tile_k"]))
            verify_sizes.append(size_desc)
        workload.sizes = [s for s in workload.sizes if s.get("desc") in verify_sizes]
        # Each size gets only its own best config — run them one at a time
        for size in workload.sizes:
            desc = size.get("desc")
            info = best.get(desc, {})
            if "error" in info:
                continue
            tc = info["tile_config"]
            single_space = [TileConfig(tc["tile_m"], tc["tile_n"], tc["tile_k"])]
            single_workload = MatmulF32()
            single_workload.sizes = [size]
            results = run_autotune(config, backend_id, single_workload, single_space)
            for desc2, trs in results.items():
                for tr in trs:
                    prev = info["p50_ms"]
                    curr = tr.p50_ms
                    if curr > 0:
                        change = (curr - prev) / prev * 100
                        status = "OK" if abs(change) < 15 else "REGRESSION" if change > 0 else "IMPROVED"
                        print(f"  {desc2}: prev={prev:.3f}ms curr={curr:.3f}ms ({change:+.1f}%) [{status}]")
        return

    search_space = build_search_space(tile_m_vals, tile_n_vals, tile_k_vals)

    print(f"Search space: {len(search_space)} tile configs")
    print(f"  tileM: {tile_m_vals}")
    print(f"  tileN: {tile_n_vals}")
    print(f"  tileK: {tile_k_vals}")
    print(f"Matmul sizes: {[s['desc'] for s in workload.sizes]}")
    print(f"Backend: {args.backend}")
    print(f"Total compiles: {len(search_space) * len(workload.sizes)}")
    print(f"Output: {output_dir}")

    results_by_size = run_autotune(config, backend_id, workload, search_space)
    best = select_best_configs(results_by_size)
    best_path, full_path = save_report(results_by_size, best, output_dir)
    print(f"\nBest configs: {best_path}")
    print(f"Full sweep:   {full_path}")


if __name__ == "__main__":
    main()
