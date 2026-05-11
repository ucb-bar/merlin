"""IREE backend: supports cuda, cuda_tile, cuda_new HALs."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from backends.base import Backend, BackendID, CompileResult, RunResult

if TYPE_CHECKING:
    from config import BenchConfig
    from workloads.base import Workload

_CODEGEN_FLAGS = {
    BackendID.IREE_CUDA: {
        "hal_backend": "cuda",
        "device": "cuda",
    },
    BackendID.IREE_CUDA_NEW: {
        "hal_backend": "cuda",
        "device": "cuda_new",
    },
    BackendID.CTL_OLD_HAL: {
        "hal_backend": "cuda_tile",
        "device": "cuda_tile",
    },
    BackendID.CTL_NEW_HAL: {
        "hal_backend": "cuda_tile",
        "device": "cuda_new",
    },
}


class IREEBackend(Backend):
    def __init__(self, backend_id: BackendID, config: BenchConfig):
        super().__init__(backend_id, config)
        flags = _CODEGEN_FLAGS[backend_id]
        self._hal_backend = flags["hal_backend"]
        self._device = flags["device"]

    def check_available(self) -> bool:
        try:
            self.config.iree_compile
            self.config.iree_benchmark_module
            return True
        except FileNotFoundError:
            return False

    def compile(self, workload: Workload, size: dict) -> CompileResult:
        out_dir = (
            self.config.output_dir / "artifacts" / self.name / workload.name
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        desc = size.get("desc", "default")
        vmfb_path = out_dir / f"{workload.name}_{desc}.vmfb"
        mlir_path = out_dir / f"{workload.name}_{desc}.mlir"

        # Export MLIR from PyTorch via Turbine AOT.
        export_time = 0.0
        torch_result = workload.torch_module(size)
        if torch_result is not None:
            try:
                export_time = self._export_turbine(
                    torch_result, mlir_path, workload
                )
            except Exception as e:
                return CompileResult(
                    success=False,
                    error=f"Turbine export failed: {e}",
                )
        else:
            mlir_text = workload.mlir_source(size)
            if mlir_text is None:
                return CompileResult(
                    success=False,
                    error="workload has no torch_module or mlir_source",
                )
            mlir_path.write_text(mlir_text)

        return self._compile_mlir(
            mlir_path, vmfb_path, export_time,
            input_type="torch" if torch_result is not None else None,
        )

    def _export_turbine(
        self, torch_result: tuple, mlir_path: Path, workload: Workload
    ) -> float:
        import torch
        import iree.turbine.aot as aot

        module, example_inputs = torch_result
        module = module.cpu().eval()
        cpu_inputs = [
            x.cpu() if isinstance(x, torch.Tensor) else x
            for x in example_inputs
        ]

        t0 = time.perf_counter()
        exported = aot.export(module, *cpu_inputs)
        exported.save_mlir(str(mlir_path))
        export_time = time.perf_counter() - t0
        return export_time

    def _compile_mlir(
        self,
        mlir_path: Path,
        vmfb_path: Path,
        export_time: float = 0.0,
        input_type: str | None = None,
    ) -> CompileResult:
        cmd = [
            self.config.iree_compile,
            str(mlir_path),
            f"--iree-hal-target-backends={self._hal_backend}",
            "-o",
            str(vmfb_path),
        ]
        if input_type:
            cmd.append(f"--iree-input-type={input_type}")
        if self._hal_backend == "cuda_tile":
            cmd.extend([
                f"--iree-cuda-tile-sm-arch={self.config.sm_arch}",
                "--iree-cuda-tile-enable-codegen=true",
            ])
        elif self._hal_backend == "cuda":
            cmd.append(f"--iree-cuda-target={self.config.sm_arch}")

        try:
            t0 = time.perf_counter()
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
                env=self.config.compile_env,
            )
            iree_compile_time = time.perf_counter() - t0
        except subprocess.TimeoutExpired:
            return CompileResult(
                success=False, error="compilation timed out (300s)"
            )

        total_compile_time = export_time + iree_compile_time

        if result.returncode != 0:
            return CompileResult(
                success=False,
                compile_time_s=total_compile_time,
                error=result.stderr[:500],
                phases={"export": export_time, "iree_compile": iree_compile_time},
            )

        return CompileResult(
            success=True,
            compile_time_s=total_compile_time,
            artifact_path=vmfb_path,
            artifact_bytes=vmfb_path.stat().st_size,
            phases={"export": export_time, "iree_compile": iree_compile_time},
        )

    def run(
        self,
        compile_result: CompileResult,
        workload: Workload,
        size: dict,
        warmup: int | None = None,
        iterations: int | None = None,
    ) -> RunResult:
        if not compile_result.success or not compile_result.artifact_path:
            return RunResult(success=False, error="no compiled artifact")

        warmup = warmup if warmup is not None else self.config.warmup
        iterations = iterations if iterations is not None else self.config.iterations
        vmfb = compile_result.artifact_path

        # Phase 1: first-run latency via iree-run-module (single invocation).
        first_run_ms = self._measure_first_run(vmfb, workload, size)

        # Phase 2: steady-state via iree-benchmark-module.
        # Retry with fewer iterations on OOM.
        bench_result = self._run_benchmark_module(
            vmfb, workload, size, warmup, iterations
        )
        if bench_result is None and iterations > 10:
            bench_result = self._run_benchmark_module(
                vmfb, workload, size, warmup, max(10, iterations // 5)
            )
        if bench_result is None:
            return RunResult(
                success=True,
                first_run_ms=first_run_ms,
                error="iree-benchmark-module failed",
            )

        latencies = bench_result["latencies_ms"]
        if not latencies:
            return RunResult(
                success=True,
                first_run_ms=first_run_ms,
                error="no latency data from benchmark",
            )

        sorted_lat = sorted(latencies)
        n = len(sorted_lat)
        total_s = sum(latencies) / 1000.0

        return RunResult(
            success=True,
            first_run_ms=first_run_ms,
            subsequent_p50_ms=sorted_lat[n // 2],
            subsequent_p99_ms=sorted_lat[int(n * 0.99)],
            subsequent_min_ms=sorted_lat[0],
            subsequent_max_ms=sorted_lat[-1],
            throughput_ops_s=n / total_s if total_s > 0 else 0,
            all_latencies_ms=latencies,
        )

    def _measure_first_run(
        self, vmfb: Path, workload: Workload, size: dict
    ) -> float:
        func_name = workload.entry_function
        inputs = workload.input_specs(size)
        cmd = [
            self.config.iree_run_module,
            f"--device={self._device}",
            f"--module={vmfb}",
            f"--function={func_name}",
        ]
        for inp in inputs:
            cmd.append(f"--input={inp}")

        try:
            t0 = time.perf_counter()
            subprocess.run(cmd, capture_output=True, timeout=60, env=self.config.env)
            return (time.perf_counter() - t0) * 1000.0
        except subprocess.TimeoutExpired:
            return -1.0

    def _run_benchmark_module(
        self,
        vmfb: Path,
        workload: Workload,
        size: dict,
        warmup: int,
        iterations: int,
    ) -> dict | None:
        func_name = workload.entry_function
        inputs = workload.input_specs(size)
        cmd = [
            self.config.iree_benchmark_module,
            f"--device={self._device}",
            f"--module={vmfb}",
            f"--function={func_name}",
            f"--benchmark_repetitions={iterations}",
            f"--benchmark_min_warmup_time={warmup * 0.01:.2f}",
            "--benchmark_min_time=0.01s",
            "--benchmark_format=json",
        ]
        for inp in inputs:
            cmd.append(f"--input={inp}")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
                env=self.config.env,
            )
        except subprocess.TimeoutExpired:
            return None

        if result.returncode != 0:
            return None

        return self._parse_benchmark_json(result.stdout)

    def _parse_benchmark_json(self, stdout: str) -> dict | None:
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            return self._parse_benchmark_text(stdout)

        latencies = []
        for bm in data.get("benchmarks", []):
            name = bm.get("name", "")
            if "_mean" in name or "_median" in name or "_stddev" in name or "_cv" in name:
                continue
            real_time = bm.get("real_time", 0)
            time_unit = bm.get("time_unit", "ns")
            if time_unit == "ns":
                latencies.append(real_time / 1e6)
            elif time_unit == "us":
                latencies.append(real_time / 1e3)
            elif time_unit == "ms":
                latencies.append(real_time)
            elif time_unit == "s":
                latencies.append(real_time * 1e3)

        return {"latencies_ms": latencies}

    def _parse_benchmark_text(self, stdout: str) -> dict | None:
        latencies = []
        for line in stdout.splitlines():
            match = re.search(
                r"([\d.]+)\s+(ns|us|ms|s)\s+[\d.]+\s+(ns|us|ms|s)\s+\d+",
                line,
            )
            if match:
                val = float(match.group(1))
                unit = match.group(2)
                if unit == "ns":
                    latencies.append(val / 1e6)
                elif unit == "us":
                    latencies.append(val / 1e3)
                elif unit == "ms":
                    latencies.append(val)
                elif unit == "s":
                    latencies.append(val * 1e3)
        if not latencies:
            return None
        return {"latencies_ms": latencies}
