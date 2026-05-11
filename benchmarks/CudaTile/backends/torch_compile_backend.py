"""torch.compile backend using inductor (auto-generates Triton kernels)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from backends.base import Backend, BackendID, CompileResult, RunResult

if TYPE_CHECKING:
    from config import BenchConfig
    from workloads.base import Workload


class TorchCompileBackend(Backend):
    def __init__(self, config: BenchConfig):
        super().__init__(BackendID.TORCH_COMPILE, config)
        self._torch = None
        self._compiled_cache: dict[str, object] = {}

    def check_available(self) -> bool:
        try:
            import torch

            if not torch.cuda.is_available():
                print(
                    "  [torch-compile] torch.cuda not available. "
                    "Install CUDA-enabled PyTorch: "
                    "pip install torch --index-url https://download.pytorch.org/whl/cu124"
                )
                return False
            return True
        except ImportError:
            print("  [torch-compile] torch not installed.")
            return False

    def _get_torch(self):
        if self._torch is None:
            import os

            if self.config.gpu_index is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu_index)
            import torch

            self._torch = torch
        return self._torch

    def compile(self, workload: Workload, size: dict) -> CompileResult:
        torch = self._get_torch()

        module, example_inputs = workload.torch_module(size)
        if module is None:
            return CompileResult(
                success=False,
                error="workload does not provide a PyTorch module",
            )

        module = module.cuda().eval()
        cuda_inputs = [
            x.cuda() if isinstance(x, torch.Tensor) else x
            for x in example_inputs
        ]

        try:
            t0 = time.perf_counter()
            compiled = torch.compile(
                module, backend="inductor", mode="max-autotune"
            )
            # Trigger compilation.
            with torch.no_grad():
                compiled(*cuda_inputs)
            torch.cuda.synchronize()
            compile_time = time.perf_counter() - t0
        except Exception as e:
            return CompileResult(success=False, error=str(e)[:500])

        cache_key = f"{workload.name}_{size.get('desc', 'default')}"
        self._compiled_cache[cache_key] = (compiled, cuda_inputs, module)

        return CompileResult(
            success=True,
            compile_time_s=compile_time,
        )

    def run(
        self,
        compile_result: CompileResult,
        workload: Workload,
        size: dict,
        warmup: int | None = None,
        iterations: int | None = None,
    ) -> RunResult:
        if not compile_result.success:
            return RunResult(success=False, error="compilation failed")

        torch = self._get_torch()
        warmup = warmup if warmup is not None else self.config.warmup
        iterations = iterations if iterations is not None else self.config.iterations

        cache_key = f"{workload.name}_{size.get('desc', 'default')}"
        entry = self._compiled_cache.get(cache_key)
        if entry is None:
            return RunResult(success=False, error="compiled module not in cache")
        compiled, cuda_inputs, _ = entry

        # Phase 1: first-run latency (already compiled, measures dispatch overhead).
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        with torch.no_grad():
            compiled(*cuda_inputs)
        end_event.record()
        torch.cuda.synchronize()
        first_run_ms = start_event.elapsed_time(end_event)

        # Phase 2: warmup.
        for _ in range(warmup):
            with torch.no_grad():
                compiled(*cuda_inputs)
        torch.cuda.synchronize()

        # Phase 3: measured iterations.
        latencies = []
        for _ in range(iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.no_grad():
                compiled(*cuda_inputs)
            end_event.record()
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))

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
