"""Base abstractions for benchmark backends."""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from config import BenchConfig
    from workloads.base import Workload


class BackendID(enum.Enum):
    IREE_CUDA = "iree-cuda"
    IREE_CUDA_NEW = "iree-cuda-new"
    CTL_OLD_HAL = "ctl-old-hal"
    CTL_NEW_HAL = "ctl-new-hal"
    TORCH_COMPILE = "torch-compile"


@dataclass
class CompileResult:
    success: bool
    compile_time_s: float = 0.0
    artifact_path: Path | None = None
    artifact_bytes: int = 0
    error: str = ""
    phases: dict[str, float] = field(default_factory=dict)


@dataclass
class RunResult:
    success: bool
    first_run_ms: float = 0.0
    subsequent_p50_ms: float = 0.0
    subsequent_p99_ms: float = 0.0
    subsequent_min_ms: float = 0.0
    subsequent_max_ms: float = 0.0
    throughput_ops_s: float = 0.0
    all_latencies_ms: list[float] = field(default_factory=list)
    error: str = ""


class Backend(ABC):
    def __init__(self, backend_id: BackendID, config: BenchConfig):
        self.backend_id = backend_id
        self.config = config

    @property
    def name(self) -> str:
        return self.backend_id.value

    @abstractmethod
    def compile(
        self, workload: Workload, size: dict
    ) -> CompileResult:
        ...

    @abstractmethod
    def run(
        self,
        compile_result: CompileResult,
        workload: Workload,
        size: dict,
        warmup: int | None = None,
        iterations: int | None = None,
    ) -> RunResult:
        ...

    @abstractmethod
    def check_available(self) -> bool:
        ...
