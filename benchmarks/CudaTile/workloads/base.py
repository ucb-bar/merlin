"""Base workload abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Workload(ABC):
    name: str
    level: int
    sizes: list[dict] = field(default_factory=list)
    acceptance_ratio: float = 1.5

    @property
    def entry_function(self) -> str:
        return "main"

    @abstractmethod
    def torch_module(self, size: dict) -> tuple:
        """Return (nn.Module, [example_inputs]). Primary source of truth."""
        ...

    @abstractmethod
    def input_specs(self, size: dict) -> list[str]:
        """Return iree-run-module --input= specs (e.g. '256x256xf32')."""
        ...

    def mlir_source(self, size: dict) -> str | None:
        """Optional hand-written MLIR for direct codegen testing."""
        return None
