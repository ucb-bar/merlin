"""Generate the merlin-target-<name>/ repository skeleton and its artifacts.

Each generator returns a list of :class:`merlin.common.artifacts.Artifact`. The pipeline
writes them under the output directory and then backfills AGENT.md coverage.
"""
from __future__ import annotations

from . import target_repo, xdsl, mlir_scaffold, zephyr_module, runtime_adapter, llvm_plan

__all__ = [
    "target_repo",
    "xdsl",
    "mlir_scaffold",
    "zephyr_module",
    "runtime_adapter",
    "llvm_plan",
]
