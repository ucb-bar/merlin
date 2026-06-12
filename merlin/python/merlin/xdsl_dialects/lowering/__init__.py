"""Staged lowering across the core dialects.

linalg input -> contract -> schedule -> interface -> target -> runtime -> command
buffer. Stages are plain module->module transforms; cross-op legality lives in
``analyses``. Entry point: :func:`pipeline.lower_repeated_rhs_matmul`.
"""
from __future__ import annotations

from .dispatch_program import (DispatchProgram, build_dispatch_program,
                               lower_model_to_dispatch_program, prune_dead_nodes,
                               verify_program)
from .interface_lowering import LoweringError
from .outline import OutlineError, OutlineResult, outline_dispatches
from .passes import CATALOG, DialectPlaneResult, catalog, run_dialect_plane
from .schedule_dispatch import Schedule, partition_dispatches
from .pipeline import LoweringResult, execute, lower_repeated_rhs_matmul
