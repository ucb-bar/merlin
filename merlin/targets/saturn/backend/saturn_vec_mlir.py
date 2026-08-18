"""Saturn's thin binding to the generic RVV vector-family MLIR emitter.

The emitter is target-agnostic (it lowers the VECTOR_MAP / VREDUCE command-buffer opcodes to an
``linalg``/``vector`` MLIR module) and lives in its generic home
:mod:`merlin.runtime.backends.rvv_vec_mlir`. This module re-exports it under the saturn backend's own
namespace so the relocated backend can reach it as a sibling without duplicating the generic code
under a target-named directory.
"""
from __future__ import annotations

from merlin.runtime.backends.rvv_vec_mlir import *  # noqa: F401,F403
from merlin.runtime.backends.rvv_vec_mlir import (  # noqa: F401
    emit_mlir,
    lower_rvv,
    run_host,
)
