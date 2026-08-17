"""Saturn's thin binding to the generic RVV vector-family C driver generator.

The generator is target-agnostic (a stripmined RVV kernel over the VECTOR_MAP / VREDUCE command-buffer
opcodes) and lives in its generic home :mod:`merlin.runtime.backends.rvv_vec_codegen`. This module
re-exports it under the saturn backend's own namespace so the relocated backend can reach it as a
sibling without duplicating the generic code under a target-named directory.
"""
from __future__ import annotations

from merlin.runtime.backends.rvv_vec_codegen import *  # noqa: F401,F403
from merlin.runtime.backends.rvv_vec_codegen import (  # noqa: F401
    VecCodegenError,
    generate_driver,
)
