"""Compatibility exports for the shared matrix calling-convention adapter."""

from merlin.compiler.lower.matrix import (
    DEFAULT_SCRATCH_BYTES,
    ShimUnit,
    emit_translation_unit,
    is_batched,
    scratch_bytes_for,
)

__all__ = ["DEFAULT_SCRATCH_BYTES", "ShimUnit", "emit_translation_unit", "is_batched", "scratch_bytes_for"]
