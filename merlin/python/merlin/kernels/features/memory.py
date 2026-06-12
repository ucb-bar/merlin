"""Accumulator / memory-state features.

Decision recorded: *does an accumulator live across the contraction* (and is it a widening
accumulator). The accumulator-lifetime motif is the seed of the ``accumulator_commit``
abstraction candidate.
"""
from __future__ import annotations

from merlin.kernels.types import NormalizedKernel


def extract_memory(nk: NormalizedKernel, fired: dict[str, list[str]]) -> dict:
    accumulator = "accumulator_lifetime" in fired
    # Widening accumulation: int8x8->int32 (RVV vw*) or Gemmini int32 accumulator tiles.
    text = nk.raw_text
    widening = bool(accumulator and (
        "__riscv_vw" in text or "1 << 31" in text or "1u << 31" in text or "0x40000000" in text
    ))
    return {"accumulator": bool(accumulator), "accumulator_widening": widening}
