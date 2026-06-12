"""Asynchrony features: double buffering / DMA-compute overlap.

Decision recorded: *does the kernel overlap data movement with compute* via manual double
buffering (ping-pong scratchpad regions). Seeds an ``async_pipeline`` abstraction candidate.
"""
from __future__ import annotations

from merlin.kernels.types import NormalizedKernel


def extract_async_ops(nk: NormalizedKernel, fired: dict[str, list[str]]) -> dict:
    double_buffering = "double_buffering" in fired
    # DMA/compute overlap is implied by double buffering on a staged-memory target.
    dma_overlap = bool(double_buffering)
    return {"double_buffering": bool(double_buffering), "dma_overlap": dma_overlap}
