"""Opt-in reference for PyTorch's AVX2 float32 row mean.

This models the CPU implementation used by the captured TorchAO reference,
not the mathematical meaning of ``aten.mean.dim`` and not a target lowering.
PyTorch wheel ``2.16.0.dev20261104+cpu`` (git ``7f24a19c0967``) uses the
``cascade_sum`` path of ``aten/src/ATen/native/cpu/SumKernel.cpp`` with an
eight-float AVX2 vector, four independent vector streams, and four carry
levels. ``mean_out`` divides that sum by the reduced dimension's extent.

Selection is deliberately explicit. Other CPU capabilities, PyTorch versions,
layouts and target schedules may choose a different order of float additions.
"""

from __future__ import annotations

import numpy as np

TORCH_COMMIT = "7f24a19c0967db04f2d5ba44c1e56eca489932ec"
VECTOR_WIDTH = 8
ILP = 4
LEVELS = 4


def avx2_f32_row_mean(rows: np.ndarray) -> np.ndarray:
    """Replay the captured AVX2 sum for contiguous power-of-two f32 rows.

    The accepted domain covers the TinyLlama RMSNorm K=2048 case. Other inputs
    fail closed until checked against the exact source CPU implementation.
    """
    rows = np.asarray(rows)
    if rows.dtype != np.float32 or rows.ndim != 2 or not rows.flags.c_contiguous:
        raise ValueError("expected contiguous [rows,K] float32 input")
    count = rows.shape[1]
    if count < 32 or count & (count - 1):
        raise ValueError("this source fidelity mode requires power-of-two K >= 32")
    if not np.isfinite(rows).all():
        raise ValueError("finite values required by this bounded fidelity model")

    vector_count = count // VECTOR_WIDTH
    size_ilp = vector_count // ILP
    level_power = max(4, (size_ilp - 1).bit_length() // LEVELS)
    level_step = 1 << level_power
    level_mask = level_step - 1
    means = np.empty(rows.shape[0], dtype=np.float32)
    for row_index, row in enumerate(rows):
        accumulators = np.zeros((LEVELS, ILP, VECTOR_WIDTH), dtype=np.float32)
        i = 0
        while i + level_step <= size_ilp:
            for _ in range(level_step):
                for stream in range(ILP):
                    offset = (i * ILP + stream) * VECTOR_WIDTH
                    accumulators[0, stream] += row[offset : offset + VECTOR_WIDTH]
                i += 1
            for level in range(1, LEVELS):
                accumulators[level] += accumulators[level - 1]
                accumulators[level - 1].fill(0)
                if i & (level_mask << (level * level_power)):
                    break
        while i < size_ilp:
            for stream in range(ILP):
                offset = (i * ILP + stream) * VECTOR_WIDTH
                accumulators[0, stream] += row[offset : offset + VECTOR_WIDTH]
            i += 1
        for level in range(1, LEVELS):
            accumulators[0] += accumulators[level]
        streams = accumulators[0]
        for stream in range(1, ILP):
            streams[0] += streams[stream]
        total = np.float32(0)
        for lane in streams[0]:
            total = np.float32(total + lane)
        means[row_index] = np.float32(total / np.float32(count))
    return means
