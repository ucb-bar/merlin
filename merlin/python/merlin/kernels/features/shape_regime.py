"""Shape-regime features (spec §4): generalize on *regimes*, never on exact (M,N,K).

A policy stated as ``if M == 64 and N == 64`` is a kernel-specific trick; one stated over
regimes (``capacity_fit``, ``tail_heavy``, ``memory_bound``) is compiler knowledge. This
extractor derives the regime vocabulary from a kernel's parsed shape when a full contraction
shape (M, N, K) is available, and is honestly ``["unknown"]`` otherwise (XNNPACK/OpenBLAS
filenames carry only register-tile MRxNR, not problem shapes).
"""
from __future__ import annotations

from merlin.kernels.types import NormalizedKernel

_DTYPE_BYTES = {"i8": 1, "u8": 1, "i32": 4, "f16": 2, "bf16": 2, "f32": 4, "f64": 8,
                "c32": 8, "c64": 16}
# Same residency budget the Stage-D capacity sweep uses (128 KiB scratchpad-class store).
RESIDENT_BUDGET_BYTES = 131072
# Arithmetic-intensity cutover (flops/byte) between memory- and compute-bound regimes.
_AI_CUTOVER = 16.0
_SKINNY_DIM = 16  # derived-ok: workload shape-classification threshold, not a hardware fact
_TAIL_DIVISOR = 16


def extract_shape_regime(nk: NormalizedKernel, fired: dict) -> dict:
    """Return ``{"shape_regime": {...}}`` derived from the kernel's parsed shape."""
    s = nk.shape or {}
    if not {"M", "N", "K"} <= set(s):
        return {"shape_regime": {"regime": ["unknown"]}}
    M, N, K = int(s["M"]), int(s["N"]), int(s["K"])
    eb = _DTYPE_BYTES.get(nk.dtype, 4)
    acc_b = 4 if nk.dtype in ("i8", "u8") else max(eb, 4)  # widening accumulator
    rhs_bytes = K * N * eb
    working_set = M * K * eb + rhs_bytes + M * N * acc_b
    bytes_moved = M * K * eb + rhs_bytes + M * N * eb  # each tensor touched once
    ai = (2.0 * M * N * K) / bytes_moved if bytes_moved else 0.0

    regime: list[str] = []
    if min(M, N) <= _SKINNY_DIM:
        regime.append("skinny")
    elif M == N == K and M >= 256:
        regime.append("large_square")
    elif working_set <= RESIDENT_BUDGET_BYTES:
        regime.append("small")
    if K % _TAIL_DIVISOR or N % _TAIL_DIVISOR:
        regime.append("tail_heavy")
    regime.append("memory_bound" if ai < _AI_CUTOVER else "compute_bound")
    regime.append("capacity_fit" if rhs_bytes <= RESIDENT_BUDGET_BYTES else "capacity_overflow")

    return {"shape_regime": {
        "regime": regime,
        "working_set_bytes": working_set,
        "rhs_size_bytes": rhs_bytes,
        "arithmetic_intensity": round(ai, 2),
        "k_divisible_16": K % 16 == 0,
    }}
