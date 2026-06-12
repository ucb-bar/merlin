"""Packing / packed-RHS features.

Decision recorded: *is a weight/RHS operand packed and reused*, not how it is packed. The
``packed_rhs`` motif fires on the source-specific markers (RVV pointer-advance,
Gemmini ``mvin2/3`` staging, AVX ``B_reg`` staging). ``packing`` is the broader "any operand
is laid out / staged for reuse" decision.
"""
from __future__ import annotations

from merlin.kernels.types import NormalizedKernel


def extract_packing(nk: NormalizedKernel, fired: dict[str, list[str]]) -> dict:
    packed_rhs = "packed_rhs" in fired
    # packing is true if a packed RHS was detected, or the kernel stages operands for a
    # matmul/conv (intrinsic lowering present on a contraction op).
    packing = packed_rhs or (
        "intrinsic_lowering" in fired and nk.op in {"gemm", "matmul", "conv", "dwconv"}
    )
    return {"packing": bool(packing), "packed_rhs": bool(packed_rhs)}
