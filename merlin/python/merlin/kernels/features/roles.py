"""L1/L2 features: ordered op_sequence and per-tensor memory roles, with *measured* reuse.

This is the richest layer: rather than flat booleans, it assigns each operand a role
(streaming / reusable_weight / accumulator / committed_output) and **measures** RHS reuse
from the code (register blocking on RVV; compute-per-weight-load on Gemmini). The measured
``rhs_reuse_count`` makes ``packed_rhs_policy``'s ``rhs_reuse_count >= 2`` condition applicable
to the very kernels it was mined from — not merely asserted.
"""
from __future__ import annotations

import re

from merlin.kernels.markers import target_family
from merlin.kernels.types import NormalizedKernel

_CONTRACTION = {"gemm", "matmul", "conv", "dwconv", "igemm"}

_VACC_RE = re.compile(r"\bvacc(\d+)\b")
_COMPUTE_RE = re.compile(r"compute_preloaded")
_WLOAD_RE = re.compile(r"\bmvin[23]\b|gemmini_extended\d*_mvin[23]")


def _measure_rhs_reuse(nk: NormalizedKernel, fired: dict) -> int:
    """Static proxy for how many times a packed RHS/weight is reused before reload.

    RVV: register blocking — a loaded weight vector feeds MR accumulators, so reuse == the
    number of distinct ``vacc`` registers (MR). Gemmini: compute ops per weight ``mvin``.
    Honest proxy (no runtime trip counts); recorded as a static estimate.
    """
    fam = target_family(nk.target)
    text = nk.raw_text
    if fam == "rvv":
        mr = len(set(_VACC_RE.findall(text)))
        return mr if mr else (1 if "packed_rhs" in fired else 0)
    if fam == "gemmini":
        n_compute = len(_COMPUTE_RE.findall(text))
        n_wload = len(_WLOAD_RE.findall(text))
        if n_wload == 0:
            return 1 if n_compute else 0
        return max(1, round(n_compute / n_wload))
    if fam == "exo_schedule":
        return 2 if "packed_rhs" in fired else 0  # schedule stages weights => reuse asserted
    return 1 if "packed_rhs" in fired else 0


def _op_sequence(nk: NormalizedKernel, feats_hint: dict) -> list[str]:
    """Reconstruct the ordered op sequence: primary op then fused epilogue ops."""
    text = nk.raw_text
    primary = "matmul" if nk.op in {"gemm", "matmul", "igemm"} else nk.op
    seq = [primary]
    if nk.op not in _CONTRACTION:
        return [nk.op]
    if "bias" in text:
        seq.append("bias_add")
    if any(t in text for t in ("vfcvt", "vfncvt", "acc_scale", "requant", "rescale")):
        seq.append("requant")
    if "relu" in text or "RELU" in text:
        seq.append("relu")
    elif any(t in text for t in ("vfmax", "vfmin", "clamp", "vmaxq", "vminq")):
        seq.append("clamp")
    return seq


def extract_roles(nk: NormalizedKernel, fired: dict[str, list[str]]) -> dict:
    packed_rhs = "packed_rhs" in fired
    accumulator = "accumulator_lifetime" in fired
    epilogue = "epilogue_before_commit" in fired
    widening = "__riscv_vw" in nk.raw_text or "1 << 31" in nk.raw_text or "1u << 31" in nk.raw_text
    rhs_reuse = _measure_rhs_reuse(nk, fired)
    op_sequence = _op_sequence(nk, {})

    memory_behavior: dict = {}
    if nk.op in _CONTRACTION:
        memory_behavior = {
            "lhs": {"role": "streaming_activation", "reuse_count": 1},
            "rhs": {
                "role": "reusable_weight" if packed_rhs else "streaming",
                "immutable": True,
                "reuse_count": rhs_reuse,
                "packed_once": packed_rhs,
            },
            "acc": {
                "role": "accumulator" if accumulator else "none",
                "widening": bool(widening),
                "materialized_before_epilogue": not epilogue,
            },
            "output": {"role": "committed_output"},
        }
    return {
        "op_sequence": op_sequence,
        "rhs_reuse_count": rhs_reuse,
        "memory_behavior": memory_behavior,
    }
