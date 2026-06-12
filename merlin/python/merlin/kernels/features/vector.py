"""Vector-length strategy features.

Decision recorded: *is the loop vector-length-agnostic* (scalable, RVV ``vsetvl`` style) vs
*fixed-width* (AVX/NEON) vs *not applicable* (systolic, e.g. Gemmini). Never records the
concrete VLEN/LMUL — that is a constant, not a decision.
"""
from __future__ import annotations

from merlin.kernels.markers import target_family
from merlin.kernels.types import NormalizedKernel


def extract_vector(nk: NormalizedKernel, fired: dict[str, list[str]]) -> dict:
    fam = target_family(nk.target)
    if "vector_length_polymorphic" in fired:
        return {"vector_length_strategy": "scalable", "tail_strategy": "predicated_or_vl_loop"}
    if fam in {"avx", "neon"}:
        return {"vector_length_strategy": "fixed", "tail_strategy": "fixed_width"}
    return {"vector_length_strategy": "na", "tail_strategy": "na"}
