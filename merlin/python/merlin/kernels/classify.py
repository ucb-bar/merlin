"""Map a kernel's typed features onto the canonical motif vocabulary.

This is the single place that defines which feature values count as a present "motif"
(an optimization *decision*). Aggregation and policy promotion consume motif names, so
keeping the mapping here means the vocabulary is defined exactly once.

Most motifs map 1:1 from a marker-derived feature. ``accumulator_commit`` is a *composite*
decision (accumulator live across an epilogue on a contraction op) — deliberately narrower
than the raw ``epilogue_before_commit`` marker, which also fires on elementwise clamp ops.
"""
from __future__ import annotations

_CONTRACTION_OPS = {"gemm", "matmul", "conv", "dwconv", "igemm", "trmm", "gemv"}

# motif name -> predicate over (features, op)
_MOTIF_RULES = {
    "packed_rhs": lambda f, op: bool(f.get("packed_rhs")),
    "accumulator_lifetime": lambda f, op: bool(f.get("accumulator")),
    "epilogue_before_commit": lambda f, op: bool(f.get("epilogue_fusion")),
    "accumulator_commit": lambda f, op: bool(
        f.get("accumulator") and f.get("epilogue_fusion") and op in _CONTRACTION_OPS
    ),
    # Reuse is now *measured*: a packed RHS that is actually reused >=2x is the resident-
    # tensor signal (distinct from merely "a pack happened").
    "reused_packed_rhs": lambda f, op: bool(
        f.get("packed_rhs") and f.get("rhs_reuse_count", 0) >= 2
    ),
    "vector_length_polymorphic": lambda f, op: f.get("vector_length_strategy") == "scalable",
    "tiling_blocking": lambda f, op: bool(f.get("tiling")),
    "double_buffering": lambda f, op: bool(f.get("double_buffering")),
    "weight_stationary_dataflow": lambda f, op: f.get("dataflow") == "weight_stationary",
    "many_small_dispatches": lambda f, op: (
        f.get("dispatch_metrics", {}).get("n_dispatches", 0) >= 20
        and f.get("dispatch_metrics", {}).get("small_dispatch_fraction", 0) >= 0.5
    ),
    "intrinsic_lowering": lambda f, op: bool(f.get("target_specific_config")),
}


def classify_motifs(features: dict, op: str = "unknown") -> set[str]:
    """Return the set of canonical motif names present given ``features`` and ``op``."""
    return {name for name, pred in _MOTIF_RULES.items() if pred(features, op)}
