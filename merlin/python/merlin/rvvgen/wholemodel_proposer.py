"""Whole-model beam proposer: propose the byte-traffic-ranked whole-model levers, not a kernel diff.

The default CCA proposer (``fork_from_action.propose_forks_from_cca``) diffs the parent's emitted
kernel CCA against a single EXPERT KERNEL objdump. That works for GEMM-microkernel levers, but it is
structurally blind to the levers that actually move whole-model time — a materialized
``linalg.transpose`` (38% of byte-traffic, 57% of openvla time) and an unvectorized reduction
(softmax/layernorm) are GRAPH properties, not properties of a GEMM kernel, so no kernel-vs-kernel diff
can surface them. Measured on the board: a scoped beam with a GEMM expert objdump produced only
spurious ``expert='na'`` divergences and zero useful forks.

This proposer instead offers the KNOWN forkable whole-model levers, ranked by measured byte-traffic
relevance (the census in ``out/artifacts/ceiling/model_op_census.json``: transpose 38% > matmul 26% >
reduce/softmax > activations). It ignores the CCA divergence list (the beam still lifts + records it
for audit) and lets the board wall + the two-phase validate decide which levers and which STACK win —
which is the point: the beam discovers the composition, ranked by real e2e wall, rather than being
told it.

Contract: ``(divergences, knobs) -> [ForkProposal]`` — a drop-in for ``beam.run_beam(proposer=...)``.
Each proposal MERGES one new feature into the parent's ``compiler_features`` (so depth-N accumulates a
stack), dropping a proposal that cannot compose (two full-schedule-replacement features clobber).
"""
from __future__ import annotations

from typing import Any

from ..kernels.rvv_knobs import ForkProposal

# Whole-model levers, most-impactful first by measured byte-traffic / e2e attribution. Each entry is
# (feature_name, is_full_schedule_replacement). The schedule-replacement one is the matmul register
# block (it supersedes the plain vf recipe); the rest are additive passes that compose on top.
RANKED_LEVERS: list[tuple[str, bool]] = [
    ("fuse_transpose_b", False),                          # transpose: 38% byte-traffic, measured -6.5% openvla
    ("accumulator_resident_wholemodel_vf_mrpad", True),   # matmul MR register block: 1.49x rdt2 matmul bucket
    ("vectorize_reduction", False),                       # reduce/softmax: 2nd byte-traffic family, was unvectorized
    ("erase_self_copy", False),                           # envelope: per-tile memrefCopy elimination
    ("vectorized_transcendental_activation", False),      # gelu/sigmoid/silu: closes the 10-17x activation gap
]


def _composes(features: list[str]) -> bool:
    """True iff the feature set is co-enable-able (no two full-schedule-replacement features)."""
    from ..llvmlower import impr_features as I
    try:
        I.normalize(features)
    except Exception:  # CompositionError (two schedule_replace) or unknown feature
        return False
    # normalize does not itself reject two schedule_replace on every path; check explicitly.
    reps = [f for f in features if getattr(I.get(f), "schedule_replace", False)]
    return len(reps) <= 1


def propose_wholemodel_levers(divergences: Any, knobs: dict[str, Any]) -> list[ForkProposal]:
    """Propose one fork per not-yet-enabled whole-model lever, merged onto the parent's features.

    ``divergences`` is accepted (and ignored) to satisfy the beam's proposer contract; the beam still
    lifts and records the parent's CCA for the per-fork audit. Ranking is by byte-traffic relevance;
    the board wall picks the winners and their stack across beam depth."""
    parent_feats = list(knobs.get("compiler_features", []) or [])
    out: list[ForkProposal] = []
    for feat, _is_replace in RANKED_LEVERS:
        if feat in parent_feats:
            continue
        merged = parent_feats + [feat]
        if not _composes(merged):
            # e.g. mrpad on top of a parent that already carries the vf schedule-replacement.
            # Try replacing the conflicting schedule-replacement feature instead of stacking.
            from ..llvmlower import impr_features as I
            base = [f for f in parent_feats if not getattr(I.get(f), "schedule_replace", False)]
            merged = base + [feat]
            if not _composes(merged):
                continue
        out.append(ForkProposal(
            overrides={"compiler_features": merged},
            lever="feature",
            targets=f"wholemodel:{feat}",
            evidence=["census:byte-traffic", f"lever:{feat}"],
            forkable=True,
            note=f"enable whole-model lever {feat} (byte-traffic ranked)",
        ))
    return out
