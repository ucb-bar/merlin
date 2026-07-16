"""Multi-level CCA comparator — structured, typed divergences (target-agnostic).

Given an expert CCA and ours (both lifted to the Common Compute Abstraction, ideally at the asm
level — the authoritative substrate), emit ``Divergence`` records: one per populated facet field
that differs. These feed the action catalog (``action_catalog.py``), which maps each to a typed
compiler change. Comparison is per-facet, so a target only diffs the facets it has (vector for
RVV, spatial for gemmini, …) — nothing RVV-specific here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .cca import CCA


@dataclass
class Divergence:
    axis: str                 # "compute.contraction_form", "vector.lmul", ...
    expert: Any
    ours: Any
    backend: str
    evidence: list[str] = field(default_factory=list)   # kernel ids justifying the expert value


def _populated_pairs(fa, fb) -> dict[str, tuple]:
    if fa is None or fb is None:
        return {}
    out = {}
    for k, va in asdict(fa).items():
        vb = asdict(fb).get(k)
        if va is not None and vb is not None and va != vb:
            out[k] = (va, vb)
    return out


def compare(expert: CCA, ours: CCA, *, evidence: list[str] | None = None) -> list[Divergence]:
    """expert-vs-ours CCA -> typed Divergences (the authoritative gap when both are asm-lifted)."""
    ev = evidence or [expert.provenance.get("source", "expert")]
    backend = (expert.backend or ours.backend or ["?"])[0]
    out: list[Divergence] = []
    for facet in ("compute", "vector", "memory", "spatial", "dataflow"):
        for k, (ve, vo) in _populated_pairs(getattr(expert, facet), getattr(ours, facet)).items():
            out.append(Divergence(axis=f"{facet}.{k}", expert=ve, ours=vo,
                                  backend=backend, evidence=list(ev)))

    # register_block (MR) is the #1 GEMM data-movement decision and the one we were structurally
    # blind to: _populated_pairs only fires when BOTH sides report it, but an UNBLOCKED kernel lifts
    # to register_block=None (== MR 1, no row reuse), so "expert blocks to MR=k, ours doesn't" never
    # surfaced. Compare MR-aware (None -> MR 1) so the gap becomes a real, routable divergence.
    def _mr(cca):
        rb = getattr(cca.compute, "register_block", None) if cca and cca.compute else None
        if isinstance(rb, (tuple, list)) and rb and isinstance(rb[0], int):
            return rb[0]
        return 1
    emr, omr = _mr(expert), _mr(ours)
    if emr > omr and not any(d.axis == "compute.register_block" for d in out):
        out.append(Divergence(axis="compute.register_block",
                              expert=expert.compute.register_block,
                              ours=(ours.compute.register_block if (ours and ours.compute) else None),
                              backend=backend, evidence=list(ev)))
    return out
