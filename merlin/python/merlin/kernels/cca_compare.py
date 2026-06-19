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
    for facet in ("compute", "vector", "spatial", "dataflow"):
        for k, (ve, vo) in _populated_pairs(getattr(expert, facet), getattr(ours, facet)).items():
            out.append(Divergence(axis=f"{facet}.{k}", expert=ve, ours=vo,
                                  backend=backend, evidence=list(ev)))
    return out
