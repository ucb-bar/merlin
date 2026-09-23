"""Shared evidence ordering, independent of DSE ranking policy."""

from __future__ import annotations

# Confidence WEIGHTS (not performance measurements). Scale priority_score only.
CONFIDENCE_WEIGHTS: dict[str, float] = {
    "measured": 1.0,
    "trace_derived": 0.8,
    "calibrated": 0.7,
    "structural_bound": 0.55,
    "analytical": 0.4,
    "assumed": 0.2,
}


def confidence_for(evidence_type: str) -> float:
    """Confidence weight for an evidence tag (unknown tags fall back to ``assumed``)."""
    return CONFIDENCE_WEIGHTS.get(evidence_type, CONFIDENCE_WEIGHTS["assumed"])


# Strongest -> weakest. Index in this tuple is the strength rank (lower == stronger).
EVIDENCE_TYPES: tuple[str, ...] = (
    "measured",
    "trace_derived",
    "calibrated",
    "structural_bound",
    "analytical",
    "assumed",
)


def _rank(evidence_type: str) -> int:
    try:
        return EVIDENCE_TYPES.index(evidence_type)
    except ValueError:
        return len(EVIDENCE_TYPES)  # unknown == weaker than anything known


def weakest_evidence(tags: list[str] | tuple[str, ...]) -> str:
    """Return the weakest (softest) evidence tag among ``tags``.

    An axis that reduces several cost components is only as trustworthy as its softest input,
    so the combined evidence is the weakest of the component tags. Defaults to ``assumed``.
    """
    if not tags:
        return "assumed"
    return max(tags, key=_rank)
