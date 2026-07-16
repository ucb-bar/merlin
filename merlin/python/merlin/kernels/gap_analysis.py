"""The completeness critic — "did the CCA capture EVERYTHING that makes the expert faster?"

The whole point of the CCA is to explain the expert's advantage as a set of compiler-exposable
decisions. So the honest test is: after we close every CCA divergence we can SEE, is there still a
real performance gap? If yes, the CCA is INCOMPLETE — it failed to capture a performance-determining
decision the expert makes (the "still 72% slower" signal), and that is the trigger to EXPAND the CCA
(add a new analyzer/facet), not to keep tuning the levers we already have.

``gap_analysis`` combines the CCA diff (what we can see) with the measured attainment (the real gap)
and classifies:
- **explained**   — open CCA divergences remain: we know what to close next.
- **unexplained** — no divergences left but still slower: the CCA missed something (expand it).
- **at parity**   — no gap (within tolerance): the CCA explained the expert.

Deterministic; no LLM. ``ours_perf``/``expert_perf`` are lower-is-better (cycles or wall time).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GapReport:
    attainment: float | None = None       # expert_perf / ours_perf (1 = parity, <1 = we're slower)
    pct_slower: float | None = None        # (ours_perf/expert_perf - 1) * 100
    open_divergences: list[str] = field(default_factory=list)   # CCA axes still differing (what to close)
    explained: bool = False                # the gap is (partly) explained by open CCA divergences
    unexplained_gap: bool = False          # divergences closed but still slower -> the CCA is INCOMPLETE
    verdict: str = ""

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def gap_analysis(expert_cca, ours_cca, *, ours_perf: float | None = None,
                 expert_perf: float | None = None, parity_tol: float = 0.05) -> GapReport:
    """Compare our (optimized) CCA vs the expert CCA AND the measured perf, and say whether the gap is
    explained by open CCA divergences or is UNEXPLAINED (the CCA failed to capture something)."""
    from .cca_compare import compare

    open_axes = [d.axis for d in compare(expert_cca, ours_cca)]
    attainment = pct = None
    if ours_perf and expert_perf and ours_perf > 0 and expert_perf > 0:
        attainment = expert_perf / ours_perf
        pct = (ours_perf / expert_perf - 1.0) * 100.0

    slower = attainment is not None and attainment < (1.0 - parity_tol)
    explained = slower and bool(open_axes)
    unexplained = slower and not open_axes

    if unexplained:
        verdict = (f"CCA INCOMPLETE: {pct:.0f}% slower with NO open CCA divergences — the CCA failed to "
                   "capture a performance-determining decision the expert makes. EXPAND the CCA "
                   "(add an analyzer/facet), do not keep tuning the captured levers.")
    elif explained:
        verdict = f"{pct:.0f}% slower; {len(open_axes)} open CCA divergence(s) to close: {open_axes}"
    elif slower:  # attainment known, slower, but no perf split possible — shouldn't happen; be honest
        verdict = f"{pct:.0f}% slower (gap classification indeterminate)"
    elif attainment is None:
        verdict = (f"no perf measured; {len(open_axes)} open CCA divergence(s): {open_axes}"
                   if open_axes else "no perf measured; CCA fully closed")
    else:
        verdict = "at parity (within tolerance) — the CCA explained the expert"
    return GapReport(attainment=attainment, pct_slower=pct, open_divergences=open_axes,
                     explained=explained, unexplained_gap=unexplained, verdict=verdict)
