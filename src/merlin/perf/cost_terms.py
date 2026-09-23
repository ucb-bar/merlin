"""Every resource a compiler edit can move, priced only where a rate was measured.

WHY THIS EXISTS. An authoring loop scored candidates on four command-buffer quantities --
movement bytes, MACs, dispatches, contraction MACs -- while the host lane was 95.8% of the measured
cycles. Six of seven optimizations found by hand were invisible to that scoreboard or scored as
regressions, and every iteration reported an exact zero delta. Adding host operation counters fixed
the invisible ones. It did NOT fix the harder half: an edit that moves work BETWEEN resources still
reads as a regression on whichever term it spent.

Measured examples of exactly that, from one model:

* moving a quantization epilogue onto the accelerator's store path removed 146.2M host operations
  and raised accelerator dispatches from 23,875 to 276,189 -- an 11.6x rise on a scored term;
* decomposing convolutions into accumulating matmuls removed 553.3M host operations (-32.75%) and
  raised DRAM traffic 24% and mesh issue cycles 21.6%.

Both are plausibly large wins and both score as losses under any rule that treats a raised term as
bad. Deciding requires an EXCHANGE RATE between resources, and inventing one is worse than having
none: a guessed rate silently decides which optimizations the loop pursues.

WHAT THIS MODULE DOES, AND REFUSES TO DO. It enumerates the terms an edit can move, keeps them
separate, and prices a composite ONLY from rates whose provenance is recorded. A term with no
measured rate does NOT get weight zero -- that is the bug that hides a resource -- it makes the
composite `UNPRICED`, and the per-term deltas are still reported so the reader sees what moved.
This mirrors the asymmetry `merlin.perf.movement_balance` already enforces for the roofline ridge:
say precisely what the evidence licenses, and refuse the rest.

Target-neutral: term names, units, rates and provenance are all supplied by the caller from its own
measurements. This module never names a target, a resource, or a rate of its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

__all__ = ["Rate", "MEASURED", "DERIVED", "UNPRICED", "compare_terms", "unpriced_terms"]

#: A rate obtained by measuring the machine (the strongest evidence).
MEASURED = "measured"
#: A rate computed from the target's declared facts (weaker: it assumes the machine matches them).
DERIVED = "derived"
#: No rate. The term is reported but cannot be weighed.
UNPRICED = "unpriced"


class Rate(NamedTuple):
    """Cycles per unit of one resource term, with the evidence that produced it.

    `provenance` is free text naming the measurement or derivation -- a run id, a counter, a
    config field. A rate without provenance is a guess wearing a number's clothes, so
    :func:`compare_terms` refuses one.
    """

    cycles_per_unit: float
    status: str
    provenance: str


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def unpriced_terms(before: Mapping[str, Any], after: Mapping[str, Any], rates: Mapping[str, Rate]) -> list[str]:
    """Terms that MOVED but have no usable rate -- the reason a composite is withheld."""
    moved: list[str] = []
    for term in sorted(set(before) | set(after)):
        lhs, rhs = before.get(term), after.get(term)
        if not (_numeric(lhs) and _numeric(rhs)) or lhs == rhs:
            continue
        rate = rates.get(term)
        if rate is None or rate.status == UNPRICED:
            moved.append(term)
    return moved


def compare_terms(before: Mapping[str, Any], after: Mapping[str, Any], rates: Mapping[str, Rate]) -> dict[str, Any]:
    """Per-term deltas, plus a composite only if every MOVED term carries a rate.

    The per-term table is always produced; it is the part that cannot mislead. The composite is
    the convenience, and it is withheld rather than approximated.
    """
    for term, rate in rates.items():
        if rate.status != UNPRICED and not str(rate.provenance).strip():
            raise ValueError(
                f"rate for {term!r} has no provenance; a rate without evidence "
                f"silently decides which optimizations are pursued"
            )

    terms: list[dict[str, Any]] = []
    for term in sorted(set(before) | set(after)):
        lhs, rhs = before.get(term), after.get(term)
        if not (_numeric(lhs) and _numeric(rhs)):
            continue
        delta = rhs - lhs
        rate = rates.get(term)
        priced = rate is not None and rate.status != UNPRICED
        terms.append(
            {
                "term": term,
                "before": lhs,
                "after": rhs,
                "delta": delta,
                "rate_status": rate.status if rate is not None else UNPRICED,
                "rate_provenance": rate.provenance if rate is not None else None,
                "cycle_delta": delta * rate.cycles_per_unit if priced else None,
            }
        )

    withheld = unpriced_terms(before, after, rates)
    priced_cycles = sum(row["cycle_delta"] for row in terms if row["cycle_delta"] is not None)
    # A term that moved without a rate makes the composite unsound: weighting it zero would hide a
    # resource the edit actually spent, which is the failure this module exists to prevent.
    composite = None if withheld else priced_cycles
    traded = (
        sorted({row["term"] for row in terms if row["delta"] > 0}) if any(row["delta"] < 0 for row in terms) else []
    )
    return {
        "schema": "resource_cost_comparison_v1",
        "terms": terms,
        "composite_cycle_delta": composite,
        "composite_status": UNPRICED if withheld else "priced",
        "unpriced_moved_terms": withheld,
        "terms_increased": traded,
        "reading": (
            "per-term deltas are always sound; the composite is withheld whenever a term moved "
            "without a measured or derived rate, because weighting an unpriced term as zero hides "
            "a resource the edit spent. `terms_increased` names what an edit TRADED AWAY when it "
            "also reduced something -- such an edit is not a regression, it is a bet, and it "
            "cannot be judged without the rates."
        ),
    }
