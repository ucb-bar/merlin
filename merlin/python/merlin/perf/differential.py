"""Rank two schedules against each other without pricing either one.

The problem this solves. A structural bound needs every resource's rate to be established, and on a
real target most are not: the walk resolves cheap combinational leaves and refuses the sequenced
units where the time actually goes. Measured on the corpus this was built against, only 2 of 21
workloads bound end to end. Read as an absolute-prediction result that is close to useless.

But choosing between two schedules does not need either total. If both leave the SAME resources
unresolved, and ask the SAME amount of work of each of those resources, then whatever those resources
cost, they cost the same in both -- and the unknown cancels out of the difference. What is left is a
difference between the parts that ARE resolved, which is exactly what the compiler controls.

That is the common case for the decisions worth making. Retiling a transfer changes movement while
leaving the compute demand alone; changing an overlap policy changes how terms compose while leaving
every demand alone. In both, the unresolved terms are untouched by the choice.

WHAT MAKES A COMPARISON SOUND, and each is checked rather than assumed:

* the same composition operator and the same overlap coefficient -- two schedules composed by
  different rules are not being measured on the same instrument;
* the same set of unresolved resources -- an unknown present in one side only cannot cancel;
* the same demand on each unresolved resource -- the *set* matching is not enough. If both leave a
  vector unit unresolved but one asks twice the work of it, the unknown contributions differ, and
  differencing the resolved parts silently attributes that gap to the wrong place.

WHAT THE OPERATOR THEN ALLOWS, which is not the same for all of them:

* ``SUM``: the total is additive in the terms, so the difference of the resolved parts IS the
  difference of the totals. Ordering and magnitude both transfer. EXACT.
* ``MAX``: not additive -- an unresolved resource that dominates both sides makes the true difference
  zero however much the resolved parts differ. But ``max`` is MONOTONE, so a smaller resolved part
  can never produce a larger total. The ordering transfers as ``<=``; the magnitude does not.
  ORDERING ONLY.
* ``PARTIAL``: the overlap credit couples the terms pairwise, so neither property survives in
  general. REFUSED rather than approximated.

A comparison that cannot be made returns a refusal naming what blocked it. It never falls back to
comparing the totals, because on this evidence one or both of them is UNKNOWN, and an UNKNOWN that
quietly becomes a number is the failure this whole layer exists to prevent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from merlin.perf.envelope import Composed, Composition

#: Operators whose totals are additive in the per-resource terms, so a resolved-part difference is
#: the true difference. Anything absent from this set gets at most an ordering.
_ADDITIVE = (Composition.SUM,)
#: Operators that are monotone non-decreasing in every term: a smaller resolved part cannot yield a
#: larger total, so the ORDER transfers even though the magnitude does not.
_MONOTONE = (Composition.SUM, Composition.MAX)

EXACT = "exact"
ORDERING_ONLY = "ordering_only"
REFUSED = "refused"


@dataclass(frozen=True)
class Comparison:
    """Which of two schedules is faster, on what basis, and what the answer does NOT include."""

    #: ``"a"``, ``"b"``, ``"tie"``, or None when the evidence cannot order them.
    faster: str | None
    #: ``b - a`` in cycles when the basis is EXACT; None otherwise. Positive means ``a`` is faster.
    delta_cycles: float | None
    #: EXACT | ORDERING_ONLY | REFUSED.
    basis: str
    reason: str
    #: The resources whose cost cancelled. Reported because the answer is only as good as this
    #: cancellation, and a reader should be able to see what was assumed equal.
    cancelled: tuple[str, ...] = ()

    @property
    def decided(self) -> bool:
        return self.faster is not None

    def claim(self) -> str:
        """The strongest sentence this comparison supports."""
        if self.basis == REFUSED:
            return f"undecidable: {self.reason}"
        if self.faster == "tie":
            return "the two schedules are indistinguishable on the resolved terms"
        if self.basis == EXACT:
            return (f"{self.faster} is faster by {abs(self.delta_cycles):.0f} cycles "
                    f"({len(self.cancelled)} unresolved resource(s) cancelled)")
        return (f"{self.faster} is no slower than the other, by a margin this evidence cannot size "
                f"({len(self.cancelled)} unresolved resource(s) cancelled)")


def _demand_amounts(demands: "Mapping[str, Any] | None") -> dict[str, float]:
    """Resource -> amount, tolerating either a demand object or a bare number."""
    out: dict[str, float] = {}
    for name, d in (demands or {}).items():
        out[name] = float(getattr(d, "amount", d))
    return out


def comparable(a: Composed, b: Composed, *,
               demands_a: "Mapping[str, Any] | None" = None,
               demands_b: "Mapping[str, Any] | None" = None) -> tuple[bool, str]:
    """Whether these two composed bounds may be differenced at all, and why not when they may not.

    Separated from :func:`compare` so a caller can ask the question without asking for an answer --
    a search that must skip incomparable pairs should not have to read a refusal to learn that."""
    if a.operator is not b.operator:
        return False, (f"composed by different operators ({a.operator.name} vs {b.operator.name}); "
                       "they are not the same instrument")
    if a.eta != b.eta:
        return False, f"different overlap coefficients ({a.eta} vs {b.eta})"
    if set(a.unresolved) != set(b.unresolved):
        only_a = sorted(set(a.unresolved) - set(b.unresolved))
        only_b = sorted(set(b.unresolved) - set(a.unresolved))
        return False, (f"the unresolved sets differ (only in a: {only_a}; only in b: {only_b}); "
                       "an unknown present on one side cannot cancel")
    if a.operator is Composition.PARTIAL:
        return False, ("a partial-overlap operator credits pairs, so neither the magnitude nor the "
                       "ordering of a resolved-part difference survives; refusing rather than "
                       "approximating")
    da, db = _demand_amounts(demands_a), _demand_amounts(demands_b)
    if da or db:
        for name in sorted(set(a.unresolved)):
            if name not in da or name not in db:
                return False, (f"the demand on unresolved resource {name!r} is not stated for both "
                               "sides, so it cannot be shown to cancel")
            if da[name] != db[name]:
                return False, (f"unresolved resource {name!r} is asked for different work "
                               f"({da[name]} vs {db[name]}), so its unknown cost does not cancel")
    elif a.unresolved:
        return False, ("demands were not supplied, so the unresolved resources cannot be shown to "
                       "carry equal work; a matching unresolved SET is not sufficient")
    return True, "same operator, same unresolved resources, equal work on each"


def compare(a: Composed, b: Composed, *,
            demands_a: "Mapping[str, Any] | None" = None,
            demands_b: "Mapping[str, Any] | None" = None,
            label_a: str = "a", label_b: str = "b") -> Comparison:
    """Order two schedules by their resolved parts, cancelling the unknowns they share."""
    ok, why = comparable(a, b, demands_a=demands_a, demands_b=demands_b)
    if not ok:
        return Comparison(faster=None, delta_cycles=None, basis=REFUSED, reason=why)

    delta = b.partial_cycles - a.partial_cycles          # positive -> a is faster
    if delta == 0:
        return Comparison(faster="tie", delta_cycles=0.0,
                          basis=EXACT if a.operator in _ADDITIVE else ORDERING_ONLY,
                          reason="the resolved parts are equal", cancelled=tuple(a.unresolved))
    winner = label_a if delta > 0 else label_b

    if a.operator in _ADDITIVE:
        return Comparison(faster=winner, delta_cycles=delta, basis=EXACT,
                          reason=("the operator is additive, so the difference of the resolved parts "
                                  "is the difference of the totals"),
                          cancelled=tuple(a.unresolved))
    if a.operator in _MONOTONE:
        return Comparison(faster=winner, delta_cycles=None, basis=ORDERING_ONLY,
                          reason=("the operator is monotone but not additive: a smaller resolved part "
                                  "cannot produce a larger total, but an unresolved resource may "
                                  "dominate both and shrink the true gap to nothing"),
                          cancelled=tuple(a.unresolved))
    return Comparison(faster=None, delta_cycles=None, basis=REFUSED,
                      reason=f"operator {a.operator.name} is neither additive nor monotone here")


def rank_schedules(candidates: "Mapping[str, Composed]", *,
                   demands: "Mapping[str, Mapping[str, Any]] | None" = None
                   ) -> tuple[list[str], list[Comparison]]:
    """Order schedules best-first by pairwise comparison, and report every pair that refused.

    Ordering is by resolved part among the mutually comparable, which is sound exactly when the
    pairwise checks pass. Schedules that cannot be compared to the leader are NOT dropped: they are
    returned in the refusals, because a candidate excluded for want of evidence is a hole in the
    search, not an answer about the candidate."""
    names = sorted(candidates)
    refusals: list[Comparison] = []
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            c = compare(candidates[x], candidates[y],
                        demands_a=(demands or {}).get(x), demands_b=(demands or {}).get(y),
                        label_a=x, label_b=y)
            if c.basis == REFUSED:
                refusals.append(c)
    order = sorted(names, key=lambda n: candidates[n].partial_cycles)
    return order, refusals
