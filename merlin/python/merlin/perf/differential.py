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


INCOMPARABLE = "incomparable"


@dataclass(frozen=True)
class VectorComparison:
    """Two schedules compared engine by engine, and what the set of answers jointly supports.

    A single number cannot order two schedules on a device with more than one engine, because
    "faster" is no longer a total order: a schedule can win on the systolic array and lose on the
    SIMT lanes, and which one is better then depends on a trade the evidence does not contain. The
    honest object is a VECTOR of per-engine comparisons plus a DOMINANCE verdict over them -- b beats
    a only when it is no worse on every engine and strictly better on at least one.
    """

    #: engine -> the comparison on that engine alone.
    per_engine: dict[str, Comparison]
    #: ``label_a`` / ``label_b`` / ``"tie"``, or None when the engines disagree or an engine refused.
    faster: str | None
    #: EXACT | ORDERING_ONLY | INCOMPARABLE | REFUSED.
    basis: str
    reason: str
    #: The summed difference. Present ONLY when every engine compared EXACT *and* the engines are
    #: known to compose additively. Adding per-engine deltas on a device whose engines overlap
    #: double-counts the overlapped cycles, so this stays None rather than becoming a plausible
    #: number -- which is the same rule the scalar comparator applies one level down.
    total_delta_cycles: float | None = None
    #: Engines whose comparison refused. A dominance claim over a set containing one of these would
    #: be asserting something about an engine nobody measured.
    undecided_engines: tuple[str, ...] = ()
    #: ``(engine, winner)`` for engines that disagree about the winner -- the actual trade-off.
    traded: tuple[tuple[str, str], ...] = ()

    @property
    def decided(self) -> bool:
        return self.faster is not None

    def claim(self) -> str:
        """The strongest sentence this vector supports."""
        if self.basis == REFUSED:
            return f"undecidable: {self.reason}"
        if self.basis == INCOMPARABLE:
            traded = ", ".join(f"{e} favours {w}" for e, w in self.traded)
            return f"incomparable: neither dominates ({traded})"
        if self.faster == "tie":
            return "the two schedules are equal on every engine"
        if self.total_delta_cycles is not None:
            return (f"{self.faster} dominates on all {len(self.per_engine)} engine(s), "
                    f"by {abs(self.total_delta_cycles):.0f} cycles summed")
        return (f"{self.faster} dominates: no worse on every engine and strictly better on at least "
                f"one, by a margin this evidence cannot sum")


def compare_by_engine(a: "Mapping[str, Composed]", b: "Mapping[str, Composed]", *,
                      demands_a: "Mapping[str, Mapping[str, Any]] | None" = None,
                      demands_b: "Mapping[str, Mapping[str, Any]] | None" = None,
                      label_a: str = "a", label_b: str = "b",
                      engines_compose: "Composition | None" = None) -> VectorComparison:
    """Compare two schedules engine by engine and report which, if either, DOMINATES.

    ``a`` and ``b`` map engine name -> that engine's composed bound. ``engines_compose`` is how the
    device's engines combine with one another -- derived from a joint occupancy vector, never
    assumed. It gates the summed delta only; the dominance verdict never needs it, which is why
    dominance is available on targets where no total is.

    **An engine named on one side only is a refusal, not a zero.** Two schedules that use different
    engine sets are doing different work, and scoring the absent engine as zero cycles is exactly how
    moving work off an accelerator reads as making the accelerator faster.
    """
    ea, eb = set(a), set(b)
    if ea != eb:
        only_a, only_b = sorted(ea - eb), sorted(eb - ea)
        return VectorComparison(
            per_engine={}, faster=None, basis=REFUSED,
            reason=(f"the schedules name different engines (only in {label_a}: {only_a}; only in "
                    f"{label_b}: {only_b}); an engine absent on one side is unmeasured there, not "
                    "zero, and treating it as zero reports moving work off an engine as speeding it up"))

    per: dict[str, Comparison] = {}
    for e in sorted(ea):
        per[e] = compare(a[e], b[e],
                         demands_a=(demands_a or {}).get(e), demands_b=(demands_b or {}).get(e),
                         label_a=label_a, label_b=label_b)

    undecided = tuple(e for e, c in per.items() if c.basis == REFUSED)
    winners = {e: c.faster for e, c in per.items() if c.basis != REFUSED}
    non_tie = {e: w for e, w in winners.items() if w != "tie"}
    distinct = set(non_tie.values())

    if len(distinct) > 1:
        # A real trade-off. Report it EVEN IF an engine refused: knowing the two disagree is a
        # stronger statement than "undecidable", and it is the answer a scheduler has to act on.
        return VectorComparison(
            per_engine=per, faster=None, basis=INCOMPARABLE,
            reason="neither schedule is at least as good on every engine",
            undecided_engines=undecided,
            traded=tuple(sorted(non_tie.items())))

    if undecided:
        return VectorComparison(
            per_engine=per, faster=None, basis=REFUSED,
            reason=(f"engine(s) {list(undecided)} could not be compared, so no claim covers every "
                    "engine; the engines that did compare do not disagree"),
            undecided_engines=undecided)

    if not distinct:
        return VectorComparison(per_engine=per, faster="tie", basis=EXACT,
                                reason="every engine is a tie", total_delta_cycles=0.0)

    winner = distinct.pop()
    all_exact = all(c.basis == EXACT for c in per.values())
    if all_exact and engines_compose in _ADDITIVE:
        total = sum(c.delta_cycles or 0.0 for c in per.values())
        return VectorComparison(
            per_engine=per, faster=winner, basis=EXACT,
            reason=("every engine compared exactly and the engines compose additively, so the "
                    "per-engine differences sum"),
            total_delta_cycles=total)
    why = ("not every engine compared exactly" if not all_exact else
           f"the engines compose by {engines_compose.name if engines_compose else 'an undeclared operator'}, "
           "so per-engine differences may not be summed")
    return VectorComparison(
        per_engine=per, faster=winner, basis=ORDERING_ONLY,
        reason=f"{winner} is no worse on every engine and better on at least one, but {why}")
