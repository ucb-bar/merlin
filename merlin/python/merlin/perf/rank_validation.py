"""Measure whether a cheap signal actually orders schedules the way the oracle does.

A screening signal is only worth exposing if it agrees with the timing oracle on the exact
comparison the search makes: two candidate programs for the SAME workload, which is faster. That
agreement is a measurement, not a property anyone may assume, and it has already caught two signals
that looked reasonable and were not:

* the correctness simulator's cycle counts agree 46.1% of the time -- a coin flip;
* a per-command cost model, accurate to 8.1% on absolute magnitude, agrees **39.3%** -- worse than a
  coin, because within one workload the term tracking the work never varies between candidates and
  the terms that do vary anti-correlate with measured time.

A signal below chance is not a weak signal. It is a signal pointing the wrong way, and an agent told
to use it will follow it. So the default is refusal: a scorer is reported, never trusted, and this
module's job is to produce the number that decides.

Design notes that matter for honesty:

* **Ties are not agreements.** A scorer that cannot separate two programs is counted as undecided,
  never as correct. Reporting ties as agreement is how a constant function scores 100%.
* **Decided count is reported with every rate.** 80% of 5 pairs is not evidence; the caller sets a
  minimum and this refuses below it rather than printing a headline.
* **Held-out slices are computed, not assumed.** A scorer fitted on one workload family that scores
  well only there has learned that family, and leave-one-out is what shows it.

Nothing here knows any target, opcode, or unit: a record is (workload, program, measured, score).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: Chance. A scorer at or below this decides nothing, whatever its decided count.
CHANCE = 0.5


@dataclass(frozen=True)
class Program:
    """One measured program: which workload it belongs to, and what the oracle said."""

    workload: str
    program: str
    measured: float
    group: str = ""          # optional coarser slice, e.g. the family the workload belongs to


@dataclass(frozen=True)
class Agreement:
    """What a scorer achieved, always with the count it achieved it on."""

    pairs: int
    decided: int
    agreed: int
    undecided: int

    @property
    def rate(self) -> float | None:
        """Agreement over DECIDED pairs, or None when nothing was decided."""
        return (self.agreed / self.decided) if self.decided else None

    def to_dict(self) -> dict[str, Any]:
        return {"pairs": self.pairs, "decided": self.decided, "agreed": self.agreed,
                "undecided": self.undecided, "rate": self.rate}


def ordered_pairs(programs: Sequence[Program]) -> list[tuple[Program, Program]]:
    """Every within-workload pair whose measured times differ.

    Within-workload is the point: comparing two programs for DIFFERENT work asks a question the
    search never asks, and a scorer can look good on it by tracking problem size. Pairs where the
    oracle itself reports a tie are excluded -- there is no correct answer to agree with.
    """
    by_workload: dict[str, list[Program]] = {}
    for p in programs:
        by_workload.setdefault(p.workload, []).append(p)
    out: list[tuple[Program, Program]] = []
    for rows in by_workload.values():
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if rows[i].measured != rows[j].measured:
                    out.append((rows[i], rows[j]))
    return out


def agreement(pairs: Sequence[tuple[Program, Program]],
              score: Mapping[str, float], *, margin: float = 0.0) -> Agreement:
    """How often the scorer's ordering matches the oracle's, over pairs it decides.

    ``margin`` is the separation the scorer must show before it is taken to have an opinion. Raising
    it trades decided count for accuracy, and both halves of that trade are reported so the caller
    can see what a higher gate cost.
    """
    agreed = decided = undecided = 0
    for a, b in pairs:
        sa, sb = score.get(a.program), score.get(b.program)
        if sa is None or sb is None or abs(sa - sb) <= margin:
            undecided += 1
            continue
        decided += 1
        # both orderings are "smaller is better": fewer predicted units, fewer measured cycles
        if (sa < sb) == (a.measured < b.measured):
            agreed += 1
    return Agreement(pairs=len(pairs), decided=decided, agreed=agreed, undecided=undecided)


def held_out(programs: Sequence[Program], score: Mapping[str, float], *,
             by: str = "workload", margin: float = 0.0) -> dict[str, Agreement]:
    """Agreement computed separately per slice, to expose a scorer that learned one slice.

    ``by`` selects the slice key: ``"workload"`` or ``"group"``. A scorer that scores well overall
    and badly on every slice but one has not found a general property of schedules.
    """
    if by not in ("workload", "group"):
        raise ValueError("slice key must be 'workload' or 'group'")
    slices: dict[str, list[Program]] = {}
    for p in programs:
        slices.setdefault(getattr(p, by), []).append(p)
    return {name: agreement(ordered_pairs(rows), score, margin=margin)
            for name, rows in sorted(slices.items())}


def verdict(overall: Agreement, slices: Mapping[str, Agreement], *,
            minimum_rate: float, minimum_decided: int,
            minimum_slice_decided: int, minimum_slices: int = 2) -> dict[str, Any]:
    """Is this scorer fit to be shown to a search? Refuses by default, and says why.

    Three ways to fail, each reported rather than collapsed into a boolean: too little evidence,
    agreement no better than chance, or agreement that does not survive being sliced. A scorer below
    chance is called out separately because it is worse than having no signal at all.
    """
    reasons: list[str] = []
    if overall.decided < minimum_decided:
        reasons.append(f"decided only {overall.decided} pair(s), below the required "
                       f"{minimum_decided}; the rate is not evidence at this count")
    rate = overall.rate
    if rate is None:
        reasons.append("the scorer decided nothing; it separates no pair of programs")
    else:
        if rate <= CHANCE:
            reasons.append(f"agreement {rate:.3f} is at or below chance ({CHANCE}); a signal that "
                           f"does not beat a coin will be followed and must not be shown")
        elif rate < minimum_rate:
            reasons.append(f"agreement {rate:.3f} is below the required {minimum_rate}")
    # EVIDENCE FROM ONE SLICE IS NOT EVIDENCE THAT GENERALISES. A scorer whose every decided pair
    # comes from a single workload family has been shown to work there and nowhere else; counting
    # the silent slices as passes is the "the check could not run, so it passed" failure. Measured
    # case: a tile-pressure heuristic scored 0.804 overall with 158 decided pairs, ALL of them from
    # one family, while a workload inside that same family scored 0.486 -- below chance.
    qualifying = {name: a for name, a in slices.items() if a.decided >= minimum_slice_decided}
    if len(qualifying) < minimum_slices:
        reasons.append(f"only {len(qualifying)} slice(s) carry at least {minimum_slice_decided} "
                       f"decided pair(s), below the required {minimum_slices}; a rate measured on "
                       f"one slice says nothing about the others, which decided too little to check")
    weak = {name: a.rate for name, a in qualifying.items()
            if a.rate is None or a.rate < minimum_rate}
    if weak:
        reasons.append(f"{len(weak)} slice(s) with enough evidence fall below the bar: "
                       + ", ".join(f"{n}={r:.3f}" if r is not None else f"{n}=undecided"
                                   for n, r in sorted(weak.items())))
    return {"exposable": not reasons, "reasons": reasons, "overall": overall.to_dict(),
            "slices": {n: a.to_dict() for n, a in slices.items()},
            "qualifying_slices": sorted(qualifying),
            "thresholds": {"minimum_rate": minimum_rate, "minimum_decided": minimum_decided,
                           "minimum_slice_decided": minimum_slice_decided,
                           "minimum_slices": minimum_slices, "chance": CHANCE}}
