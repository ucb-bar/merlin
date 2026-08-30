"""Confront the dependence graph with a per-cycle trace: which separations were real.

The graph prices the separations a schedule MUST leave between instructions. Until it is checked
against a machine it is only plausible -- every weight on it came from a published latency or from
an UNKNOWN read as zero, and neither is a measurement. A per-cycle trace carrying the program
counter settles it, because the cycle each instruction issued is then observable.

WHAT A MEASURED SEPARATION IS, AND WHAT IT IS NOT
--------------------------------------------------
For an edge ``u -> v`` with a required separation ``R`` (unknown), a run that executed CORRECTLY left
``S = issue(v) - issue(u)`` cycles between them. Correctness gives ``S >= R``, so:

* ``S`` is an **upper bound on the requirement**, never the requirement itself. A schedule that
  over-delays -- and every schedule in this corpus does -- leaves ``S`` far above ``R``. So a
  measured separation may narrow an UNKNOWN from above and may never be promoted to "the latency"
  (:mod:`merlin.perf.harvest`'s rule for contended observations, applied to edges).
* A predicted weight ``W > S`` is **FALSIFIED**: the graph claims a separation larger than one a
  correct execution actually used. That is the one direction in which a single run can refute the
  model outright, and it is why this check is worth running.

An instruction that issued more than once yields one observation per issue; the MINIMUM is the
tightest separation the machine was ever seen to accept and is the informative one. An instruction
that never issued is excluded and reported -- an unexecuted instruction has no separation, which is
not the same as a separation of zero.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from merlin.perf.decompose import UNKNOWN

__all__ = [
    "ClassPricing", "SeparationObservation", "TimeAttribution", "issue_times",
    "measured_separations", "price_unknown_classes", "time_attribution",
]


def issue_times(pc_by_cycle: "Sequence[int]") -> dict[int, tuple[int, ...]]:
    """``{instruction index: cycles it issued}`` from the per-cycle program counter.

    An instruction issues on the cycle the counter ARRIVES at it; cycles where the counter does not
    move are the machine stalled on the instruction already there, not a second issue of it. On a
    target whose counter is a word index this maps straight onto the program's instruction indices;
    on one where it is a byte address the caller scales it before calling, since the scale is a
    target fact and not something to guess here.
    """
    out: dict[int, list[int]] = {}
    prev = None
    for cycle, pc in enumerate(pc_by_cycle):
        if pc != prev:
            out.setdefault(int(pc), []).append(cycle)
        prev = pc
    return {k: tuple(v) for k, v in sorted(out.items())}


@dataclass(frozen=True)
class SeparationObservation:
    """One edge, what the graph predicted, and what the machine actually left."""

    src: int
    dst: int
    kind: str
    edge_class: str
    #: The graph's weight, or UNKNOWN where it does not price this edge.
    predicted: "float | object"
    #: ``issue(dst) - issue(src)`` -- an UPPER bound on the required separation.
    measured: int
    #: Every observed separation, when either endpoint issued more than once.
    all_measured: tuple[int, ...] = ()

    @property
    def falsified(self) -> bool:
        """The graph demanded more separation than a correct execution actually used."""
        return self.predicted is not UNKNOWN and float(self.predicted) > self.measured

    @property
    def slack(self) -> "float | None":
        """How much more separation the schedule left than the graph required."""
        if self.predicted is UNKNOWN:
            return None
        return self.measured - float(self.predicted)


def measured_separations(dag, issues: Mapping[int, "tuple[int, ...]"]
                         ) -> tuple[list[SeparationObservation], list[str]]:
    """``(observations, skipped)`` -- every edge whose endpoints both issued, confronted.

    ``skipped`` names the edges that could not be checked because an endpoint never executed. They
    are returned rather than dropped: an edge nobody exercised is a hole in the validation, not a
    passing edge.
    """
    obs: list[SeparationObservation] = []
    skipped: list[str] = []
    for e in dag.edges:
        a, b = issues.get(e.src), issues.get(e.dst)
        if not a or not b:
            which = "src" if not a else "dst"
            skipped.append(f"{e.src}->{e.dst} ({e.kind}): {which} never issued")
            continue
        seps = tuple(sorted(y - x for x in a for y in b if y > x))
        if not seps:
            skipped.append(f"{e.src}->{e.dst} ({e.kind}): dst never issued after src")
            continue
        obs.append(SeparationObservation(
            src=e.src, dst=e.dst, kind=e.kind, edge_class=e.edge_class,
            predicted=(e.cycles if e.known else UNKNOWN),
            measured=seps[0], all_measured=seps))
    return obs, skipped


@dataclass(frozen=True)
class ClassPricing:
    """What a trace says about an edge class the graph could not price."""

    edge_class: str
    n_edges: int
    #: The tightest separation observed for this class -- the best UPPER bound on its requirement.
    tightest: int
    loosest: int
    #: Stated on every pricing, because it is the whole caveat: this narrows the unknown from above.
    basis: str = ("trace_derived: a correct run left at least the required separation, so this is an "
                  "UPPER bound on the requirement and may not be promoted to the latency")


def price_unknown_classes(observations: "Sequence[SeparationObservation]") -> dict[str, ClassPricing]:
    """Narrow each UNPRICED edge class from above, using what the machine actually left.

    This is the only route from an UNKNOWN separation to a number that is not a guess -- and it
    yields a bound rather than a value, which is why the class keeps its UNKNOWN status and gains a
    ceiling instead of losing it.
    """
    by_class: dict[str, list[int]] = {}
    for o in observations:
        if o.predicted is UNKNOWN:
            by_class.setdefault(o.edge_class or "unclassified", []).append(o.measured)
    return {k: ClassPricing(edge_class=k, n_edges=len(v), tightest=min(v), loosest=max(v))
            for k, v in sorted(by_class.items())}


@dataclass(frozen=True)
class TimeAttribution:
    """Where a run's cycles went, per instruction, from the program counter alone."""

    total_cycles: int
    #: instruction index -> cycles the counter sat on it. Sums to the run.
    by_instruction: dict[int, int]
    #: mnemonic -> cycles, when the caller supplies the program's instructions.
    by_mnemonic: dict[str, int]
    #: The few instructions that hold most of the time, largest first.
    top: tuple[tuple[int, int], ...]

    def concentration(self, n: int = 8) -> float:
        """Fraction of the run held by the ``n`` costliest instructions."""
        return sum(c for _i, c in self.top[:n]) / self.total_cycles if self.total_cycles else 0.0


def time_attribution(pc_by_cycle: "Sequence[int]",
                     instructions: "Sequence | None" = None) -> TimeAttribution:
    """Attribute every cycle of a run to the instruction the counter was sitting on.

    NOT a longest path. A chain of MEASURED separations is a chain of elapsed times, so its longest
    path degenerates to the span of the program -- on a straight-line kernel it recovers the makespan
    from a single early-to-late edge and explains nothing. Elapsed time is not a dependence.

    Counting where the counter waited *is* informative and needs no model: an in-order machine that
    cannot advance is being held by whatever sits at that address, so this says which instructions
    the run was actually spent on, and a schedule change can be aimed at them.
    """
    by_i: dict[int, int] = {}
    for pc in pc_by_cycle:
        by_i[int(pc)] = by_i.get(int(pc), 0) + 1
    by_m: dict[str, int] = {}
    if instructions is not None:
        for i, c in by_i.items():
            if i < len(instructions):
                m = getattr(instructions[i], "mnemonic", None) or "?"
                by_m[m] = by_m.get(m, 0) + c
    top = tuple(sorted(by_i.items(), key=lambda kv: -kv[1]))
    return TimeAttribution(total_cycles=len(pc_by_cycle), by_instruction=by_i,
                           by_mnemonic=by_m, top=top)
