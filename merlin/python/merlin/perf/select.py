"""Bounded candidate selection: which change to try next, and when to stop trying.

Two axes, no more. The reference corpus this layer was built against is 21/21 single-op, single-tile
kernels, which cannot support a five-level tiling hierarchy or a multi-stage pipeline policy -- a
deeper space would be a space whose points the evidence cannot tell apart. So the levers are exactly

* :attr:`Axis.DMA_TILING` -- how many movement commands carry the payload, i.e. the descriptor shape;
* :attr:`Axis.OVERLAP` -- which pair of resource groups runs concurrently instead of taking turns.

**Both axes can be UNKNOWN, and UNKNOWN is not a default candidate.** The movement axis needs a
per-command byte volume, which is derivable only once the movement commands carry enough role
information to size them; where that is missing, :func:`dma_axis` returns an
:class:`~merlin.perf.decompose.Unavailable`-backed :class:`AxisEvidence` naming what is absent rather
than a candidate list built on a guessed granule. Manufacturing a plausible descriptor sweep out of
nothing is how a selection loop spends a real budget exploring a fictional space.

**What this is composed from, and what it deliberately is not.** There is no new search method here.
Enumeration inside each axis is :func:`merlin.dse.search.grid.grid_search` -- an explicit sweep over a
small space, scored at every point, which is the sanctioned method whose semantics actually match the
problem. Budget accounting is :mod:`merlin.perf.budget` over
:mod:`merlin.targetgen.tier_policy`'s ledger. Re-evaluation is suppressed by content-addressing a
candidate the way :mod:`merlin.targetgen.oracle_schedule` content-addresses a verdict: a point whose
digest has already been evaluated is served from the cache and charged nothing. No beam, no MCTS, no
Bayesian optimizer, no surrogate model: the repo's search layer rules those out, and the measurement
that motivated this module says the evaluation is cheap enough that near-exhaustive scoring of a
small derived space is affordable. Rationing the cheap thing was the original design error; adding a
clever optimizer to ration it better would be the same error with more code.

**VOI has three factors, not four.** ``Impact x Uncertainty / Cost``. *Generality* -- how many targets
a finding transfers to -- was dropped: with one target in the comparison it takes the same value for
every candidate, and a constant factor cannot change a ranking. It is a term to restore only when a
second target is actually in the loop.

*Uncertainty* is the width of the prediction interval the evidence supports, relative to its top:
``(hi - lo) / hi``. A candidate whose saving is already pinned down has uncertainty 0 and therefore
VOI 0 -- correctly, because VOI ranks what is worth *querying*, not what is worth *implementing*. A
known-good change should be applied, not re-measured.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from merlin.dse.search.grid import grid_search
from merlin.targetgen.oracle_schedule import PASS, CapsuleState, Verdict
from merlin.targetgen.oracle_schedule import UNKNOWN as VERDICT_UNKNOWN

from .amplification import WorkloadAmplification
from .budget import Budget
from .decompose import UNKNOWN, ActivitySource, ResourceKind, Unavailable, _Unknown, is_unknown
from .headroom import WorkloadHeadroom

#: The one "tier" a candidate is evaluated at in this loop. Named so a candidate's evaluation can be
#: content-addressed with :mod:`merlin.targetgen.oracle_schedule`'s own verdict primitive.
EVALUATED = "evaluated"

#: Dropped from the VOI product on purpose; see the module docstring. Kept as a named constant so a
#: reader who goes looking for it finds the decision instead of an omission.
GENERALITY_DROPPED = ("Generality is constant across candidates while one target is in the "
                      "comparison, so it cannot change a ranking. Restore it when a second target "
                      "enters the loop.")


class Axis(str, Enum):
    """The levers the corpus can distinguish. Two, and adding a third needs corpus evidence first."""

    #: How many movement commands carry the payload (descriptor shape / DMA tiling).
    DMA_TILING = "dma_tiling"
    #: Which pair of resource groups runs concurrently rather than taking turns.
    OVERLAP = "overlap_policy"


@dataclass(frozen=True)
class Candidate:
    """One point on one axis, with the saving INTERVAL the evidence supports.

    ``saving_lo`` and ``saving_hi`` are separate because the evidence almost never pins a saving to a
    point. Collapsing them to a single number would erase the only thing that decides whether a query
    is worth making -- and would let an unobserved overlap ceiling read as a promise.
    """

    axis: Axis
    workload: str
    #: The axis point itself, as sorted ``(key, value)`` pairs so the digest is stable.
    setting: tuple[tuple[str, object], ...]
    baseline_cycles: int
    #: Cycles the change could save at best, given the evidence.
    saving_hi: "float | _Unknown"
    #: Cycles the change is already backed to save. UNKNOWN when the evidence does not split the
    #: interval -- never 0, which would read as "certainly worthless".
    saving_lo: "float | _Unknown"
    rationale: str
    #: True while ``saving_hi`` is a ceiling nobody has observed the realised part of.
    is_upper_bound: bool = True

    @property
    def digest(self) -> str:
        body = json.dumps({"axis": self.axis.value, "workload": self.workload,
                           "setting": [[k, v] for k, v in self.setting]},
                          sort_keys=True, default=str)
        return hashlib.sha1(body.encode("utf-8")).hexdigest()[:12]

    @property
    def id(self) -> str:
        pairs = ",".join(f"{k}={v}" for k, v in self.setting)
        return f"{self.workload}:{self.axis.value}[{pairs}]"

    @property
    def predicted_cycles(self) -> "float | _Unknown":
        """Best-case cycles after the change. UNKNOWN propagates from the saving."""
        if is_unknown(self.saving_hi):
            return UNKNOWN
        return max(0.0, self.baseline_cycles - float(self.saving_hi))

    def to_dict(self) -> dict:
        def _s(v: object) -> object:
            return "UNKNOWN" if is_unknown(v) else v

        return {"id": self.id, "digest": self.digest, "axis": self.axis.value,
                "workload": self.workload, "setting": dict(self.setting),
                "baseline_cycles": self.baseline_cycles, "saving_lo": _s(self.saving_lo),
                "saving_hi": _s(self.saving_hi), "predicted_cycles": _s(self.predicted_cycles),
                "is_upper_bound": self.is_upper_bound, "rationale": self.rationale}


@dataclass(frozen=True)
class AxisEvidence:
    """Whether one axis is usable on one workload, and if so what it offers.

    ``established`` is tri-state. ``None`` means the axis could not be evaluated -- the evidence it
    needs is missing -- which is a different claim from ``False`` ("this axis has nothing to offer
    here"), and only the second is a finding about the target.
    """

    axis: Axis
    workload: str
    established: bool | None
    candidates: tuple[Candidate, ...] = ()
    missing: tuple[str, ...] = ()
    rationale: str = ""

    @property
    def unavailable(self) -> "Unavailable | None":
        if self.established is not None:
            return None
        return Unavailable(f"the {self.axis.value} axis on {self.workload!r}", self.missing,
                           self.rationale)

    def to_dict(self) -> dict:
        return {"axis": self.axis.value, "workload": self.workload, "established": self.established,
                "candidates": [c.to_dict() for c in self.candidates],
                "missing": list(self.missing), "rationale": self.rationale}


def _movement_busy(source: ActivitySource) -> int | None:
    cycles = [r.busy_cycles for r in source.resources if r.kind is ResourceKind.MOVEMENT]
    return sum(cycles) if cycles else None


def _transfer_ladder(observed: int, floor: int) -> list[int]:
    """The command counts worth sweeping between what the program issues and the fewest that could
    carry the payload: repeated halving, plus both endpoints.

    Halving rather than every integer because the sweep must stay small enough to score exhaustively
    and because the quantity that matters -- bytes per command -- moves multiplicatively. Derived
    from the observation at both ends; no step size is baked in.
    """
    if observed <= floor:
        return [observed]
    out = {observed, floor}
    t = observed
    while t > floor:
        t = max(floor, t // 2)
        out.add(t)
    return sorted(out, reverse=True)


def dma_axis(source: ActivitySource,
             amp: "WorkloadAmplification | Unavailable | None") -> AxisEvidence:
    """Candidate descriptor shapes for one workload, or a named refusal.

    Needs three things and says which is missing: a movement resource in the activity source, an
    amplification result, and a DERIVED per-command byte volume (``block_bytes``). The last is the
    one that is commonly absent -- sizing a command requires the command stream to carry enough role
    information to say how many bytes it moves -- and without it there is no sweep, because every
    point of the sweep is "issue N commands of B bytes" and B is the unknown.

    The saving interval comes from the amplification split. The top of the interval assumes movement
    time falls with moved bytes; the bottom keeps only the share the fixed-granule artifact accounts
    for (``artifact_share``), which is the part that amortizes as commands get fuller regardless of
    whether any refetch is actually eliminated.
    """
    move_busy = _movement_busy(source)
    if move_busy is None or move_busy <= 0:
        return AxisEvidence(Axis.DMA_TILING, source.workload, None,
                            missing=("a movement resource with non-zero occupancy",),
                            rationale=(f"{source.workload}: the activity source charges no cycles to "
                                       f"a resource of kind {ResourceKind.MOVEMENT.value!r}, so a "
                                       f"descriptor change has nothing to act on"))
    if amp is None or isinstance(amp, Unavailable):
        missing = amp.missing if isinstance(amp, Unavailable) else ("a data-movement amplification "
                                                                    "observation",)
        detail = amp.detail if isinstance(amp, Unavailable) else "no amplification supplied"
        return AxisEvidence(Axis.DMA_TILING, source.workload, None, missing=tuple(missing),
                            rationale=detail)
    if is_unknown(amp.block_bytes) or is_unknown(amp.transfers_min):
        return AxisEvidence(
            Axis.DMA_TILING, source.workload, None,
            missing=("the per-command byte volume (>=2 observed movement commands, or per-command "
                     "byte counts)",),
            rationale=(f"{source.workload}: moved {amp.moved_bytes} bytes for {amp.useful_bytes} "
                       f"useful, but the byte volume of one command is not derivable, so a "
                       f"descriptor sweep would be a sweep over an assumed granule"))

    block = float(amp.block_bytes)
    floor = int(amp.transfers_min)
    # The command count comes back out of the amplification split rather than from moved/block:
    # with heterogeneous descriptors the block is the LARGEST command, so dividing the total by it
    # undercounts the commands actually issued.
    observed = (int(round(float(amp.redundancy_factor) * floor))
                if not is_unknown(amp.redundancy_factor) else floor)
    share = amp.artifact_share
    ladder = _transfer_ladder(max(observed, floor), floor)

    def _evaluate(point: Mapping[str, object]) -> float:
        t = int(point["transfers"])
        moved = max(float(amp.useful_bytes), t * block)
        return move_busy * (1.0 - moved / amp.moved_bytes) if amp.moved_bytes else 0.0

    rows = grid_search({"transfers": [t for t in ladder if t != observed]}, _evaluate)

    cands: list[Candidate] = []
    for row in rows:
        hi = float(row["score"])
        if hi <= 0:
            continue
        lo: float | _Unknown = UNKNOWN if is_unknown(share) else max(0.0, hi * float(share))
        cands.append(Candidate(
            axis=Axis.DMA_TILING, workload=source.workload,
            setting=(("transfers", int(row["transfers"])), ("block_bytes", block)),
            baseline_cycles=source.total_cycles, saving_hi=hi, saving_lo=lo,
            is_upper_bound=True,
            rationale=(f"{observed} command(s) of {block:.0f} B move {amp.moved_bytes} B for "
                       f"{amp.useful_bytes} useful (x{amp.ratio:.3g}); issuing "
                       f"{int(row['transfers'])} would move "
                       f"{max(float(amp.useful_bytes), int(row['transfers']) * block):.0f} B, and "
                       f"movement holds {move_busy} of {source.total_cycles} cycles")))
    if not cands:
        return AxisEvidence(Axis.DMA_TILING, source.workload, False,
                            rationale=(f"{source.workload}: {observed} command(s) already at or "
                                       f"below the {floor}-command floor for {amp.useful_bytes} "
                                       f"useful bytes at a {block:.0f} B granule; no descriptor "
                                       f"shape moves fewer bytes"))
    cands.sort(key=lambda c: (-float(c.saving_hi), c.id))
    return AxisEvidence(Axis.DMA_TILING, source.workload, True, tuple(cands),
                        rationale=(f"swept {len(cands)} descriptor shape(s) between the observed "
                                   f"{observed} and the {floor}-command floor"))


def overlap_axis(source: ActivitySource,
                 hr: "WorkloadHeadroom | Unavailable | None") -> AxisEvidence:
    """Candidate overlap policies for one workload, one per concurrency-capable pair.

    The saving interval is where the honesty lives. With the realised overlap unobserved
    (``is_upper_bound``) the bottom of the interval is 0 -- the program may already overlap the pair
    completely, in which case the change is worth nothing -- and the top is ``min(T_a, T_b)``. Once
    the realised overlap IS observed the interval collapses and the candidate's VOI goes to zero:
    there is nothing left to learn by querying it, only something left to do.
    """
    if hr is None or isinstance(hr, Unavailable):
        missing = hr.missing if isinstance(hr, Unavailable) else ("a concurrency headroom result",)
        detail = hr.detail if isinstance(hr, Unavailable) else "no headroom supplied"
        return AxisEvidence(Axis.OVERLAP, source.workload, None, missing=tuple(missing),
                            rationale=detail)

    by_pair = {(p.a, p.b): p for p in hr.pairs}

    def _evaluate(point: Mapping[str, object]) -> float:
        return float(by_pair[(point["a"], point["b"])].saving_cycles)

    rows = grid_search({"a": [p.a for p in hr.pairs], "b": [p.b for p in hr.pairs]},
                       lambda pt: _evaluate(pt) if (pt["a"], pt["b"]) in by_pair else -1.0)

    cands: list[Candidate] = []
    for row in rows:
        pair = by_pair.get((row["a"], row["b"]))
        if pair is None or pair.saving_cycles <= 0:
            continue
        hi = float(pair.saving_cycles)
        lo: float | _Unknown = 0.0 if pair.is_upper_bound else hi
        cands.append(Candidate(
            axis=Axis.OVERLAP, workload=source.workload,
            setting=(("group_a", pair.a), ("group_b", pair.b)),
            baseline_cycles=hr.total_cycles, saving_hi=hi, saving_lo=lo,
            is_upper_bound=pair.is_upper_bound,
            rationale=(f"{pair.a} is busy {pair.busy_a} cycles and {pair.b} {pair.busy_b}; running "
                       f"them together saves at most min(a, b) = {pair.saving_cycles} of "
                       f"{hr.total_cycles}"
                       + ("; the realised overlap is unobserved, so this is a ceiling"
                          if pair.is_upper_bound else "; measured against the realised overlap"))))
    if not cands:
        return AxisEvidence(Axis.OVERLAP, source.workload, False,
                            rationale=(f"{source.workload}: no concurrency-capable pair has any "
                                       f"overlappable time ({hr.grouping})"))
    cands.sort(key=lambda c: (-float(c.saving_hi), c.id))
    return AxisEvidence(Axis.OVERLAP, source.workload, True, tuple(cands),
                        rationale=f"{len(cands)} concurrency-capable pair(s) with headroom")


def derive_axes(source: ActivitySource, *,
                amplification: "WorkloadAmplification | Unavailable | None" = None,
                headroom: "WorkloadHeadroom | Unavailable | None" = None,
                ) -> dict[Axis, AxisEvidence]:
    """Both axes for one workload. Every axis is always present in the result, established or not --
    an axis dropped because its evidence was missing is an axis a reader cannot tell from one that
    was never asked about."""
    return {Axis.DMA_TILING: dma_axis(source, amplification),
            Axis.OVERLAP: overlap_axis(source, headroom)}


def candidates_from(axes: "Mapping[Axis, AxisEvidence] | Iterable[AxisEvidence]") -> list[Candidate]:
    """Flatten established axes into one candidate list. The axes are NOT multiplied together: a
    compound point (a descriptor change AND an overlap change) predicts a saving only under a
    composition rule the corpus does not establish, and inventing one here would price a combination
    nothing has measured."""
    evs = list(axes.values()) if isinstance(axes, Mapping) else list(axes)
    out: list[Candidate] = []
    for ev in evs:
        if ev.established:
            out.extend(ev.candidates)
    out.sort(key=lambda c: (c.workload, c.axis.value, c.id))
    return out


# --- value of information ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VOI:
    """``Impact x Uncertainty / Cost`` for one candidate, with every factor kept separate.

    Any UNKNOWN factor makes the score UNKNOWN. It is never coerced to zero: "this candidate is
    worthless" and "nobody can price this candidate" rank identically only in a tool that has stopped
    distinguishing them.
    """

    candidate_id: str
    axis: Axis
    impact: "float | _Unknown"
    uncertainty: "float | _Unknown"
    cost_units: "float | _Unknown"
    score: "float | _Unknown"
    unit_name: str
    missing: tuple[str, ...] = ()
    rationale: str = ""

    @property
    def known(self) -> bool:
        return not is_unknown(self.score)

    def to_dict(self) -> dict:
        def _s(v: object) -> object:
            return "UNKNOWN" if is_unknown(v) else v

        return {"candidate_id": self.candidate_id, "axis": self.axis.value,
                "impact": _s(self.impact), "uncertainty": _s(self.uncertainty),
                "cost_units": _s(self.cost_units), "score": _s(self.score),
                "unit": self.unit_name, "missing": list(self.missing),
                "rationale": self.rationale, "generality": GENERALITY_DROPPED}


def voi(candidate: Candidate, *, reference_cycles: int, budget: Budget,
        cost_units: float = 1.0) -> VOI:
    """Score one candidate. ``reference_cycles`` is the runtime the impact is a share OF -- pass the
    corpus total when ranking across workloads, so a large saving on a tiny workload does not
    outrank a small saving on the workload that dominates the run.

    ``cost_units`` is denominated in ``budget.unit`` -- the channel MEASUREMENT said is scarce. That
    is the whole point of the parameter: on one target a candidate costs one agent synthesis call and
    the simulator time is rounding error; on another the same candidate costs a tens-of-minutes
    simulation and the ranking inverts. Neither is written down here.
    """
    unit = budget.unit.name
    missing: list[str] = []
    if reference_cycles <= 0:
        return VOI(candidate.id, candidate.axis, UNKNOWN, UNKNOWN, cost_units, UNKNOWN, unit,
                   ("a positive reference cycle count",),
                   "impact is a share of a runtime; a zero reference has no shares")

    hi, lo = candidate.saving_hi, candidate.saving_lo
    if is_unknown(hi):
        impact: float | _Unknown = UNKNOWN
        missing.append("the candidate's best-case saving")
    else:
        impact = float(hi) / reference_cycles

    if is_unknown(hi) or is_unknown(lo) or float(hi) <= 0:
        uncertainty: float | _Unknown = UNKNOWN
        if not is_unknown(hi) and float(hi) > 0:
            missing.append("the backed (lower) end of the saving interval")
    else:
        uncertainty = max(0.0, min(1.0, (float(hi) - float(lo)) / float(hi)))

    if cost_units is None or cost_units <= 0:
        cost: float | _Unknown = UNKNOWN
        missing.append(f"a positive cost in {unit} items")
    else:
        cost = float(cost_units)

    if is_unknown(impact) or is_unknown(uncertainty) or is_unknown(cost):
        score: float | _Unknown = UNKNOWN
    else:
        score = float(impact) * float(uncertainty) / float(cost)

    return VOI(candidate.id, candidate.axis, impact, uncertainty, cost, score, unit,
               tuple(missing),
               (f"impact = {'UNKNOWN' if is_unknown(impact) else format(impact, '.4g')} of "
                f"{reference_cycles} reference cycles; uncertainty = "
                f"{'UNKNOWN' if is_unknown(uncertainty) else format(uncertainty, '.3g')} "
                f"(interval width / ceiling); cost = "
                f"{'UNKNOWN' if is_unknown(cost) else format(cost, '.4g')} {unit} item(s)"))


def rank(candidates: Sequence[Candidate], *, reference_cycles: int, budget: Budget,
         cost_units: "float | Mapping[str, float]" = 1.0) -> list[VOI]:
    """Candidates by descending VOI. Unscorable candidates sort LAST but are RETAINED, with their
    missing evidence attached -- dropping them would hide the hole the axis has."""
    per = cost_units if isinstance(cost_units, Mapping) else None
    out = [voi(c, reference_cycles=reference_cycles, budget=budget,
               cost_units=float(per.get(c.id, 1.0)) if per else float(cost_units))
           for c in candidates]
    out.sort(key=lambda v: (v.known is False, -(float(v.score) if v.known else 0.0), v.candidate_id))
    return out


# --- stop conditions ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class StopPolicy:
    """Declared thresholds. These are CHOICES, not measurements, and are named so a report can say
    which choice produced the stop."""

    #: Stop once measured performance reaches this fraction of the conservative attainable target.
    attainment_fraction: float = 0.90
    #: Stop once the best remaining candidate predicts less than this fractional improvement.
    predicted_remaining: float = 0.03
    #: An improvement below this fraction counts as no improvement.
    plateau_improvement: float = 0.01
    #: How many consecutive such queries end the search.
    plateau_queries: int = 3


@dataclass(frozen=True)
class StopVerdict:
    """One stop condition's answer, always with a reason -- including when it did NOT fire."""

    name: str
    fired: bool
    reason: str
    missing: tuple[str, ...] = ()
    #: False when this condition CANNOT be answered in the caller's wiring, as opposed to having
    #: been answered "no". The two are the same `fired: False` to a reader, and that is how a
    #: condition that has never once been able to contribute comes to sit beside live ones looking
    #: like a check that keeps passing. Measured: `predicted_remaining_below` returned the same
    #: `missing` on 8 of 8 calls of a recorded run, because nothing in that configuration enumerates
    #: an unevaluated candidate for it to price -- there, the search's candidate generator IS the
    #: agent, so the host holds no pool. That is correct behaviour and it must still be visible.
    evaluable: bool = True

    def to_dict(self) -> dict:
        return {"name": self.name, "fired": self.fired, "reason": self.reason,
                "missing": list(self.missing), "evaluable": self.evaluable}


@dataclass(frozen=True)
class SearchState:
    """Everything the stop conditions read. Kept separate from the loop so each predicate can be
    tested on a hand-built state with no evaluator, no oracle and no budget spend."""

    baseline_cycles: int
    best_cycles: "float | _Unknown"
    budget: Budget
    #: The conservative attainable target, in cycles -- a LOWER bound on runtime (the structural
    #: envelope's resolved-resources-only bound is the conservative one). UNKNOWN when unresolved.
    attainable_cycles: "float | _Unknown" = UNKNOWN
    #: Best cycles predicted by any candidate still unevaluated. UNKNOWN when nothing is left or
    #: nothing left can be priced.
    predicted_best_cycles: "float | _Unknown" = UNKNOWN
    #: Fractional improvement contributed by each query, oldest first.
    improvements: tuple[float, ...] = ()

    @property
    def queries(self) -> int:
        return len(self.improvements)


def attainment_reached(state: SearchState, policy: StopPolicy) -> StopVerdict:
    """Measured performance has reached ``attainment_fraction`` of the conservative attainable target.

    Compared as a performance RATIO, not a cycle difference: ``attainable / measured`` is 1.0 at the
    bound and falls as the measurement is slower, so 0.9 means "within 11% of the bound". An UNKNOWN
    target does not fire this -- an unresolved bound must never read as a bound that was reached.
    """
    name = "attainment_reached"
    if is_unknown(state.attainable_cycles):
        return StopVerdict(name, False,
                           "the conservative attainable target is UNKNOWN, so attainment cannot be "
                           "evaluated; not stopping",
                           ("a resolved structural bound for this workload",))
    if is_unknown(state.best_cycles) or float(state.best_cycles) <= 0:
        return StopVerdict(name, False, "no measured cycle count yet; not stopping",
                           ("at least one evaluated candidate",))
    ratio = float(state.attainable_cycles) / float(state.best_cycles)
    if ratio >= policy.attainment_fraction:
        return StopVerdict(name, True,
                           f"measured {float(state.best_cycles):.0f} cycles is {ratio:.1%} of the "
                           f"conservative attainable {float(state.attainable_cycles):.0f}, at or "
                           f"above the {policy.attainment_fraction:.0%} policy threshold")
    return StopVerdict(name, False,
                       f"measured {float(state.best_cycles):.0f} cycles is {ratio:.1%} of the "
                       f"conservative attainable {float(state.attainable_cycles):.0f}, below the "
                       f"{policy.attainment_fraction:.0%} threshold")


def predicted_remaining_below(state: SearchState, policy: StopPolicy) -> StopVerdict:
    """The best remaining candidate predicts less than ``predicted_remaining`` improvement.

    Uses the candidates' own predictions, so it can stop a search before paying for a query that the
    model already says is not worth making. UNKNOWN predictions do not fire it: a candidate nobody
    could price is not evidence that nothing is left.
    """
    name = "predicted_remaining_below"
    if is_unknown(state.best_cycles) or float(state.best_cycles) <= 0:
        return StopVerdict(name, False, "no measured cycle count to improve on yet; not stopping",
                           ("at least one evaluated candidate",))
    if is_unknown(state.predicted_best_cycles):
        # NOT EVALUABLE, rather than evaluated and negative. A caller that never enumerates an
        # unevaluated candidate can never supply this, so reporting it as a plain "did not fire"
        # puts a condition that cannot contribute beside three that can.
        return StopVerdict(name, False,
                           "no remaining candidate carries a prediction, so the remaining "
                           "improvement is UNKNOWN; not stopping",
                           ("a predicted cycle count for at least one unevaluated candidate",),
                           evaluable=False)
    remaining = (float(state.best_cycles) - float(state.predicted_best_cycles)) / float(state.best_cycles)
    if remaining < policy.predicted_remaining:
        return StopVerdict(name, True,
                           f"the best remaining candidate predicts {remaining:.2%} improvement over "
                           f"{float(state.best_cycles):.0f} cycles, below the "
                           f"{policy.predicted_remaining:.0%} policy threshold")
    return StopVerdict(name, False,
                       f"the best remaining candidate predicts {remaining:.2%} improvement, at or "
                       f"above the {policy.predicted_remaining:.0%} threshold")


def plateaued(state: SearchState, policy: StopPolicy) -> StopVerdict:
    """``plateau_queries`` consecutive queries each improved by less than ``plateau_improvement``.

    Requires that many queries to have actually happened. Firing on a shorter history would stop a
    search at its first flat query, which is the normal shape of an exploration that has not started.
    """
    name = "plateaued"
    n = policy.plateau_queries
    if len(state.improvements) < n:
        return StopVerdict(name, False,
                           f"only {len(state.improvements)} quer(ies) so far; the rule needs {n} "
                           f"consecutive ones")
    last = state.improvements[-n:]
    if all(i < policy.plateau_improvement for i in last):
        shown = ", ".join(f"{i:.2%}" for i in last)
        return StopVerdict(name, True,
                           f"the last {n} queries improved by {shown}, each below the "
                           f"{policy.plateau_improvement:.0%} policy threshold")
    best = max(last)
    return StopVerdict(name, False,
                       f"the last {n} queries include one improving {best:.2%}, at or above the "
                       f"{policy.plateau_improvement:.0%} threshold")


def budget_exhausted(state: SearchState, policy: StopPolicy) -> StopVerdict:
    """The budget is spent, in whichever unit measurement said is scarce."""
    name = "budget_exhausted"
    why = state.budget.exhausted_reason
    if why is not None:
        return StopVerdict(name, True,
                           f"budget denominated in {state.budget.unit.name!r}: {why}")
    remaining = state.budget.remaining_items
    left = "unbounded" if is_unknown(remaining) else f"{float(remaining):g}"
    return StopVerdict(name, False,
                       f"{state.budget.spent_items:g} {state.budget.unit.name} item(s) spent, "
                       f"{left} remaining")


#: The four conditions, in the order a report should read them.
STOP_CONDITIONS = (attainment_reached, predicted_remaining_below, plateaued, budget_exhausted)


def check_stop(state: SearchState, policy: StopPolicy | None = None) -> tuple[StopVerdict, ...]:
    """Every condition's verdict, fired or not. All four are always returned: a condition omitted
    because it did not fire is a condition a reader cannot tell from one that was not checked."""
    pol = policy or StopPolicy()
    return tuple(cond(state, pol) for cond in STOP_CONDITIONS)


def fired(verdicts: Sequence[StopVerdict]) -> tuple[StopVerdict, ...]:
    return tuple(v for v in verdicts if v.fired)


# --- the loop ----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class QueryRecord:
    """One evaluated candidate, and where the search stood after it."""

    index: int
    candidate_id: str
    axis: Axis
    digest: str
    measured_cycles: "float | _Unknown"
    best_cycles: "float | _Unknown"
    improvement: float
    voi_score: "float | _Unknown"
    #: Cumulative spend in the scarce unit, which is what a convergence curve is plotted against.
    cumulative_items: float
    cumulative_seconds: float
    cumulative_dollars: float

    def to_dict(self) -> dict:
        def _s(v: object) -> object:
            return "UNKNOWN" if is_unknown(v) else v

        return {"index": self.index, "candidate_id": self.candidate_id, "axis": self.axis.value,
                "digest": self.digest, "measured_cycles": _s(self.measured_cycles),
                "best_cycles": _s(self.best_cycles), "improvement": self.improvement,
                "voi_score": _s(self.voi_score), "cumulative_items": self.cumulative_items,
                "cumulative_seconds": self.cumulative_seconds,
                "cumulative_dollars": self.cumulative_dollars}


@dataclass
class SearchResult:
    """What a bounded selection run did, and why it stopped."""

    baseline_cycles: int
    best_cycles: "float | _Unknown"
    best_candidate_id: str | None
    queries: list[QueryRecord] = field(default_factory=list)
    stop: tuple[StopVerdict, ...] = ()
    ranked: tuple[VOI, ...] = ()
    budget: Budget | None = None
    policy: StopPolicy = field(default_factory=StopPolicy)
    #: Candidates never evaluated, with the reason -- budget, cache hit, or an unscorable VOI.
    skipped: tuple[tuple[str, str], ...] = ()

    @property
    def stopped_by(self) -> tuple[str, ...]:
        return tuple(v.name for v in self.stop if v.fired)

    @property
    def improvement(self) -> "float | _Unknown":
        if is_unknown(self.best_cycles) or self.baseline_cycles <= 0:
            return UNKNOWN
        return (self.baseline_cycles - float(self.best_cycles)) / self.baseline_cycles

    def to_dict(self) -> dict:
        def _s(v: object) -> object:
            return "UNKNOWN" if is_unknown(v) else v

        return {"baseline_cycles": self.baseline_cycles, "best_cycles": _s(self.best_cycles),
                "best_candidate_id": self.best_candidate_id,
                "improvement": _s(self.improvement),
                "queries": [q.to_dict() for q in self.queries],
                "stop": [v.to_dict() for v in self.stop],
                "stopped_by": list(self.stopped_by),
                "ranked": [v.to_dict() for v in self.ranked],
                "budget": self.budget.to_dict() if self.budget else None,
                "policy": {"attainment_fraction": self.policy.attainment_fraction,
                           "predicted_remaining": self.policy.predicted_remaining,
                           "plateau_improvement": self.policy.plateau_improvement,
                           "plateau_queries": self.policy.plateau_queries},
                "skipped": [{"candidate_id": c, "reason": r} for c, r in self.skipped],
                "generality": GENERALITY_DROPPED}


def search(candidates: Sequence[Candidate], *, evaluate, budget: Budget,
           baseline_cycles: int, reference_cycles: int | None = None,
           attainable_cycles: "float | _Unknown" = UNKNOWN,
           policy: StopPolicy | None = None,
           cost_units: "float | Mapping[str, float]" = 1.0) -> SearchResult:
    """Spend ``budget`` on the highest-VOI candidates until a stop condition fires.

    ``evaluate(candidate) -> measured cycles`` is the only thing that touches an oracle, so the whole
    loop is testable against a deterministic fake. A candidate whose digest has already been
    evaluated is served from the cache and charged NOTHING -- the same content-addressing rule
    :mod:`merlin.targetgen.oracle_schedule` applies to verdicts, for the same reason: identical bytes
    cannot produce a different answer, so paying for them again buys nothing.

    Stop conditions are checked BEFORE each query as well as after the last one, so an exhausted
    budget or an already-reached target costs zero queries rather than one.
    """
    pol = policy or StopPolicy()
    ref = reference_cycles if reference_cycles is not None else baseline_cycles
    ranked = rank(candidates, reference_cycles=ref, budget=budget, cost_units=cost_units)
    by_id = {c.id: c for c in candidates}
    per_cost = cost_units if isinstance(cost_units, Mapping) else None

    best: float | _Unknown = UNKNOWN
    best_id: str | None = None
    improvements: list[float] = []
    queries: list[QueryRecord] = []
    skipped: list[tuple[str, str]] = []
    #: One :class:`~merlin.targetgen.oracle_schedule.CapsuleState` per DIGEST, so the "a verdict
    #: earned by different bytes is not a verdict about this submission" rule is the one that
    #: decides re-evaluation here too, rather than a second hand-rolled cache that can drift from it.
    seen: dict[str, CapsuleState] = {}
    pending = list(ranked)

    def _state() -> SearchState:
        remaining = [by_id[v.candidate_id].predicted_cycles for v in pending]
        known = [float(r) for r in remaining if not is_unknown(r)]
        return SearchState(baseline_cycles=baseline_cycles, best_cycles=best, budget=budget,
                           attainable_cycles=attainable_cycles,
                           predicted_best_cycles=min(known) if known else UNKNOWN,
                           improvements=tuple(improvements))

    verdicts = check_stop(_state(), pol)
    while pending and not fired(verdicts):
        v = pending.pop(0)
        cand = by_id[v.candidate_id]
        st = seen.setdefault(cand.digest, CapsuleState(name=cand.id, digest=cand.digest))
        if st.known(EVALUATED) != VERDICT_UNKNOWN:
            skipped.append((cand.id, f"identical to an already-evaluated candidate "
                                     f"(digest {cand.digest}); served from cache, charged nothing"))
            verdicts = check_stop(_state(), pol)
            continue
        items = float(per_cost.get(cand.id, 1.0)) if per_cost else (
            float(cost_units) if not isinstance(cost_units, Mapping) else 1.0)
        ok, why = budget.can_afford(items)
        if not ok:
            skipped.append((cand.id, why or "budget"))
            pending.insert(0, v)          # not a verdict on it; it simply did not run
            verdicts = check_stop(_state(), pol)
            break
        budget.charge(items=items, label=cand.id)
        measured = evaluate(cand)
        st.verdicts[EVALUATED] = Verdict(status=PASS, digest=cand.digest)

        prev = best
        if not is_unknown(measured) and (is_unknown(best) or float(measured) < float(best)):
            best, best_id = float(measured), cand.id
        gain = 0.0
        if not is_unknown(prev) and not is_unknown(best) and float(prev) > 0:
            gain = max(0.0, (float(prev) - float(best)) / float(prev))
        elif is_unknown(prev) and not is_unknown(best) and baseline_cycles > 0:
            gain = max(0.0, (baseline_cycles - float(best)) / baseline_cycles)
        improvements.append(gain)

        queries.append(QueryRecord(
            index=len(queries) + 1, candidate_id=cand.id, axis=cand.axis, digest=cand.digest,
            measured_cycles=measured, best_cycles=best, improvement=gain, voi_score=v.score,
            cumulative_items=budget.spent_items, cumulative_seconds=budget.spent_seconds,
            cumulative_dollars=budget.spent_dollars))
        verdicts = check_stop(_state(), pol)

    for v in pending:
        if not any(v.candidate_id == c for c, _ in skipped):
            skipped.append((v.candidate_id, "not reached before the search stopped"))

    return SearchResult(baseline_cycles=baseline_cycles, best_cycles=best, best_candidate_id=best_id,
                        queries=queries, stop=verdicts, ranked=tuple(ranked), budget=budget,
                        policy=pol, skipped=tuple(skipped))


# --- convergence curve -------------------------------------------------------------------------------

def convergence_rows(result: SearchResult) -> list[dict]:
    """The curve: best-so-far against cumulative spend IN THE MEASURED SCARCE UNIT.

    The x axis is the point. Plotting against "queries" would be plotting against the unit the
    original design assumed was scarce; plotting against ``cumulative_items`` of
    ``budget.unit`` plots against whichever unit measurement actually elected, and the same code
    produces a differently-denominated curve on a target where the simulator is the expensive thing.
    """
    unit = result.budget.unit.name if result.budget else "unit"
    rows = [{"query": 0, "cumulative_items": 0.0, "cumulative_seconds": 0.0,
             "cumulative_dollars": 0.0, "best_cycles": float(result.baseline_cycles),
             "improvement": 0.0, "unit": unit, "candidate_id": ""}]
    for q in result.queries:
        rows.append({"query": q.index, "cumulative_items": q.cumulative_items,
                     "cumulative_seconds": q.cumulative_seconds,
                     "cumulative_dollars": q.cumulative_dollars,
                     "best_cycles": (float(q.best_cycles) if not is_unknown(q.best_cycles)
                                     else float(result.baseline_cycles)),
                     "improvement": q.improvement, "unit": unit, "candidate_id": q.candidate_id})
    return rows


def write_convergence(result: SearchResult, out_dir: Path, *, stem: str = "convergence") -> dict:
    """Write the curve as CSV + JSON, and a PNG when a plotting backend is importable.

    Returns ``{"csv":…, "json":…, "png": … | None, "plot": "written"|"not_run: …"}``. A plot that
    could not be drawn is reported as ``not_run``, never omitted and never counted as written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = convergence_rows(result)
    unit = result.budget.unit if result.budget else None
    unit_name = unit.name if unit else "unit"

    cols = ["query", "cumulative_items", "cumulative_seconds", "cumulative_dollars",
            "best_cycles", "improvement", "candidate_id"]
    csv_path = out_dir / f"{stem}.csv"
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r[c]) for c in cols))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(
        {"unit": unit.to_dict() if unit else None, "rows": rows, "result": result.to_dict()},
        indent=2, default=str) + "\n", encoding="utf-8")

    png_path: Path | None = None
    status = "not_run: no plotting backend"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001 -- an optional plotting dependency
        status = f"not_run: {type(exc).__name__}: {exc}"
    else:
        fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=160)
        ax.step([r["cumulative_items"] for r in rows], [r["best_cycles"] for r in rows],
                where="post", marker="o", markersize=3.0, linewidth=1.4)
        price = ""
        if unit is not None and unit.seconds_per_item is not None:
            price = f" — {unit.seconds_per_item:.3g} s/item, measured"
            if unit.dollars_per_item is not None:
                price += f", ${unit.dollars_per_item:.2f}/item"
        ax.set_xlabel(f"cumulative {unit_name} items{price}")
        ax.set_ylabel("best measured cycles")
        ax.set_title("Convergence over the measured scarce unit")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        for v in result.stop:
            if v.fired and rows:
                ax.axvline(rows[-1]["cumulative_items"], linestyle="--", linewidth=0.9, alpha=0.6)
                ax.annotate(v.name, xy=(rows[-1]["cumulative_items"], rows[-1]["best_cycles"]),
                            xytext=(-4, 10), textcoords="offset points", ha="right", fontsize=7)
                break
        fig.tight_layout()
        png_path = out_dir / f"{stem}.png"
        fig.savefig(png_path)
        plt.close(fig)
        status = "written"

    return {"csv": csv_path, "json": json_path, "png": png_path, "plot": status}


#: The concern a selection run's products belong to: it is a statement about where a target's
#: optimization surface has room, keyed by target at folder level like every other product here.
PRODUCT_TOPIC = "optimization-surface"


def emit_product(result: SearchResult, *, target: str, version: int = 1, notes: str = "") -> dict:
    """Write the run's curve and result into a versioned product dir under the ``out/`` root.

    Everything generated goes through :mod:`merlin.common.artifacts`; nothing here builds a path by
    hand. Returns the :func:`~merlin.common.artifacts.new_product` handle plus whatever
    :func:`write_convergence` reported, including a plot that could not be drawn -- which is recorded
    as ``not_run``, never as a success.
    """
    from merlin.common.artifacts import new_product

    prod = new_product(PRODUCT_TOPIC, version=version, target=target,
                       notes=notes or "bounded candidate selection over the two derived axes")
    written = write_convergence(result, prod.path)
    for name in ("convergence.csv", "convergence.json"):
        prod.add_artifact(name)
    if written["png"] is not None:
        prod.add_artifact("convergence.png")
    prod.write_manifest()
    return {"product": prod, **written}
