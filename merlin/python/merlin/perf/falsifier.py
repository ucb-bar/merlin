"""eta as a falsifier: did the reordering actually buy overlap, or only survive the hardware?

On a hardware-interlocked, command-driven accelerator a reservation station resolves every hazard,
so *every* reordering of the command stream returns the same answer. A scheduling capsule whose
falsifier is bit-exactness therefore passes every candidate it is ever handed, and a falsifier that
cannot fire establishes nothing (``docs/design/performance_levers_per_archetype.md``). The lever that
does exist there is overlap -- movement under compute, an operand held resident across tiles, an
issue hoisted ahead of a wait -- and the only quantity that can reject a schedule is therefore

    eta = realised overlap cycles / available overlap cycles

measured on a JOINT occupancy vector. This module turns that ratio into a verdict a capsule can be
graded on, and its whole design is about the ways the ratio lies.

**Three states, never two.** ``ROSE`` / ``DID_NOT_RISE`` / ``UNDETERMINABLE``. The third is not a
polite version of the second: this tree has a recurring bug class in which a thing that could not be
measured was reported as a measured zero (a joint vector with one live column reporting 0% overlap;
a unit with no busy port reading as permanently idle and moving one kernel's idle fraction from 89.9%
to 39.2%; a 1 MiB window in our own harness recorded as a hardware limit). eta is exactly where that
recurs, because the arithmetic always produces a number. So every refusal below returns
``UNDETERMINABLE`` with the reason attached, and a candidate whose eta is undeterminable is NOT
promoted -- an unfired falsifier is not a passed one.

**What makes an eta comparison sound**, each checked rather than assumed:

* *the vector could have shown overlap at all* -- fewer than two live engines reports zero
  arithmetically, and that zero is indistinguishable from a machine that genuinely serialises. The
  check is at ENGINE level, one step stronger than
  :func:`~merlin.perf.occupancy.joint_counts`'s ``overlap_observable``: two live columns inside ONE
  declared engine still cannot show two engines running together;
* *nothing was left unread* -- a unit the instrument did not sample cannot be shown idle, and adding
  its cycles back can move realised overlap in either direction, so an unmeasured unit refuses the
  reading rather than shrinking it;
* *the two runs name the same engines* -- an engine present on one side only is unmeasured there, not
  zero, and scoring it zero reports moving work off an engine as speeding it up
  (:func:`~merlin.perf.differential.compare_by_engine` refuses the same way);
* *the two runs measured on the same axis* -- an eta taken over declared ENGINES and one taken over
  resource KINDS are two instruments, not two readings;
* *the two runs did the same work* -- eta is a ratio, so halving the work can raise it. A stated work
  fingerprint that differs refuses the comparison; two runs that state none are compared on the
  caller's word, which is recorded in the verdict.

The denominator is deliberately the SAME one :func:`~merlin.perf.headroom.composition_operator`
uses -- the second-largest per-engine busy count, i.e. ``min`` over the busiest pair, which is the
most any single pair could ever overlap. Two etas that mean different things must not be compared,
and the ledger's eta is the one a reader will hold this verdict against. With three or more engines
overlapping in disjoint pairs the ratio can exceed 1; that is a true statement about the vector (more
than one pair overlapped) and is reported, not clipped.

Nothing here names a target, a unit, an engine or a kind: the engine set is the producer's
declaration and arrives as a parameter.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

__all__ = [
    "ABDecision", "ACCEPT", "DID_NOT_RISE", "ENGINE_AXIS", "EtaObservation", "EtaVerdict",
    "KIND_AXIS", "REJECT", "ROSE", "UNDETERMINABLE", "ab_decision", "compare_eta",
    "eta_from_occupancy", "eta_from_timing_block",
]

#: The three -- and only three -- answers to "did eta rise". ``DID_NOT_RISE`` covers equal and fell
#: (:attr:`EtaVerdict.fell` separates them); ``UNDETERMINABLE`` is never folded into it.
ROSE = "rose"
DID_NOT_RISE = "did_not_rise"
#: Spelled identically to :data:`merlin.targetgen.oracle_schedule.UNDETERMINABLE`, and meaning the
#: same thing: the tooling could not decide, which is neither a pass nor a fail.
UNDETERMINABLE = "undeterminable"

#: The A/B gate's verdicts. ``UNDETERMINABLE`` again does not promote -- see the module docstring.
ACCEPT = "accept"
REJECT = "reject"

#: Which axis an observation resolved its concurrency on. A vector bound to the target's DECLARED
#: engines and one bucketed by resource KIND are different instruments: the engine axis can see two
#: engines of one kind running together, the kind axis cannot, so the same hardware yields two
#: different etas. Comparing across them is refused.
ENGINE_AXIS = "engine"
KIND_AXIS = "kind"


def _live(busy: Mapping[str, int], cycles: int) -> list[str]:
    """The groups that could have shown overlap: busy at least once and not busy for the whole run.

    Same rule as :func:`~merlin.perf.occupancy.joint_counts` applies to columns, for the same reason:
    a group that is high on every sampled cycle is constant, and nothing was observed of it either
    way. The fallback to "merely busy at least once" is also carried over, so a short vector in which
    everything happens to be saturated is not emptied out.
    """
    varying = [g for g, n in busy.items() if 0 < n < cycles]
    return varying or [g for g, n in busy.items() if n > 0]


@dataclass(frozen=True)
class EtaObservation:
    """One run's realised overlap, its ceiling, and -- when there is none -- why there is none.

    ``realised_cycles`` and ``available_cycles`` are ``None`` together whenever the vector cannot
    support a reading. They are never 0: a zero from an unobservable vector is the exact failure this
    module exists to prevent.
    """

    label: str
    #: Cycles in which two or more groups on :attr:`axis` were busy together.
    realised_cycles: int | None
    #: The ceiling on that: ``min`` over the busiest pair, i.e. the second-largest group busy count.
    available_cycles: int | None
    #: The groups the vector RESOLVED -- not the groups that happened to be busy. A group present
    #: with zero busy is measured-and-zero; a group absent from the vector is unmeasured, and only
    #: the second refuses a comparison.
    engines: tuple[str, ...]
    busy: dict[str, int] = field(default_factory=dict)
    sampled_cycles: int = 0
    axis: str = ENGINE_AXIS
    #: The caller's fingerprint for the work this run performed. ``None`` == not stated.
    work: str | None = None
    #: Why there is no reading, or what the reading rests on. Always populated.
    detail: str = ""
    #: Units the instrument states it did not read. Non-empty forces UNDETERMINABLE.
    unmeasured: tuple[str, ...] = ()

    @property
    def measured(self) -> bool:
        return self.realised_cycles is not None and self.available_cycles not in (None, 0)

    @property
    def eta(self) -> float | None:
        """The realised fraction of available overlap, or ``None`` when it is not measurable.

        ``None`` is returned for an unobservable vector AND for a zero denominator. The second is
        easy to miss: a run where only one group is ever busy has no overlappable time, so 0/0 is
        undefined -- and returning 0.0 would report "this schedule realises none of its overlap"
        about a schedule that had none available.
        """
        if not self.measured:
            return None
        return self.realised_cycles / self.available_cycles

    def to_dict(self) -> dict:
        return {"label": self.label, "axis": self.axis, "eta": self.eta,
                "realised_cycles": self.realised_cycles,
                "available_cycles": self.available_cycles,
                "engines": list(self.engines), "busy": dict(self.busy),
                "sampled_cycles": self.sampled_cycles, "work": self.work,
                "unmeasured_units": list(self.unmeasured), "detail": self.detail}


def _refusal(label: str, detail: str, *, engines=(), busy=None, cycles=0, axis=ENGINE_AXIS,
             work=None, unmeasured=()) -> EtaObservation:
    return EtaObservation(label=label, realised_cycles=None, available_cycles=None,
                          engines=tuple(engines), busy=dict(busy or {}), sampled_cycles=cycles,
                          axis=axis, work=work, detail=detail, unmeasured=tuple(unmeasured))


def _observation(label: str, busy: Mapping[str, int], realised: int, *, cycles: int, axis: str,
                 work: str | None, detail: str) -> EtaObservation:
    """Assemble a reading from per-group busy counts and a realised overlap count.

    ``available`` is the second-largest busy count. That is the largest ceiling any single pair has,
    and it is the denominator :func:`~merlin.perf.headroom.composition_operator` uses -- deliberately
    the same number, so this verdict's eta and the perf ledger's eta are the same quantity.
    """
    vals = sorted(busy.values(), reverse=True)
    available = vals[1] if len(vals) > 1 else 0
    if available == 0:
        return _refusal(label, ("no pair of groups has any overlappable time (the second-busiest "
                                "group is busy 0 cycles), so eta is 0/0 -- undefined, not zero"),
                        engines=sorted(busy), busy=busy, cycles=cycles, axis=axis, work=work)
    return EtaObservation(label=label, realised_cycles=int(realised), available_cycles=int(available),
                          engines=tuple(sorted(busy)), busy=dict(busy), sampled_cycles=cycles,
                          axis=axis, work=work, detail=detail)


def eta_from_occupancy(label: str, hot: Mapping[str, Sequence[bool]], *,
                       unit_of: Mapping[str, str],
                       kinds: Mapping[str, str] | None = None,
                       work: str | None = None,
                       unmeasured: Sequence[str] = ()) -> EtaObservation:
    """eta over a per-cycle joint occupancy vector, on the target's DECLARED engine axis.

    ``hot`` is ``{column: [busy per cycle]}`` and ``unit_of`` is the producer's column -> engine
    binding. Subsumption and the observability flag come from
    :func:`~merlin.perf.occupancy.joint_counts` rather than being re-derived, so a signal counted
    beside its own components (204 fabricated overlap cycles on one measured design) is removed here
    too, and two columns in different DECLARED engines are never folded into each other.

    A retained column that carries busy cycles but no binding refuses the reading. It cannot be
    dropped (its cycles are real) and it cannot be attributed (nobody said to what), so which engines
    were running together is genuinely unknown -- and an engine-axis eta computed without it is a
    number about a vector that was never measured.
    """
    from merlin.perf.occupancy import joint_counts

    jc = joint_counts(hot, kinds, unit_of)
    cycles = int(jc["sampled_cycles"])
    cols = list(jc["joint_columns"])
    unbound = sorted(c for c in cols if c not in unit_of and any(hot[c]))
    if unbound:
        return _refusal(label, (f"column(s) {unbound} carry busy cycles but are bound to no declared "
                                "engine; which engines ran together cannot be established from this "
                                "vector"), cycles=cycles, work=work, unmeasured=unmeasured)
    if unmeasured:
        return _refusal(label, (f"the instrument states it did not read unit(s) {sorted(unmeasured)}; "
                                "an unread unit is UNKNOWN, never idle, and restoring its cycles can "
                                "move realised overlap in either direction"),
                        cycles=cycles, work=work, unmeasured=unmeasured)

    engines = sorted({unit_of[c] for c in cols if c in unit_of})
    eng_hot = {e: [any(hot[c][i] for c in cols if unit_of.get(c) == e) for i in range(cycles)]
               for e in engines}
    busy = {e: sum(v) for e, v in eng_hot.items()}

    # Two separate observability gates, and both are needed. joint_counts answers "could ANY two
    # columns have overlapped"; this module's question is one step narrower -- could any two declared
    # ENGINES have. A vector whose two live columns sit inside one engine passes the first and fails
    # the second, and reporting its zero as an eta would be the column-level version of the same
    # measured mistake one level up.
    live = _live(busy, cycles)
    if not jc["overlap_observable"]:
        return _refusal(label, ("fewer than two live columns: this vector reports zero overlap "
                                "arithmetically and could not have reported anything else"),
                        engines=engines, busy=busy, cycles=cycles, work=work)
    if len(live) < 2:
        return _refusal(label, (f"only {len(live)} declared engine(s) are live ({sorted(live)}); the "
                                "columns that do vary belong to one engine, so no engine pair could "
                                "have been seen running together"),
                        engines=engines, busy=busy, cycles=cycles, work=work)

    realised = sum(1 for i in range(cycles) if sum(1 for e in engines if eng_hot[e][i]) >= 2)
    return _observation(label, busy, realised, cycles=cycles, axis=ENGINE_AXIS, work=work,
                        detail=(f"joint occupancy over {len(engines)} declared engine(s), "
                                f"{len(jc['subsumed_columns'])} column(s) folded as sub-signals"))


def eta_from_timing_block(label: str, tier_record, *, work: str | None = None) -> EtaObservation:
    """eta from a graded tier record's timing block, on the resource-KIND axis.

    This is the shape the capsule runner already emits, so a perf A/B can be gated without a second
    instrument. It resolves on :data:`KIND_AXIS` because the block's licensed overlap reading is the
    across-kinds one (two engines of a single kind being busy together is not movement/compute
    overlap), and the busy counts are therefore grouped the same way -- the grouping
    :func:`~merlin.perf.headroom.resource_groups` defaults to.

    Every refusal the block itself makes is honoured rather than worked around: a producer that
    asserts its buckets PARTITION the timeline reports zero overlap by construction and licenses no
    reading at all, and a producer that names units it did not read has left the vector incomplete.
    """
    from merlin.perf.observations import SAMPLED_QUANTITY, block_from_tier_record

    block = block_from_tier_record(tier_record)
    if block is None or not block.usable:
        return _refusal(label, ("the tier record carries no usable timing block, so nothing was "
                                "measured about overlap here"), axis=KIND_AXIS, work=work)
    if block.unmeasured_units:
        return _refusal(label, (f"the instrument states it did not read unit(s) "
                                f"{sorted(block.unmeasured_units)}; an unread unit is UNKNOWN, never "
                                "idle"), axis=KIND_AXIS, work=work,
                        unmeasured=block.unmeasured_units)
    if block.alias_collisions is None or block.alias_collisions > 0:
        # A limit found in our own harness is evidence about the harness -- but an address that
        # collided inside a wrapping window means these cycles are not about the program submitted,
        # and an unstated count means nobody can tell. Both refuse; neither is a zero.
        n = "an unstated number of" if block.alias_collisions is None else str(block.alias_collisions)
        return _refusal(label, (f"{n} access(es) may have collided inside the wrapping memory "
                                "window, so this run's cycles are not established to be about the "
                                "program that was submitted"), axis=KIND_AXIS, work=work)

    realised = block.overlap_cycles()
    if realised is None:
        return _refusal(label, ("the block licenses no overlap reading (it asserts a partitioned "
                                "bucket set, or carries no joint-occupancy entry); a partition "
                                "charges every cycle to one owner and reports zero overlap whether "
                                "or not the hardware overlaps"), axis=KIND_AXIS, work=work)

    busy_by_unit = block.busy_by_unit()
    declared = block.kinds()
    missing = sorted(set(busy_by_unit) - set(declared))
    if missing:
        return _refusal(label, (f"the producer stated no kind for unit(s) {missing}; a role read out "
                                "of a unit's NAME is not a derivation, so the kind axis cannot be "
                                "resolved"), axis=KIND_AXIS, work=work)
    cycles = block.quantity(SAMPLED_QUANTITY) or 0
    busy: dict[str, int] = {}
    for unit, n in busy_by_unit.items():
        busy[declared[unit]] = busy.get(declared[unit], 0) + int(n)

    live = _live(busy, int(cycles))
    if len(live) < 2:
        return _refusal(label, (f"only {len(live)} resource kind(s) are live ({sorted(live)}); this "
                                "reading reports zero overlap arithmetically and could not have "
                                "reported anything else"),
                        engines=sorted(busy), busy=busy, cycles=int(cycles), axis=KIND_AXIS,
                        work=work)
    return _observation(label, busy, int(realised), cycles=int(cycles), axis=KIND_AXIS, work=work,
                        detail=(f"across-kinds joint occupancy over {len(busy)} resource kind(s), "
                                "as licensed by the producer's non-partitioned assertion"))


@dataclass(frozen=True)
class EtaVerdict:
    """Did eta rise between two runs of the same work -- and if that cannot be said, why not."""

    state: str
    base: EtaObservation
    candidate: EtaObservation
    #: ``candidate.eta - base.eta``; ``None`` whenever the state is UNDETERMINABLE.
    delta: float | None
    reason: str

    @property
    def fell(self) -> bool:
        """Strictly worse, as opposed to merely not better. Reported, never used to promote."""
        return self.delta is not None and self.delta < 0

    @property
    def rose(self) -> bool:
        return self.state == ROSE

    def claim(self) -> str:
        """The strongest sentence this verdict supports."""
        if self.state == UNDETERMINABLE:
            return f"undeterminable: {self.reason}"
        a, b = self.base.eta, self.candidate.eta
        moved = "rose" if self.state == ROSE else ("fell" if self.fell else "did not move")
        return (f"eta {moved} from {a:.4f} to {b:.4f} "
                f"({self.candidate.realised_cycles}/{self.candidate.available_cycles} vs "
                f"{self.base.realised_cycles}/{self.base.available_cycles} cycles)")

    def to_dict(self) -> dict:
        return {"state": self.state, "delta": self.delta, "reason": self.reason,
                "fell": self.fell, "base": self.base.to_dict(),
                "candidate": self.candidate.to_dict()}


def compare_eta(base: EtaObservation, candidate: EtaObservation, *,
                tolerance: float = 0.0) -> EtaVerdict:
    """Rank two schedules of the SAME work by realised overlap. Three states, never two.

    ``tolerance`` is the margin by which eta must rise to count. It defaults to 0.0 because eta is a
    ratio of two integer cycle counts off a deterministic trace -- there is no noise floor to clear.
    An instrument that samples rather than counts has one, and its floor is its own to supply; it
    must never be assumed here, because a tolerance invented to make a result land is the same error
    as a defaulted composition operator.

    Returns UNDETERMINABLE, not a decision, whenever:

    * either side has no reading (unobservable vector, unread unit, undefined denominator);
    * the two sides resolved on different axes -- two instruments, not two readings;
    * the engine sets differ -- an engine named on one side only is unmeasured there, not zero;
    * both sides state a work fingerprint and the fingerprints differ -- eta is a ratio, and doing
      less work is not the same as overlapping more of it.
    """
    for side, obs in (("base", base), ("candidate", candidate)):
        if obs.eta is None:
            return EtaVerdict(UNDETERMINABLE, base, candidate, None,
                              f"the {side} run ({obs.label}) has no eta: {obs.detail}")
    if base.axis != candidate.axis:
        return EtaVerdict(UNDETERMINABLE, base, candidate, None,
                          (f"the runs resolved concurrency on different axes ({base.axis} vs "
                           f"{candidate.axis}); they are two instruments, not two readings"))
    if set(base.engines) != set(candidate.engines):
        only_b = sorted(set(base.engines) - set(candidate.engines))
        only_c = sorted(set(candidate.engines) - set(base.engines))
        return EtaVerdict(UNDETERMINABLE, base, candidate, None,
                          (f"the runs resolve different groups (only in base: {only_b}; only in "
                           f"candidate: {only_c}); a group absent from one vector is unmeasured "
                           "there, not zero, and scoring it zero reports moving work off an engine "
                           "as speeding it up"))
    if base.work is not None and candidate.work is not None and base.work != candidate.work:
        return EtaVerdict(UNDETERMINABLE, base, candidate, None,
                          (f"the runs state different work ({base.work!r} vs {candidate.work!r}); "
                           "eta is a ratio, so doing less work can raise it without any schedule "
                           "being better"))

    delta = candidate.eta - base.eta
    if delta > tolerance:
        return EtaVerdict(ROSE, base, candidate, delta,
                          (f"realised overlap rose by {delta:.4f} of the available overlap "
                           f"(tolerance {tolerance})"))
    unstated = base.work is None or candidate.work is None
    note = ("" if not unstated else
            "; neither run stated a work fingerprint, so identical work is the caller's assertion")
    return EtaVerdict(DID_NOT_RISE, base, candidate, delta,
                      (f"realised overlap did not rise ({delta:+.4f} of the available overlap, "
                       f"tolerance {tolerance}){note}"))


@dataclass(frozen=True)
class ABDecision:
    """Whether a performance candidate is accepted, and on which evidence it turned."""

    state: str
    eta: EtaVerdict
    reason: str
    #: The correctness answer the caller supplied, carried so the record shows what was checked.
    bit_exact: bool | None = None
    #: The Phase-F invariant answer (see :mod:`merlin.perf.fork`), tri-state.
    invariants_held: bool | None = None

    @property
    def accepted(self) -> bool:
        return self.state == ACCEPT

    def to_dict(self) -> dict:
        return {"state": self.state, "reason": self.reason, "bit_exact": self.bit_exact,
                "invariants_held": self.invariants_held, "eta": self.eta.to_dict()}


def ab_decision(base: EtaObservation, candidate: EtaObservation, *,
                bit_exact: bool | None,
                invariants_held: bool | None = None,
                tolerance: float = 0.0) -> ABDecision:
    """The pass condition for a performance A/B: correctness is necessary, eta is what decides.

    This is the whole point of the module. On an interlocked machine ``bit_exact`` is True for every
    candidate the hardware will run, so a gate that stops there passes everything; here it is only
    the entry condition, and the verdict is eta's.

    ``invariants_held`` is the Phase-F answer from :func:`merlin.perf.fork.check_invariants` --
    ``True`` held, ``False`` weakened, ``None`` not established. It has NO default of ``True``: a
    Phase-P candidate whose functional invariants nobody re-proved is undeterminable, and defaulting
    it to held is precisely how an unmeasured thing becomes a measured pass.

    * anything ``False`` (wrong answer, weakened invariant) -> ``REJECT``;
    * anything ``None`` (correctness unknown, invariants unchecked, eta unreadable) ->
      ``UNDETERMINABLE``, which does NOT promote;
    * bit-exact, invariants held, eta did not rise -> ``REJECT``. This is the case the archetype
      makes routine and the case the whole module exists for;
    * bit-exact, invariants held, eta rose -> ``ACCEPT``.
    """
    verdict = compare_eta(base, candidate, tolerance=tolerance)
    common = dict(eta=verdict, bit_exact=bit_exact, invariants_held=invariants_held)

    if bit_exact is False:
        return ABDecision(REJECT, reason=("the candidate did not reproduce the baseline's answer; a "
                                          "schedule that changes the result is not a schedule of the "
                                          "same program"), **common)
    if invariants_held is False:
        return ABDecision(REJECT, reason=("the candidate weakened a Phase-F functional invariant; the "
                                          "performance phase may not spend correctness it forked"),
                          **common)
    if bit_exact is None:
        return ABDecision(UNDETERMINABLE, reason=("whether the candidate reproduced the baseline's "
                                                  "answer was not established"), **common)
    if invariants_held is None:
        return ABDecision(UNDETERMINABLE, reason=("the Phase-F invariants were not checked against "
                                                  "this candidate, so nothing establishes that the "
                                                  "forked compiler is still functionally complete"),
                          **common)
    if verdict.state == UNDETERMINABLE:
        return ABDecision(UNDETERMINABLE, reason=f"eta is undeterminable: {verdict.reason}", **common)
    if verdict.state == DID_NOT_RISE:
        return ABDecision(REJECT, reason=(f"{verdict.claim()}; the reordering is correct by "
                                          "construction on an interlocked machine, so preserving the "
                                          "answer is not evidence that it bought anything"), **common)
    return ABDecision(ACCEPT, reason=verdict.claim(), **common)
