"""The structural envelope: the lower bound a workload cannot beat, over ALL of a target's resources.

Textbook roofline is ``T = max(T_compute, T_memory)``. Two axes, and a ``max`` that silently asserts
perfect overlap. Both halves of that are assumptions, and on a target that does not overlap the
``max`` understates runtime -- in the flattering direction, which is the direction nobody audits. So
this module builds the general form::

    T_lower = compose({demand_r / peak_r}_r, fixed_terms)      compose in {max | sum | partial}

and refuses to pick ``compose`` for you. The operator arrives from
:func:`merlin.perf.headroom.composition_operator`, which derives it from an overlap observation
*independent of the activity buckets* -- and returns
:class:`~merlin.perf.decompose.Unavailable` when no such observation exists.

Four generalizations over the textbook form, each of which changed an answer here:

**(a) N resources, not two axes.** The bound ranges over every resource the activity graph names.
The result reports which one *binds* and the margin to second place -- the generalized ridge point.
The shape mirrors ``mlc``'s ``core_ipc``, which already returns a ``terms`` mapping plus a
``limiter`` naming which term is the constraint; a second vocabulary for the same idea would be a
second thing to keep in sync.

**(b) The bytes axis is MOVED bytes.** A demand declared on :attr:`Basis.ALGORITHMIC` bytes -- what
the computation needs -- is optimistic by the transfer-amplification factor, measured at 9-28x on
this corpus. Such a demand is only converted when a measured amplification factor accompanies it;
otherwise the resource's time is UNKNOWN with that stated, never silently priced at the algorithmic
figure.

**(c) Fixed terms are intercepts, first-class.** Pipeline fill and drain are paid once per drained
result, not per unit of work; a rate-only model mispredicts every small workload. A
:class:`FixedTerm` may belong to one resource (its fill, inside its busy time) or to the workload
(reset, startup -- outside any resource's occupancy, and therefore outside any overlap).

**(d) UNKNOWN propagates, and it is the COMMON case.** The RTL pipeline-depth walk resolves 43 of 84
modules on one target and 31 of 116 on the other, and the refusals are not random: they are the
*sequenced* units, which is where the time is. Both archetypes refuse their own dominant resource --
one target's mesh feeds back on 36 of 36 outputs, the other's movement engine on 21 of 21 while
movement is 60-93.7% of every cycle count. A composer that assumes derivable peaks works on one
archetype and collapses on the next. So an unresolved peak makes the composed bound UNKNOWN, and the
bound over the resolved subset is reported under a **different name**
(:attr:`Composed.partial_cycles`) -- exactly as the timing walk reports ``partial_depth`` under a
different name from ``pipeline_depth``, because the two answer different questions.

Reading a pipeline depth
------------------------
:meth:`FixedTerm.from_pipeline_depth` consumes the frozen ``facts["timing"]`` record contract:

* ``pipeline_depth`` is an ``int`` **only** when every output is acyclic. ``0`` is a REAL answer (a
  combinational module), ``None`` is UNKNOWN (a sequenced one). ``if not depth:`` conflates them and
  repeats the UNKNOWN-reads-as-0.0 bug one level up, so the check here is ``is None``.
* ``partial_depth`` is the acyclic maximum on a module whose real latency is *not* bounded by
  wiring. It is **never** a latency. Substituting it would produce a plausible, precise, wrong peak,
  which is the failure class this package exists to prevent -- so it is refused by name.
* ``None`` from ``discovered_timing`` means the RTL was not reachable (or the host's fact cache was
  never built). That is **uncached, not absent**, and it is a different :class:`Unavailable` from a
  design whose modules were walked and refused -- only the latter justifies falling back to
  measurement.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .decompose import UNKNOWN, ResourceKind, Unavailable, _Unknown, is_unknown
from .headroom import Composition

__all__ = [
    "Basis",
    "Composed",
    "FixedTerm",
    "Peak",
    "ResourceDemand",
    "ResourceTime",
    "StructuralEnvelope",
    "compose",
    "envelope",
    "resource_time",
]

#: Relative slack when checking that a derived rate ceiling is not violated by its own samples.
#: Floating-point division of exact integer counts, nothing more.
_RATE_EPS = 1e-9


class Basis(str, Enum):
    """What a movement demand counts.

    ``MOVED`` is what crossed the bus. ``ALGORITHMIC`` is what the computation needs. They differ by
    the transfer-amplification factor, and a bound built on the second is optimistic by exactly it.
    """

    MOVED = "moved"
    ALGORITHMIC = "algorithmic"


@dataclass(frozen=True)
class Peak:
    """The most work a resource can retire per cycle, and how that was established.

    ``value`` is a rate in ``unit`` per cycle, or :data:`UNKNOWN` with a ``reason``. A peak of zero
    is refused: a resource that can retire nothing is not a resource with a peak, it is a resource
    that cannot run the workload, and dividing by it manufactures an infinity.
    """

    resource: str
    value: "float | _Unknown"
    unit: str
    evidence_kind: str
    provenance: str
    reason: str = ""
    #: Observations behind the value. A rate claimed from one point cannot be told apart from a rate
    #: plus an intercept, which is the >=2-points-per-parameter rule at its smallest.
    n_samples: int = 0
    #: True when the value is a CEILING on the achieved rate rather than a nameplate peak. Every
    #: time derived from it is then a genuine lower bound on occupancy.
    is_ceiling: bool = False

    def __post_init__(self) -> None:
        if not str(self.resource).strip():
            raise ValueError("a peak must name its resource")
        if not str(self.unit).strip():
            raise ValueError(f"peak for {self.resource!r} has no unit; a bare rate is not a rate")
        reason = str(self.reason or "").strip()
        if self.value is UNKNOWN:
            if not reason:
                raise ValueError(
                    f"peak for {self.resource!r} is UNKNOWN with no reason. Recording UNKNOWN is "
                    "the honest outcome; recording it silently is not -- say what could not be "
                    "established, because 'not derivable' and 'nobody looked' need different work.")
        else:
            if reason:
                raise ValueError(f"peak for {self.resource!r} has a value and an unknown reason")
            if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
                raise TypeError(f"peak for {self.resource!r} must be a number or UNKNOWN")
            if self.value <= 0:
                raise ValueError(
                    f"peak for {self.resource!r} is {self.value}; a non-positive peak is not a "
                    "peak. Record UNKNOWN with a reason instead of a rate that divides to infinity.")
        object.__setattr__(self, "reason", reason)

    @property
    def known(self) -> bool:
        return self.value is not UNKNOWN

    @classmethod
    def unknown(cls, resource: str, unit: str, reason: str, *, provenance: str,
                evidence_kind: str = "structural_bound", n_samples: int = 0) -> "Peak":
        """A peak that could not be established, with the reason it could not."""
        return cls(resource=resource, value=UNKNOWN, unit=unit, evidence_kind=evidence_kind,
                   provenance=provenance, reason=reason, n_samples=n_samples)

    @classmethod
    def observed_ceiling(cls, resource: str, samples: "Sequence[tuple[float, float]]", *,
                         unit: str, provenance: str) -> "Peak":
        """The highest rate any observation achieved -- a CEILING, derived and then **falsified**.

        ``samples`` are ``(demand, busy_cycles)`` pairs. The rate is ``max(demand / busy)`` over the
        points that ran, which makes ``demand / rate <= busy`` hold on the point it came from. It is
        then checked against **every** sample, including the ones that were busy for zero cycles.
        A violation means the declared demand does not drive this resource's occupancy -- the demand
        model is wrong, not merely imprecise -- and the peak is UNKNOWN rather than a rate that
        would predict more cycles than the hardware spent.

        Two samples minimum: one point cannot separate a rate from a rate plus an intercept.
        """
        pts = [(float(d), float(b)) for d, b in samples]
        running = [(d, b) for d, b in pts if b > 0]
        if len(running) < 2:
            return cls.unknown(
                resource, unit,
                f"{len(running)} observation(s) with non-zero occupancy; a rate needs at least two "
                "points, because one point cannot separate a rate from a rate plus an intercept",
                provenance=provenance, evidence_kind="trace_derived", n_samples=len(running))
        if not any(d > 0 for d, _b in running):
            return cls.unknown(resource, unit,
                               "every observation declares zero demand, so no rate is observable",
                               provenance=provenance, evidence_kind="trace_derived",
                               n_samples=len(running))
        rate = max(d / b for d, b in running if d > 0)
        violations = [(d, b) for d, b in pts if d > 0 and d / rate > b * (1.0 + _RATE_EPS)]
        if violations:
            worst = max(violations, key=lambda p: (p[0] / rate) - p[1])
            return cls.unknown(
                resource, unit,
                f"the ceiling {rate:.6g} {unit}/cycle is refuted by {len(violations)} of "
                f"{len(pts)} observation(s): the worst declares {worst[0]:g} {unit} against "
                f"{worst[1]:g} busy cycle(s), so this demand does not drive the resource's "
                "occupancy and no rate over it is a bound",
                provenance=provenance, evidence_kind="trace_derived", n_samples=len(pts))
        return cls(resource=resource, value=rate, unit=unit, evidence_kind="trace_derived",
                   provenance=provenance, n_samples=len(pts), is_ceiling=True)


@dataclass(frozen=True)
class FixedTerm:
    """An intercept: cycles paid once, not per unit of work.

    ``resource`` names the engine that pays it (a pipeline fill is inside that engine's busy time
    and therefore overlaps with other engines exactly as its engine does). An empty ``resource`` is a
    whole-workload intercept -- reset, startup, teardown -- which sits outside every engine's
    occupancy and so is added after composition rather than being subject to overlap.
    """

    name: str
    cycles: int
    resource: str = ""
    law: str = ""
    provenance: str = ""
    evidence_kind: str = "structural_bound"

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("a fixed term must be named")
        if isinstance(self.cycles, bool) or not isinstance(self.cycles, int):
            raise TypeError(f"fixed term {self.name!r} cycles must be an int (0 is a real answer)")
        if self.cycles < 0:
            raise ValueError(f"fixed term {self.name!r} cycles must be >= 0, got {self.cycles}")

    @classmethod
    def from_pipeline_depth(cls, record: "Mapping[str, Any] | None", *, name: str,
                            resource: str = "") -> "FixedTerm | Unavailable":
        """A module's structural pipeline depth as an intercept, or the reason there is none.

        Consumes one record of the frozen ``facts["timing"]`` contract. Three refusals, each
        distinct, because they license different follow-up:

        * ``record is None`` -- the RTL was not reachable or this host's fact cache was never
          built. **Uncached is not absent**, and it does not license a measurement fallback: nobody
          looked yet.
        * ``pipeline_depth is None`` with a ``partial_depth`` -- the module is SEQUENCED. The
          acyclic maximum is reported, and it is refused here by name: it is not this module's
          latency, and substituting it yields a precise wrong answer.
        * ``pipeline_depth is None`` with no ``partial_depth`` -- every output is reached through
          feedback. This is the refusal that *does* license falling back to measurement.

        A resolved depth of ``0`` is a combinational module and a real answer; it is returned as a
        zero-cycle intercept, which is a fact, not a missing one.
        """
        if record is None:
            return Unavailable(
                f"pipeline fill for {name}",
                ("a timing record for this module",),
                "the RTL was not reachable and no timing facts are cached on this host: UNCACHED "
                "is not ABSENT, and a design nobody walked is not a design with no sequenced logic")
        if "pipeline_depth" not in record:
            raise ValueError(
                f"timing record for {name!r} carries no 'pipeline_depth' key; the fact contract "
                "always sets it (to an int or to None) and a record without it is malformed")
        depth = record["pipeline_depth"]
        if depth is None:
            partial = record.get("partial_depth")
            evidence = str(record.get("evidence") or "")
            if partial is not None:
                return Unavailable(
                    f"pipeline fill for {name}",
                    ("a finite wiring depth for a sequenced module",),
                    f"pipeline_depth is UNKNOWN and partial_depth is {partial}, which is the "
                    "maximum over the ACYCLIC outputs only and is NOT this module's latency; "
                    f"substituting it would be precise and wrong. {evidence}")
            return Unavailable(
                f"pipeline fill for {name}",
                ("the sequencer's own limits, or a measurement",),
                f"every output is reached through feedback, so no finite wiring depth is this "
                f"module's latency. {evidence}")
        if isinstance(depth, bool) or not isinstance(depth, int):
            raise TypeError(f"timing record for {name!r} has a non-int pipeline_depth {depth!r}")
        return cls(name=name, cycles=int(depth), resource=resource, law="rtl_pipeline_depth",
                   provenance=str(record.get("evidence") or record.get("source") or ""))

    @classmethod
    def from_fill_law(cls, law: str, dimension: int, *, name: str,
                      resource: str = "") -> "FixedTerm":
        """A pipeline fill from a named structural law over the unit's own dimension.

        Delegates to :func:`merlin.perf.record.fill_cycles` so the laws live in one place; that
        function fails closed on a law it does not implement rather than guessing a fill.
        """
        from .record import fill_cycles

        return cls(name=name, cycles=fill_cycles(law, dimension), resource=resource, law=law,
                   provenance=f"{law}(dimension={dimension})")


@dataclass(frozen=True)
class ResourceDemand:
    """How much work one resource is asked to do, in the resource's own unit."""

    resource: str
    kind: ResourceKind
    amount: float
    unit: str
    basis: Basis = Basis.MOVED
    #: Measured moved/useful ratio. REQUIRED to price an ``ALGORITHMIC`` demand, forbidden on a
    #: ``MOVED`` one (the moved figure has already paid it).
    amplification: "float | _Unknown | None" = None
    provenance: str = ""

    def __post_init__(self) -> None:
        if not str(self.resource).strip():
            raise ValueError("a demand must name its resource")
        if not str(self.unit).strip():
            raise ValueError(f"demand for {self.resource!r} has no unit")
        if isinstance(self.amount, bool) or not isinstance(self.amount, (int, float)):
            raise TypeError(f"demand for {self.resource!r} must be a number")
        if self.amount < 0:
            raise ValueError(f"demand for {self.resource!r} must be >= 0, got {self.amount}")
        if self.basis is Basis.MOVED and self.amplification is not None:
            raise ValueError(
                f"demand for {self.resource!r} is already on the MOVED basis, so an amplification "
                "factor would apply it twice")


@dataclass(frozen=True)
class ResourceTime:
    """How long one resource needs, or the reason that is not established."""

    resource: str
    kind: ResourceKind
    cycles: "float | _Unknown"
    unit: str
    basis: Basis
    #: Intercept cycles included in ``cycles`` (this resource's own fill/drain).
    fixed_cycles: int = 0
    evidence_kind: str = "structural_bound"
    provenance: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if self.cycles is UNKNOWN and not str(self.reason or "").strip():
            raise ValueError(f"time for {self.resource!r} is UNKNOWN with no reason")
        if self.cycles is not UNKNOWN and self.cycles < 0:
            raise ValueError(f"time for {self.resource!r} must be >= 0")

    @property
    def known(self) -> bool:
        return self.cycles is not UNKNOWN


def resource_time(demand: ResourceDemand, peak: Peak,
                  fixed: "Sequence[FixedTerm]" = ()) -> ResourceTime:
    """``demand / peak`` plus this resource's own intercepts. UNKNOWN propagates from either input.

    The amplification gate is generalization (b): an ``ALGORITHMIC`` demand priced without a
    measured amplification factor is a bound that is optimistic by that factor, so it is refused
    rather than published.
    """
    if demand.resource != peak.resource:
        raise ValueError(f"demand names resource {demand.resource!r} but the peak names "
                         f"{peak.resource!r}; a rate for one unit cannot price another")
    own = [f for f in fixed if f.resource == demand.resource]
    fixed_cycles = sum(f.cycles for f in own)
    prov = "; ".join(p for p in ([demand.provenance, peak.provenance]
                                 + [f.provenance for f in own]) if p)

    if demand.amount == 0:
        # A resource the program never asks to do anything takes zero cycles at ANY positive rate,
        # so this resolves even where the peak does not -- and it pays no fill, because a pipeline
        # fill is charged per drained result and there are none. Derived, not defaulted.
        return ResourceTime(
            resource=demand.resource, kind=demand.kind, cycles=0.0, unit=demand.unit,
            basis=demand.basis, fixed_cycles=0, evidence_kind=peak.evidence_kind,
            provenance=prov)

    amount: "float | _Unknown" = demand.amount
    if demand.basis is Basis.ALGORITHMIC:
        amp = demand.amplification
        if amp is None or is_unknown(amp):
            return ResourceTime(
                resource=demand.resource, kind=demand.kind, cycles=UNKNOWN, unit=demand.unit,
                basis=demand.basis, fixed_cycles=fixed_cycles, evidence_kind=peak.evidence_kind,
                provenance=prov,
                reason="the demand counts the bytes the ALGORITHM needs, and no measured transfer "
                       "amplification accompanies it. A bound built on algorithmic rather than "
                       "moved bytes is optimistic by exactly that factor (9-28x on this corpus), "
                       "which is the flattering direction")
        amount = demand.amount * float(amp)

    if not peak.known:
        return ResourceTime(
            resource=demand.resource, kind=demand.kind, cycles=UNKNOWN, unit=demand.unit,
            basis=demand.basis, fixed_cycles=fixed_cycles, evidence_kind=peak.evidence_kind,
            provenance=prov, reason=f"the peak for {demand.resource!r} is UNKNOWN: {peak.reason}")

    return ResourceTime(
        resource=demand.resource, kind=demand.kind,
        cycles=float(amount) / float(peak.value) + fixed_cycles, unit=demand.unit,
        basis=demand.basis, fixed_cycles=fixed_cycles, evidence_kind=peak.evidence_kind,
        provenance=prov)


@dataclass(frozen=True)
class Composed:
    """The composed lower bound, and what was left out of it."""

    #: The bound over ALL resources, or UNKNOWN when any resource's time is not established.
    cycles: "float | _Unknown"
    #: The bound over the RESOLVED subset. A different name from :attr:`cycles` on purpose: it
    #: answers a different question (it is a bound, but a weaker one), exactly as the RTL timing
    #: walk keeps ``partial_depth`` apart from ``pipeline_depth``. Adding a resource can only
    #: raise the composed time, so this is always a valid -- if looser -- lower bound.
    partial_cycles: float
    #: The busiest single resource plus the workload intercepts. No composition may fall below it:
    #: even perfect overlap cannot finish before the slowest resource does.
    floor_cycles: float
    operator: Composition
    eta: float
    #: Cycles the operator credited to overlap (``sum`` of the terms less the composed value).
    overlap_saving: float
    unresolved: tuple[str, ...]
    workload_fixed_cycles: int
    #: Cycles from resolved resources of the FIXED kind. Charged to no engine, therefore serial:
    #: they are added after the operator rather than being offered to it as overlappable time.
    serial_fixed_cycles: float = 0.0
    #: True when the composed value was raised to :attr:`floor_cycles`, which happens when a
    #: pairwise overlap credit over three or more resources would have over-counted the saving.
    clamped_to_floor: bool = False

    @property
    def known(self) -> bool:
        return self.cycles is not UNKNOWN


def _apply(values: "Sequence[float]", operator: Composition, eta: float) -> float:
    if not values:
        return 0.0
    if operator is Composition.SUM:
        return float(sum(values))
    if operator is Composition.MAX:
        return float(max(values))
    if operator is Composition.PARTIAL:
        pairs = sum(min(values[i], values[j])
                    for i in range(len(values)) for j in range(i + 1, len(values)))
        return float(sum(values)) - eta * pairs
    raise ValueError(f"unknown composition operator {operator!r}")


def compose(times: "Sequence[ResourceTime]", *, operator: Composition, eta: float,
            workload_fixed: "Sequence[FixedTerm]" = ()) -> Composed:
    """Compose per-resource times into one lower bound under a **supplied** operator.

    There is no default. ``operator`` and ``eta`` come from
    :func:`merlin.perf.headroom.composition_operator`, which derives them from an overlap
    observation independent of the activity buckets and refuses when there is none. Passing ``max``
    because it is the textbook form is the specific error this signature exists to make visible.

    Only resources of an ENGINE kind are offered to the operator. A resource of the FIXED kind
    counts cycles charged to no engine -- reset, sequencing, issue -- which by definition are not
    concurrent with an engine, so they are added after composition and never credited as overlap.
    That is generalization (c) applied to a resource rather than to a term.

    ``PARTIAL`` credits ``eta`` of every overlappable pair. Over three or more resources that
    pairwise credit can exceed the saving perfect overlap would actually give, so the result is
    floored at the busiest single resource -- which is a structural statement (nothing finishes
    before its slowest resource), not a fudge, and the clamp is recorded.
    """
    if not isinstance(operator, Composition):
        raise TypeError(f"operator must be a Composition, got {operator!r}; the composition "
                        "operator is derived and passed, never defaulted")
    eta = float(eta)
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta {eta} is outside [0, 1]; it is a realised fraction of the available "
                         "overlap, not a scale factor")
    stray = [f.name for f in workload_fixed if f.resource]
    if stray:
        raise ValueError(f"workload_fixed carries resource-owned term(s) {stray}; a fill that "
                         "belongs to an engine is inside that engine's time and overlaps with it")

    resolved = [t for t in times if t.known]
    unresolved = tuple(t.resource for t in times if not t.known)
    values = [float(t.cycles) for t in resolved if t.kind.is_engine]
    serial = float(sum(float(t.cycles) for t in resolved if not t.kind.is_engine))
    fixed_cycles = sum(f.cycles for f in workload_fixed)
    pedestal = serial + fixed_cycles

    base = _apply(values, operator, eta)
    floor = (max(values) if values else 0.0) + pedestal
    total = base + pedestal
    clamped = total < floor
    if clamped:
        total = floor
    return Composed(
        cycles=(UNKNOWN if unresolved else total), partial_cycles=total, floor_cycles=floor,
        operator=operator, eta=eta, overlap_saving=float(sum(values)) + pedestal - total,
        unresolved=unresolved, workload_fixed_cycles=fixed_cycles, clamped_to_floor=clamped,
        serial_fixed_cycles=serial)


@dataclass(frozen=True)
class StructuralEnvelope:
    """One workload's structural envelope: the bound, the binding resource, and the margin.

    ``terms`` / ``limiter`` mirror the shape ``mlc``'s ``core_ipc`` already returns for the same
    question -- a mapping of the candidate constraints plus the name of the one that binds -- so the
    two read the same way. The margin to second place is the generalized ridge point: a small margin
    means the bottleneck is not established and two resources are effectively tied, which is a
    different report from a clear binder even though both name one resource.
    """

    workload: str
    times: tuple[ResourceTime, ...]
    fixed: tuple[FixedTerm, ...]
    composed: Composed
    #: ``{resource: cycles}`` over the resolved resources only.
    terms: dict[str, float]
    #: The resource whose time binds, or UNKNOWN when nothing resolved.
    limiter: "str | _Unknown"
    limiter_cycles: "float | _Unknown"
    #: Cycles between the binder and the runner-up. UNKNOWN when fewer than two resolved.
    margin_to_second: "float | _Unknown"
    margin_share: "float | _Unknown"
    #: True while some resource's time is UNKNOWN: an unresolved resource could bind instead, so
    #: the named limiter is the binder *of what resolved*, not established as the binder.
    limiter_is_provisional: bool

    @property
    def lower_bound_cycles(self) -> "float | _Unknown":
        return self.composed.cycles

    @property
    def partial_lower_bound_cycles(self) -> float:
        return self.composed.partial_cycles

    @property
    def unresolved(self) -> tuple[str, ...]:
        return self.composed.unresolved

    def to_dict(self) -> dict[str, Any]:
        def _s(v: Any) -> Any:
            return "UNKNOWN" if v is UNKNOWN else v

        return {
            "workload": self.workload,
            "lower_bound_cycles": _s(self.composed.cycles),
            "partial_lower_bound_cycles": self.composed.partial_cycles,
            "floor_cycles": self.composed.floor_cycles,
            "operator": self.composed.operator.value,
            "eta": self.composed.eta,
            "terms": dict(self.terms),
            "limiter": _s(self.limiter),
            "limiter_cycles": _s(self.limiter_cycles),
            "margin_to_second": _s(self.margin_to_second),
            "margin_share": _s(self.margin_share),
            "limiter_is_provisional": self.limiter_is_provisional,
            "unresolved": list(self.composed.unresolved),
            "unresolved_reasons": {t.resource: t.reason for t in self.times if not t.known},
            "workload_fixed_cycles": self.composed.workload_fixed_cycles,
        }


def envelope(workload: str, times: "Sequence[ResourceTime]", *, operator: Composition, eta: float,
             fixed: "Sequence[FixedTerm]" = ()) -> StructuralEnvelope:
    """Build the envelope for one workload from its per-resource times and the derived operator."""
    workload_fixed = tuple(f for f in fixed if not f.resource)
    composed = compose(times, operator=operator, eta=eta, workload_fixed=workload_fixed)

    resolved = sorted(((t.resource, float(t.cycles)) for t in times if t.known),
                      key=lambda p: (-p[1], p[0]))
    terms = {n: c for n, c in resolved}
    if resolved:
        limiter, limiter_cycles = resolved[0]
        if len(resolved) > 1:
            margin: "float | _Unknown" = limiter_cycles - resolved[1][1]
            share: "float | _Unknown" = (margin / limiter_cycles) if limiter_cycles > 0 else 0.0
        else:
            margin = UNKNOWN
            share = UNKNOWN
    else:
        limiter = UNKNOWN
        limiter_cycles = UNKNOWN
        margin = UNKNOWN
        share = UNKNOWN

    return StructuralEnvelope(
        workload=workload, times=tuple(times), fixed=tuple(fixed), composed=composed, terms=terms,
        limiter=limiter, limiter_cycles=limiter_cycles, margin_to_second=margin, margin_share=share,
        limiter_is_provisional=bool(composed.unresolved))
