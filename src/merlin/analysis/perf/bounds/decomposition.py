"""Bottleneck decomposition from any per-unit activity source.

Given how many cycles each of a target's resources was busy, and how long the workload ran, report
each resource's share of the run and which one *binds* -- the resource whose occupancy is closest to
the wall.

Gated on a derived trait, not on an archetype and not on a target: the question "which resource
binds?" is only answerable where the target exposes a **per-unit activity decomposition**. Where it
does not, the answer is :data:`UNKNOWN` with the missing trait named -- never a fabricated share and
never a zero.

Three properties of an activity source decide what may be concluded from it, so they are recorded on
the source rather than assumed:

``partitioned``
    Every cycle is charged to exactly one owner, so the buckets *sum* to the total. A partitioned
    source can never show two resources busy in the same cycle, which means **it cannot be used as
    evidence that overlap does not happen** -- it returns zero overlap by construction. Downstream
    (:mod:`merlin.analysis.perf.bounds.headroom`) needs to know this to avoid deriving a composition operator from an
    artifact of the accounting.

``completion_observable``
    The source can say when a resource's work *finished*, not merely that it was busy.

``provenance``
    Where the numbers came from. A decomposition with no provenance is not evidence.

This module also owns the small vocabulary the rest of the performance analyses share
(:class:`Resource`, :class:`ResourceKind`, :class:`ActivitySource`, :data:`UNKNOWN`,
:class:`Unavailable`) so the four analyses agree on what a resource is without importing each other's
internals.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from merlin.compiler.partition.global_plan import CycleInterval

if TYPE_CHECKING:
    from merlin.analysis.perf.bounds.headroom import Composition

# ---- decompose ---------------------------------------------------------------------------------

#: How "not known" is spelled once serialized -- the same token
#: :mod:`merlin.common.provenance` writes into pin records, so one reader recognises both.
UNKNOWN_TOKEN = "UNKNOWN"


class UnknownValueError(TypeError):
    """A value that is not known was used as if it were a number."""


class _Unknown:
    """The one inhabitant of "this is not known".

    Every numeric and truthiness protocol refuses, so ``float(UNKNOWN)`` raises instead of quietly
    becoming ``0.0``. An UNKNOWN share and a measured-zero share are different facts and this type
    exists to keep them different all the way to the report.
    """

    __slots__ = ()
    _instance: "_Unknown | None" = None

    def __new__(cls) -> "_Unknown":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return UNKNOWN_TOKEN

    __str__ = __repr__

    def __reduce__(self):
        return (_unknown_singleton, ())

    def __copy__(self) -> "_Unknown":
        return self

    def __deepcopy__(self, memo: dict) -> "_Unknown":
        return self

    def __eq__(self, other: object) -> bool:
        return other is self

    def __ne__(self, other: object) -> bool:
        return other is not self

    def __hash__(self) -> int:
        return hash(UNKNOWN_TOKEN)

    def _refuse(self, *_args: Any, **_kwargs: Any) -> Any:
        raise UnknownValueError(
            "this quantity is UNKNOWN and cannot be used as a number. It is not zero and it is not "
            "missing-so-assume-nothing-happened: it was never established. Handle it explicitly "
            "(`is UNKNOWN` / `is_unknown`) or propagate it; never write `x or 0`."
        )

    __bool__ = _refuse
    __float__ = _refuse
    __int__ = _refuse
    __index__ = _refuse
    __round__ = _refuse
    __abs__ = _refuse
    __neg__ = _refuse
    __add__ = __radd__ = _refuse
    __sub__ = __rsub__ = _refuse
    __mul__ = __rmul__ = _refuse
    __truediv__ = __rtruediv__ = _refuse
    __lt__ = __le__ = __gt__ = __ge__ = _refuse


def _unknown_singleton() -> "_Unknown":
    """Module-level factory so :class:`_Unknown` survives pickling as the same object."""
    return _Unknown()


#: The not-known value. Compare with ``is``.
UNKNOWN: _Unknown = _Unknown()


@dataclass(frozen=True)
class Unavailable:
    """A result that could not be produced, with the reason **named**.

    An analysis that cannot run returns one of these rather than a number, and rather than a bare
    sentinel: the caller (and the report) gets to see *which* trait or *which* evidence was absent,
    which is the difference between "this target does not do that" and "nobody measured it yet".
    """

    what: str
    #: The trait(s) or evidence whose absence blocked the analysis, named so a reader can go get them.
    missing: tuple[str, ...]
    detail: str = ""

    @property
    def value(self) -> _Unknown:
        return UNKNOWN

    def __str__(self) -> str:
        return f"{UNKNOWN_TOKEN}: {self.what} -- missing {', '.join(self.missing)}" + (
            f" ({self.detail})" if self.detail else ""
        )


def is_unknown(value: object) -> bool:
    """True for the UNKNOWN sentinel and for an :class:`Unavailable` result."""
    return value is UNKNOWN or isinstance(value, Unavailable)


class ResourceKind(str, Enum):
    """What a resource *is*, so analyses can reason about it without knowing its name.

    The kind is derived from the target's own description of the unit (its manifest compute-unit
    ``kind``, its ISA role, or the activity source's own labelling) and passed in. It is never
    inferred from the bucket's spelling.
    """

    #: An engine that performs arithmetic.
    COMPUTE = "compute"
    #: An engine that moves data between memories.
    MOVEMENT = "movement"
    #: Cycles charged to no engine: startup, pipeline fill and drain, issue stalls. An intercept,
    #: not a rate -- first-class, because a rate-only model mispredicts every small workload.
    FIXED = "fixed"
    #: A resource whose role was not established. Never silently folded into another kind.
    OTHER = "other"

    @property
    def is_engine(self) -> bool:
        """True when the kind names an engine that does work (as opposed to the fixed residual)."""
        return self in (ResourceKind.COMPUTE, ResourceKind.MOVEMENT, ResourceKind.OTHER)


@dataclass(frozen=True)
class Resource:
    """One resource's occupancy over a single workload."""

    name: str
    kind: ResourceKind
    busy_cycles: int

    def __post_init__(self) -> None:
        if self.busy_cycles < 0:
            raise ValueError(f"{self.name}: busy_cycles must be >= 0, got {self.busy_cycles}")


@dataclass(frozen=True)
class ActivitySource:
    """Per-unit activity for ONE workload, plus the properties that decide what it can prove."""

    workload: str
    total_cycles: int
    resources: tuple[Resource, ...]
    #: Buckets partition the timeline (they sum to the total). A partitioned source reports zero
    #: overlap *by construction* and therefore cannot be evidence about overlap either way.
    partitioned: bool | None = None
    #: The source can observe when a resource's work completed, not only that it was busy.
    completion_observable: bool | None = None
    provenance: str = ""

    def __post_init__(self) -> None:
        if self.total_cycles < 0:
            raise ValueError(f"{self.workload}: total_cycles must be >= 0")
        seen = Counter(r.name for r in self.resources)
        dupes = sorted(n for n, c in seen.items() if c > 1)
        if dupes:
            raise ValueError(f"{self.workload}: duplicate resource name(s) {dupes}")

    @property
    def engines(self) -> tuple[Resource, ...]:
        return tuple(r for r in self.resources if r.kind.is_engine)

    @property
    def fixed_cycles(self) -> int:
        return sum(r.busy_cycles for r in self.resources if r.kind is ResourceKind.FIXED)

    def busy(self, name: str) -> int:
        for r in self.resources:
            if r.name == name:
                return r.busy_cycles
        raise KeyError(f"{self.workload}: no resource named {name!r}")


def activity_from_busy(
    workload: str,
    total_cycles: int,
    busy: Mapping[str, int],
    kinds: Mapping[str, ResourceKind],
    *,
    partitioned: bool | None = None,
    completion_observable: bool | None = None,
    provenance: str = "",
) -> ActivitySource:
    """Build an :class:`ActivitySource` from a plain ``{unit: busy_cycles}`` mapping.

    ``kinds`` must name every unit. It is required rather than guessed: the bucket names belong to
    the target's own activity source, and reading a role out of a spelling is exactly the mistake
    that once mapped a local register load onto "DMA". A unit with no declared kind is a hole in the
    input, so this raises rather than defaulting to :data:`ResourceKind.OTHER`.
    """
    missing = sorted(set(busy) - set(kinds))
    if missing:
        raise ValueError(
            f"{workload}: no declared ResourceKind for unit(s) {missing}. Supply the kind from the "
            f"target's manifest/ISA roles -- it must not be inferred from the bucket's name."
        )
    resources = tuple(Resource(name=n, kind=kinds[n], busy_cycles=int(v)) for n, v in busy.items())
    return ActivitySource(
        workload=workload,
        total_cycles=int(total_cycles),
        resources=resources,
        partitioned=partitioned,
        completion_observable=completion_observable,
        provenance=provenance,
    )


@dataclass(frozen=True)
class Trait:
    """A derived yes/no property of a target, with the evidence that settled it.

    ``satisfied`` is tri-state: ``True`` / ``False`` / ``None`` for "could not be established".
    ``None`` is not ``False`` -- "this target has no per-unit decomposition" and "nobody has produced
    one yet" are different claims and only the first licenses a negative conclusion.
    """

    name: str
    satisfied: bool | None
    evidence: str = ""
    missing: tuple[str, ...] = ()


def activity_trait(
    sources: Sequence[ActivitySource] = (),
    *,
    manifest: Mapping[str, Any] | None = None,
    facts: Mapping[str, Any] | None = None,
) -> Trait:
    """Does this target expose a per-unit activity decomposition?

    Satisfied when at least one workload comes with **two or more engine buckets** and a
    provenance -- one bucket is a total, not a decomposition, and cannot name a bottleneck. The
    manifest/facts are accepted so the reason can name what the target *does* declare (how many
    compute units it has) when no activity source was supplied; they can never substitute for
    measured occupancy, because declaring a unit is not observing it.
    """
    usable = [s for s in sources if len(s.engines) >= 2 and s.total_cycles > 0]
    if usable:
        names = sorted({r.name for s in usable for r in usable[0].engines})
        provs = sorted({s.provenance for s in usable if s.provenance})
        if not provs:
            return Trait(
                "per_unit_activity_decomposition",
                None,
                evidence=f"{len(usable)} workload(s) with buckets {names}",
                missing=("provenance for the activity source",),
            )
        return Trait(
            "per_unit_activity_decomposition",
            True,
            evidence=f"{len(usable)} workload(s), engine buckets {names}, from {provs}",
        )

    declared = 0
    if manifest is not None:
        declared = len(manifest.get("compute_units") or ()) + len(manifest.get("derived_compute_units") or ())
    detail = f"the manifest declares {declared} compute unit(s)" if manifest is not None else "no manifest supplied"
    if sources:
        detail += f"; {len(sources)} activity source(s) supplied but none carries >=2 engine buckets"
    return Trait(
        "per_unit_activity_decomposition",
        None,
        evidence=detail,
        missing=("per-unit busy-cycle accounting (>=2 engine buckets for one workload)",),
    )


@dataclass(frozen=True)
class Decomposition:
    """Where one workload's cycles went, and which resource binds."""

    workload: str
    total_cycles: int
    busy: dict[str, int]
    shares: dict[str, float]
    kinds: dict[str, ResourceKind]
    #: The engine with the largest occupancy -- the one that binds.
    binding: str
    binding_share: float
    #: How far ahead the binding resource is of the runner-up, in share points. A small margin means
    #: the bottleneck is not established; two resources are effectively tied.
    margin_to_second: float
    #: Cycles charged to the FIXED kind (startup, fill, drain) as a share of the run.
    fixed_share: float
    #: total - sum(busy). Non-zero means the buckets neither partition nor overlap cleanly; it is
    #: reported rather than absorbed, because a silent residual is how a model hides its error.
    unattributed_cycles: int
    partitioned: bool | None
    provenance: str

    @property
    def shares_by_kind(self) -> dict[ResourceKind, float]:
        out: dict[ResourceKind, float] = {}
        for name, share in self.shares.items():
            out[self.kinds[name]] = out.get(self.kinds[name], 0.0) + share
        return out

    @property
    def binding_kind(self) -> ResourceKind:
        return self.kinds[self.binding]


def decompose(source: ActivitySource | None) -> Decomposition | Unavailable:
    """Decompose one workload's runtime into per-resource shares and name the binding resource.

    Returns :class:`Unavailable` (never a fabricated number) when the source is absent, has no
    engine buckets, or ran for zero cycles.
    """
    if source is None:
        return Unavailable("bottleneck decomposition", ("a per-unit activity source",), "no activity source supplied")
    trait = activity_trait([source])
    if trait.satisfied is not True and len(source.engines) < 2:
        return Unavailable(
            "bottleneck decomposition",
            trait.missing or ("per-unit busy-cycle accounting",),
            f"{source.workload}: {trait.evidence}",
        )
    if source.total_cycles <= 0:
        return Unavailable(
            "bottleneck decomposition",
            ("a non-zero total cycle count",),
            f"{source.workload}: total_cycles={source.total_cycles}",
        )

    total = source.total_cycles
    busy = {r.name: r.busy_cycles for r in source.resources}
    shares = {n: v / total for n, v in busy.items()}
    kinds = {r.name: r.kind for r in source.resources}

    engines = sorted(source.engines, key=lambda r: (-r.busy_cycles, r.name))
    binding = engines[0]
    second = engines[1].busy_cycles / total if len(engines) > 1 else 0.0
    return Decomposition(
        workload=source.workload,
        total_cycles=total,
        busy=busy,
        shares=shares,
        kinds=kinds,
        binding=binding.name,
        binding_share=binding.busy_cycles / total,
        margin_to_second=binding.busy_cycles / total - second,
        fixed_share=source.fixed_cycles / total,
        unattributed_cycles=total - sum(busy.values()),
        partitioned=source.partitioned,
        provenance=source.provenance,
    )


@dataclass(frozen=True)
class CorpusDecomposition:
    """Decompositions for a whole corpus, plus what the corpus says as a body."""

    workloads: dict[str, Decomposition] = field(default_factory=dict)
    unavailable: dict[str, Unavailable] = field(default_factory=dict)

    @property
    def binding_counts(self) -> Counter:
        """How often each resource binds. The modal binder is the corpus's regime."""
        return Counter(d.binding for d in self.workloads.values())

    @property
    def binding_kind_counts(self) -> Counter:
        return Counter(d.binding_kind for d in self.workloads.values())

    def modal_binding_kind(self) -> ResourceKind | _Unknown:
        """The kind that binds most workloads -- UNKNOWN on an empty corpus or an exact tie.

        A tie is UNKNOWN rather than an arbitrary pick: "this corpus has a regime" is a claim, and a
        50/50 split does not support it.
        """
        counts = self.binding_kind_counts
        if not counts:
            return UNKNOWN
        ranked = counts.most_common()
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return UNKNOWN
        return ranked[0][0]


def decompose_corpus(sources: Iterable[ActivitySource]) -> CorpusDecomposition:
    """:func:`decompose` over a corpus, keeping the failures visible instead of dropping them."""
    ok: dict[str, Decomposition] = {}
    bad: dict[str, Unavailable] = {}
    for s in sources:
        r = decompose(s)
        if isinstance(r, Unavailable):
            bad[s.workload] = r
        else:
            ok[s.workload] = r
    return CorpusDecomposition(workloads=ok, unavailable=bad)


def busy_by_kind(source: ActivitySource) -> dict[ResourceKind, int]:
    """Sum occupancy per kind.

    Aggregating within a kind is the conservative reading: absent port evidence, two engines of the
    same kind are assumed to contend rather than to run together, so their busy cycles add. The
    choice is recorded by :mod:`merlin.analysis.perf.bounds.headroom` in every result that depends on it.
    """
    out: dict[ResourceKind, int] = {}
    for r in source.resources:
        out[r.kind] = out.get(r.kind, 0) + r.busy_cycles
    return out


# ---- cycle_bound -------------------------------------------------------------------------------
# What a capsule may put in its one slot for a cycle ceiling, and what that slot means when it holds
# a word instead of a number.
#
# ``performance.cost.projected_cycles`` is the field a cost plane reads to find the ceiling a capsule
# declares, and such a reader needs an integer: a word resolves to no ceiling, so the assessment comes
# back incomplete and the plane can reject a physically impossible count but never a merely slow one.
#
# Measured across the corpus before this module existed: 113 of 595 capsules carried the field, all 113
# wrote ``derived_at_preflight``, none wrote an integer, and no code anywhere read that string -- no
# preflight pass resolved it and no branch matched on it. It was a promissory note nothing honoured.
# The field also had no schema type at all (the ``performance`` block reached disk only through
# ``additionalProperties: true``), so a typo in it was indistinguishable from a declaration.
#
# This module does not give any capsule a bound. It makes the ABSENCE of one a declaration rather than
# an accident, which is the precondition for a capsule ever owing a number -- and it names, in
# :data:`DERIVED_AT_ASSESSMENT`, the one word that means "a number is owed and it is derived, not
# typed", so the resolver in :mod:`merlin.analysis.perf.contract.cost_plane` can honour it without spelling it a second
# time.
#
# Target-agnostic by construction: a cycle count is a fact about one target and none is named here. The
# vocabulary below is about what KIND of thing the field holds, never about how many cycles anything
# takes.

#: The words the field may carry INSTEAD of a number, each saying why the capsule states no literal
#: count and what -- if anything -- stands in its place.
#:
#: Closed on purpose. Left open, an unrecognised word resolves to "no ceiling" downstream exactly as a
#: recognised one does, so a typo produces a capsule that quietly demands nothing -- which is the
#: failure this whole module is about, reintroduced one spelling at a time.
NO_CYCLE_BOUND: dict[str, str] = {
    "derived_at_preflight": (
        "the member states no literal count: its ceiling is DERIVED at assessment time from the "
        "array the target's own facts declare and the work its own command buffer counted, times "
        "the slack the gate declaration states -- see merlin.analysis.perf.contract.cost_plane.resolve_ceiling"
    ),
    "unbounded": (
        "this member exists to be measured, not to be held to a ceiling -- a law-fitting point has no "
        "budget it could exceed"
    ),
    "not_a_cycle_observable": (
        "this member's property is read from the instruction stream the candidate EMITTED, not from a "
        "cycle count -- so it owes no ceiling, and one derived for it would be a budget checked "
        "against a number nothing in its evidence produces"
    ),
}

#: The subset of :data:`NO_CYCLE_BOUND` whose members are a PROMISE OF A DERIVATION rather than a
#: refusal of one. A capsule spelling one of these owes a cycle count; the number simply does not live
#: in the capsule, because it is a fact about the machine the capsule is being run on and a capsule is
#: run on more than one.
#:
#: Named here rather than matched on in the reader, so the word exists in exactly one place. It was a
#: promissory note nothing honoured for 113 capsules; a second spelling of it in the resolver is how
#: it would become one again.
DERIVED_AT_ASSESSMENT: frozenset[str] = frozenset({"derived_at_preflight"})

#: The word for a member that owes NO count at all. The complement of the set above, computed rather
#: than typed so the two cannot disagree about a word added to the vocabulary.
NO_BOUND_AT_ALL: frozenset[str] = frozenset(NO_CYCLE_BOUND) - DERIVED_AT_ASSESSMENT


class CycleBoundError(ValueError):
    """The declared cycle ceiling is neither a count nor a declared reason for having none."""


def validate(value: Any, *, owner: str) -> None:
    """Raise unless ``value`` is a positive int or a declared no-bound word.

    Fail closed in both directions.

    * A NON-POSITIVE int is refused. Zero is not a ceiling any correct run could meet, and it would
      have passed the generic non-emptiness check the performance block already applies -- ``0`` is
      not ``None``, not an empty string and not an empty container -- so it is the one malformed
      value that reaches a reader looking exactly like a bound.
    * A count written as a STRING (``"4096"``) is refused for the opposite reason: it reads as a
      bound to a human reviewing the profile and resolves to no ceiling in the reader, so the capsule
      looks stricter than it is.
    * An UNRECOGNISED word is refused rather than treated as "no bound", because silently meaning the
      permissive thing is how the field got into this state.
    """
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise CycleBoundError(
            f"{owner}: performance.cost.projected_cycles must be a positive integer (a cycle "
            f"ceiling) or one of {sorted(NO_CYCLE_BOUND)}; got {value!r}"
        )
    if isinstance(value, int):
        if value <= 0:
            raise CycleBoundError(
                f"{owner}: performance.cost.projected_cycles is {value!r}. A ceiling of {value} is "
                f"not a bound any run can meet -- state a positive count, or one of "
                f"{sorted(NO_CYCLE_BOUND)} to declare that this member owes none"
            )
        return
    if value not in NO_CYCLE_BOUND:
        raise CycleBoundError(
            f"{owner}: performance.cost.projected_cycles is {value!r}, which is neither a cycle "
            f"count nor a declared reason for having none. Use a positive integer, or one of "
            f"{sorted(NO_CYCLE_BOUND)} -- a word this vocabulary does not know resolves to no "
            f"ceiling downstream, so the member would be graded as though it had claimed nothing"
        )


def declared_cycles(capsule: Any) -> int | None:
    """The integer ceiling ``capsule`` declares, or ``None`` when it declares none.

    ``None`` is returned for a declared no-bound word AND for a capsule carrying no performance block
    at all: both are "this capsule owes no cycle count", and a caller that needs to tell them apart
    should read the field itself. A malformed value RAISES rather than reading as ``None`` -- a
    capsule whose declaration nobody could parse has not declared the absence of a bound, and
    treating it as though it had is the permissive default this module exists to remove.
    """
    node: Any = capsule
    for key in ("performance", "cost", "projected_cycles"):
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    if node is None:
        return None
    validate(node, owner=str((capsule or {}).get("name", "<unnamed capsule>")))
    return int(node) if isinstance(node, int) else None


# ---- analytical_resources ----------------------------------------------------------------------
# Target-neutral, explicitly selected resource composition shared by screens.


def compose_resource_times(
    resources: Mapping[str, tuple[ResourceKind, float]],
    *,
    operator: Composition,
    eta: float,
    provenance: str,
) -> float:
    # The envelope consumes this module's resource vocabulary during import.
    # Import its composition implementation only after that vocabulary exists.
    from merlin.analysis.perf.bounds.envelope import Basis, ResourceTime, compose

    times = tuple(
        ResourceTime(name, kind, value, "cycles", Basis.MOVED, evidence_kind="calibration_fit", provenance=provenance)
        for name, (kind, value) in sorted(resources.items())
    )
    result = compose(times, operator=operator, eta=eta)
    if not result.known:
        raise ValueError("resource composition remained unresolved")
    return float(result.cycles)


def compose_resource_intervals(
    resources: Mapping[str, tuple[ResourceKind, CycleInterval]],
    *,
    operator: Composition,
    eta: float,
    provenance: str,
) -> CycleInterval:
    missing = tuple(reason for _, interval in resources.values() for reason in interval.missing)
    if missing:
        return CycleInterval.unknown(*missing)
    endpoints = []
    for endpoint in ("lo", "hi"):
        endpoints.append(
            compose_resource_times(
                {name: (kind, float(getattr(interval, endpoint))) for name, (kind, interval) in resources.items()},
                operator=operator,
                eta=eta,
                provenance=provenance,
            )
        )
    return CycleInterval(*endpoints, provenance=(provenance,))


__all__ = [
    "ActivitySource",
    "CorpusDecomposition",
    "Decomposition",
    "Resource",
    "ResourceKind",
    "Trait",
    "UNKNOWN",
    "UNKNOWN_TOKEN",
    "Unavailable",
    "UnknownValueError",
    "activity_from_busy",
    "activity_trait",
    "busy_by_kind",
    "decompose",
    "decompose_corpus",
    "is_unknown",
]
