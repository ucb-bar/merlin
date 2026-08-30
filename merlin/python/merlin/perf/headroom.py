"""Concurrency headroom: what perfect overlap of two resources would save.

Two resources that run at the same time cost ``max(T_a, T_b)``; two that take turns cost
``T_a + T_b``. The difference -- the most that overlapping them could ever save -- is exactly
``min(T_a, T_b)``. That is the whole arithmetic. What this module adds is the discipline around it:

* **Every concurrency-capable pair, not a hardcoded compute/memory pair.** The pairs come from the
  resource groups actually present in the activity source, so a target with three engines gets three
  pairs and a target with one gets none.
* **Gated on derived traits.** Overlap is only a lever where there are >= 2 engines with independent
  ports and an explicit way to observe completion. Where any of those is not established the answer
  is :class:`~merlin.perf.decompose.Unavailable` naming the one that is missing.
* **The composition operator is never defaulted.** Textbook roofline takes ``max(compute, memory)``,
  which silently assumes perfect overlap. A target that does not overlap *sums* instead, and deriving
  ``max`` where the truth is ``sum`` understates runtime in the flattering direction. So
  :func:`composition_operator` refuses to answer from a partitioned activity source -- buckets that
  partition the timeline report zero overlap by construction, whether or not overlap exists -- and
  demands an independent observation.
* **``min(a, b)`` is an upper bound until current overlap is known.** If a pair already overlaps, the
  remaining saving is smaller. Absent an observation, results carry ``is_upper_bound=True`` rather
  than being published as an achieved saving.

Grouping: absent evidence that two engines of the same kind have independent ports, they are grouped
and their busy cycles added (see :func:`~merlin.perf.decompose.busy_by_kind`). That is the
conservative direction -- it never invents a pair -- and the choice is recorded on every result in
``grouping``.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .decompose import (
    UNKNOWN,
    ActivitySource,
    ResourceKind,
    Trait,
    Unavailable,
    _Unknown,
)

__all__ = [
    "Composition",
    "ConcurrencyTraits",
    "CorpusHeadroom",
    "PairHeadroom",
    "WorkloadHeadroom",
    "composition_operator",
    "concurrency_traits",
    "corpus_headroom",
    "headroom",
    "resource_groups",
]


class Composition(str, Enum):
    """How a target's resource times compose into a runtime."""

    #: The resources take turns: ``T = sum(terms)``.
    SUM = "sum"
    #: The resources run together: ``T = max(terms)``.
    MAX = "max"
    #: Partial overlap: ``T = sum(terms) - eta * sum(min over overlapping pairs)``.
    PARTIAL = "partial"


def resource_groups(source: ActivitySource,
                    grouping: Mapping[str, str] | None = None) -> tuple[dict[str, int], dict[str, ResourceKind], str]:
    """Collapse a source's engines into the groups that may run concurrently.

    Returns ``(busy_by_group, kind_by_group, rationale)``. With no explicit ``grouping`` the default
    is one group per :class:`ResourceKind`: two engines of the same kind are assumed to contend
    (their cycles add) until port evidence says otherwise, which is the direction that cannot invent
    headroom. Pass ``grouping`` (``{resource_name: group_name}``) when the target's manifest or RTL
    establishes independent ports within a kind.
    """
    busy: dict[str, int] = {}
    kinds: dict[str, ResourceKind] = {}
    if grouping is None:
        for r in source.engines:
            key = r.kind.value
            busy[key] = busy.get(key, 0) + r.busy_cycles
            kinds[key] = r.kind
        return busy, kinds, ("by resource kind (same-kind engines assumed to contend; no port "
                             "evidence supplied)")
    unmapped = sorted(r.name for r in source.engines if r.name not in grouping)
    if unmapped:
        raise ValueError(f"{source.workload}: grouping does not cover engine(s) {unmapped}")
    for r in source.engines:
        key = grouping[r.name]
        busy[key] = busy.get(key, 0) + r.busy_cycles
        prev = kinds.get(key)
        kinds[key] = r.kind if prev in (None, r.kind) else ResourceKind.OTHER
    return busy, kinds, "caller-supplied grouping (independent ports established externally)"


@dataclass(frozen=True)
class ConcurrencyTraits:
    """The three things that have to hold before overlap is a lever at all."""

    #: How many groups of engines the activity source resolves.
    n_groups: int
    #: Do >= 2 of those groups have independent ports? Tri-state; ``None`` = not established.
    independent_ports: bool | None
    #: Can the target observe when an engine's work completed (not merely that it was busy)?
    explicit_completion: bool | None
    evidence: str = ""

    @property
    def satisfied(self) -> bool:
        return (self.n_groups >= 2 and self.independent_ports is True
                and self.explicit_completion is True)

    @property
    def missing(self) -> tuple[str, ...]:
        out: list[str] = []
        if self.n_groups < 2:
            out.append(f">=2 concurrency-capable engine groups (found {self.n_groups})")
        if self.independent_ports is not True:
            out.append("evidence that >=2 engines have independent ports")
        if self.explicit_completion is not True:
            out.append("an explicit completion signal per engine")
        return tuple(out)

    def as_traits(self) -> tuple[Trait, ...]:
        return (
            Trait("concurrency_capable_groups", self.n_groups >= 2,
                  evidence=f"{self.n_groups} engine group(s)"),
            Trait("independent_ports", self.independent_ports, evidence=self.evidence),
            Trait("explicit_completion", self.explicit_completion),
        )


def concurrency_traits(sources: Sequence[ActivitySource] = (), *,
                       manifest: Mapping[str, Any] | None = None,
                       grouping: Mapping[str, str] | None = None) -> ConcurrencyTraits:
    """Derive the concurrency traits from measured activity, with the manifest as corroboration.

    ``independent_ports`` is established when a single workload shows **two engine groups of
    different kinds** carrying work: a data-movement engine and an arithmetic engine are distinct
    units by the definition of the kinds, and the kinds themselves came from the target's own
    description of its units, so the two cannot be one issue port. Two groups of the *same* kind do
    not settle it -- they may be two views of one port -- and neither does a manifest that merely
    *declares* several units, because declaring a unit is not observing it run.

    ``explicit_completion`` is taken verbatim from the activity sources and is never defaulted: if no
    source states it, it stays ``None`` and the gate refuses.
    """
    best_groups = 0
    kinds_seen: set[ResourceKind] = set()
    for s in sources:
        busy, kinds, _ = resource_groups(s, grouping)
        # n_groups is STRUCTURAL -- a declared engine that happened to be idle on one workload is
        # still an engine. Port evidence, by contrast, needs groups that actually carried work.
        best_groups = max(best_groups, len(busy))
        kinds_seen |= {kinds[g] for g, v in busy.items() if v > 0}

    declared = len(manifest.get("compute_units") or ()) if manifest is not None else 0
    if best_groups >= 2 and len(kinds_seen) >= 2:
        ports: bool | None = True
        evidence = (f"a single workload shows {best_groups} engine groups of "
                    f"{len(kinds_seen)} distinct kinds ({sorted(k.value for k in kinds_seen)}) "
                    f"carrying work; the manifest declares {declared} compute unit(s)")
    else:
        ports = None
        evidence = (f"observed {best_groups} active engine group(s) spanning "
                    f"{len(kinds_seen)} kind(s); the manifest declares {declared} compute unit(s). "
                    f"Distinct ports were not established -- declaring a unit is not observing it.")

    completion: bool | None = None
    stated = {s.completion_observable for s in sources}
    if stated and None not in stated:
        completion = all(stated)
    return ConcurrencyTraits(n_groups=best_groups, independent_ports=ports,
                             explicit_completion=completion, evidence=evidence)


def composition_operator(sources: Sequence[ActivitySource] = (), *,
                         observed_overlap_cycles: Mapping[str, int] | None = None,
                         tolerance: float = 0.05) -> tuple[Composition, float] | Unavailable:
    """Derive how this target's resource times compose. **Never defaults to ``max``.**

    ``observed_overlap_cycles`` maps workload -> cycles in which two or more engine groups were
    busy simultaneously, from a source *independent of the activity buckets*. It is required,
    because a partitioned activity source charges every cycle to exactly one owner and so reports
    zero overlap whether or not the hardware overlaps -- using it as the evidence would derive
    ``sum`` from an artifact of the accounting, just as assuming ``max`` derives overlap from
    nothing.

    Returns ``(Composition, eta)`` where ``eta`` is the realised fraction of the available overlap
    (0 -> SUM, 1 -> MAX), or :class:`Unavailable` naming what was not established.
    """
    if not sources:
        return Unavailable("composition operator", ("at least one activity source",))
    if observed_overlap_cycles is None:
        partitioned = [s.workload for s in sources if s.partitioned]
        detail = ("the activity buckets partition the timeline, so they report zero overlap by "
                  f"construction and cannot settle this ({len(partitioned)} of {len(sources)} "
                  "workloads)") if partitioned else "no overlap observation supplied"
        return Unavailable("composition operator",
                           ("an overlap observation independent of the activity buckets",), detail)

    available = 0
    realised = 0
    for s in sources:
        if s.workload not in observed_overlap_cycles:
            return Unavailable("composition operator",
                               (f"an overlap observation for workload {s.workload!r}",),
                               "UNKNOWN propagates: one unobserved workload leaves the corpus "
                               "operator unestablished rather than partially derived")
        busy, _, _ = resource_groups(s)
        vals = sorted(busy.values(), reverse=True)
        available += vals[1] if len(vals) > 1 else 0
        realised += int(observed_overlap_cycles[s.workload])
    if available == 0:
        return Unavailable("composition operator", ("a workload where two groups are both busy",),
                           "no pair has any overlappable time, so the operator is unobservable")
    eta = realised / available
    if eta <= tolerance:
        return Composition.SUM, eta
    if eta >= 1.0 - tolerance:
        return Composition.MAX, eta
    return Composition.PARTIAL, eta


@dataclass(frozen=True)
class PairHeadroom:
    """The saving available from overlapping one pair of resource groups."""

    a: str
    b: str
    busy_a: int
    busy_b: int
    #: ``min(busy_a, busy_b)`` less any overlap already realised.
    saving_cycles: int
    saving_share: float
    #: True while current overlap for this pair is unobserved, i.e. the saving is a ceiling.
    is_upper_bound: bool


@dataclass(frozen=True)
class WorkloadHeadroom:
    """Concurrency headroom for one workload, over every concurrency-capable pair."""

    workload: str
    total_cycles: int
    pairs: tuple[PairHeadroom, ...]
    grouping: str
    #: The best pair, or ``None`` when no pair has any overlappable time.
    best: PairHeadroom | None

    @property
    def saving_cycles(self) -> int:
        return self.best.saving_cycles if self.best else 0

    @property
    def saving_share(self) -> float:
        return self.best.saving_share if self.best else 0.0

    @property
    def is_upper_bound(self) -> bool:
        return bool(self.best and self.best.is_upper_bound)


def headroom(source: ActivitySource, *,
             traits: ConcurrencyTraits | None = None,
             manifest: Mapping[str, Any] | None = None,
             grouping: Mapping[str, str] | None = None,
             observed_overlap_cycles: int | None = None) -> WorkloadHeadroom | Unavailable:
    """Overlap headroom for one workload: ``min(T_a, T_b)`` over every concurrency-capable pair.

    ``observed_overlap_cycles`` is the overlap the workload already realises. Left ``None`` the
    result is an upper bound and says so (``is_upper_bound``); it is never quietly assumed to be 0,
    because assuming zero overlap manufactures headroom exactly where a well-scheduled program has
    none.
    """
    tr = traits if traits is not None else concurrency_traits([source], manifest=manifest,
                                                              grouping=grouping)
    if not tr.satisfied:
        return Unavailable("concurrency headroom", tr.missing, tr.evidence)

    busy, _, rationale = resource_groups(source, grouping)
    names = sorted(busy)
    pairs: list[PairHeadroom] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ceiling = min(busy[a], busy[b])
            realised = 0 if observed_overlap_cycles is None else int(observed_overlap_cycles)
            saving = max(0, ceiling - realised)
            pairs.append(PairHeadroom(
                a=a, b=b, busy_a=busy[a], busy_b=busy[b], saving_cycles=saving,
                saving_share=saving / source.total_cycles if source.total_cycles else 0.0,
                is_upper_bound=observed_overlap_cycles is None))
    pairs.sort(key=lambda p: (-p.saving_cycles, p.a, p.b))
    best = pairs[0] if pairs and pairs[0].saving_cycles > 0 else None
    return WorkloadHeadroom(workload=source.workload, total_cycles=source.total_cycles,
                            pairs=tuple(pairs), grouping=rationale, best=best)


@dataclass(frozen=True)
class CorpusHeadroom:
    """Headroom summed over a corpus (or a named subset of it)."""

    workloads: dict[str, WorkloadHeadroom] = field(default_factory=dict)
    unavailable: dict[str, Unavailable] = field(default_factory=dict)

    @property
    def total_saving_cycles(self) -> int:
        return sum(w.saving_cycles for w in self.workloads.values())

    @property
    def total_cycles(self) -> int:
        return sum(w.total_cycles for w in self.workloads.values())

    @property
    def n_affected(self) -> int:
        """Workloads where some pair actually has headroom."""
        return sum(1 for w in self.workloads.values() if w.saving_cycles > 0)

    @property
    def affected_cycles(self) -> int:
        return sum(w.total_cycles for w in self.workloads.values() if w.saving_cycles > 0)

    @property
    def saving_share(self) -> float | _Unknown:
        """Total saving over the total runtime of the workloads considered."""
        total = self.total_cycles
        return self.total_saving_cycles / total if total else UNKNOWN

    @property
    def is_upper_bound(self) -> bool:
        return any(w.is_upper_bound for w in self.workloads.values())


def corpus_headroom(sources: Iterable[ActivitySource], *,
                    only: Iterable[str] | None = None,
                    traits: ConcurrencyTraits | None = None,
                    manifest: Mapping[str, Any] | None = None,
                    grouping: Mapping[str, str] | None = None,
                    observed_overlap_cycles: Mapping[str, int] | None = None) -> CorpusHeadroom:
    """:func:`headroom` over a corpus. ``only`` restricts it to a named subset.

    The subset matters: summed over *every* workload the number includes ones where overlap is not
    the lever, which flatters or deflates the figure depending on the mix. Pass the set the analysis
    is about (for instance the roles classified OPTIMIZE by :mod:`merlin.perf.workload_roles`).
    """
    keep = None if only is None else set(only)
    sources = [s for s in sources if keep is None or s.workload in keep]
    tr = traits if traits is not None else concurrency_traits(sources, manifest=manifest,
                                                              grouping=grouping)
    ok: dict[str, WorkloadHeadroom] = {}
    bad: dict[str, Unavailable] = {}
    for s in sources:
        overlap = None if observed_overlap_cycles is None else observed_overlap_cycles.get(s.workload)
        r = headroom(s, traits=tr, grouping=grouping, observed_overlap_cycles=overlap)
        if isinstance(r, Unavailable):
            bad[s.workload] = r
        else:
            ok[s.workload] = r
    return CorpusHeadroom(workloads=ok, unavailable=bad)
