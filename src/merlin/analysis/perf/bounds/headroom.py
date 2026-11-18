"""Concurrency headroom: what perfect overlap of two resources would save.

Two resources that run at the same time cost ``max(T_a, T_b)``; two that take turns cost
``T_a + T_b``. The difference -- the most that overlapping them could ever save -- is exactly
``min(T_a, T_b)``. That is the whole arithmetic. What this module adds is the discipline around it:

* **Every concurrency-capable pair, not a hardcoded compute/memory pair.** The pairs come from the
  resource groups actually present in the activity source, so a target with three engines gets three
  pairs and a target with one gets none.
* **Gated on derived traits.** Overlap is only a lever where there are >= 2 engines with independent
  ports and an explicit way to observe completion. Where any of those is not established the answer
  is :class:`~merlin.analysis.perf.bounds.decomposition.Unavailable` naming the one that is missing.
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
and their busy cycles added (see :func:`~merlin.analysis.perf.bounds.decomposition.busy_by_kind`). That is the
conservative direction -- it never invents a pair -- and the choice is recorded on every result in
``grouping``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from merlin.analysis.perf.bounds.decomposition import (
    UNKNOWN,
    ActivitySource,
    ResourceKind,
    Trait,
    Unavailable,
    _Unknown,
    is_unknown,
)

if TYPE_CHECKING:
    from merlin.analysis.perf.bounds.per_shape_ceiling import Machine
    from merlin.analysis.perf.bounds.schedule_proxy import Contraction

# ---- headroom ----------------------------------------------------------------------------------


class Composition(str, Enum):
    """How a target's resource times compose into a runtime."""

    #: The resources take turns: ``T = sum(terms)``.
    SUM = "sum"
    #: The resources run together: ``T = max(terms)``.
    MAX = "max"
    #: Partial overlap: ``T = sum(terms) - eta * sum(min over overlapping pairs)``.
    PARTIAL = "partial"


def resource_groups(
    source: ActivitySource, grouping: Mapping[str, str] | None = None
) -> tuple[dict[str, int], dict[str, ResourceKind], str]:
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
        return busy, kinds, ("by resource kind (same-kind engines assumed to contend; no port evidence supplied)")
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
        return self.n_groups >= 2 and self.independent_ports is True and self.explicit_completion is True

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
            Trait("concurrency_capable_groups", self.n_groups >= 2, evidence=f"{self.n_groups} engine group(s)"),
            Trait("independent_ports", self.independent_ports, evidence=self.evidence),
            Trait("explicit_completion", self.explicit_completion),
        )


def concurrency_traits(
    sources: Sequence[ActivitySource] = (),
    *,
    manifest: Mapping[str, Any] | None = None,
    grouping: Mapping[str, str] | None = None,
) -> ConcurrencyTraits:
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
        evidence = (
            f"a single workload shows {best_groups} engine groups of "
            f"{len(kinds_seen)} distinct kinds ({sorted(k.value for k in kinds_seen)}) "
            f"carrying work; the manifest declares {declared} compute unit(s)"
        )
    else:
        ports = None
        evidence = (
            f"observed {best_groups} active engine group(s) spanning "
            f"{len(kinds_seen)} kind(s); the manifest declares {declared} compute unit(s). "
            f"Distinct ports were not established -- declaring a unit is not observing it."
        )

    completion: bool | None = None
    stated = {s.completion_observable for s in sources}
    if stated and None not in stated:
        completion = all(stated)
    return ConcurrencyTraits(
        n_groups=best_groups, independent_ports=ports, explicit_completion=completion, evidence=evidence
    )


def composition_operator(
    sources: Sequence[ActivitySource] = (),
    *,
    observed_overlap_cycles: Mapping[str, int] | None = None,
    tolerance: float = 0.05,
) -> tuple[Composition, float] | Unavailable:
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
        detail = (
            (
                "the activity buckets partition the timeline, so they report zero overlap by "
                f"construction and cannot settle this ({len(partitioned)} of {len(sources)} "
                "workloads)"
            )
            if partitioned
            else "no overlap observation supplied"
        )
        return Unavailable(
            "composition operator", ("an overlap observation independent of the activity buckets",), detail
        )

    available = 0
    realised = 0
    for s in sources:
        if s.workload not in observed_overlap_cycles:
            return Unavailable(
                "composition operator",
                (f"an overlap observation for workload {s.workload!r}",),
                "UNKNOWN propagates: one unobserved workload leaves the corpus "
                "operator unestablished rather than partially derived",
            )
        busy, _, _ = resource_groups(s)
        vals = sorted(busy.values(), reverse=True)
        available += vals[1] if len(vals) > 1 else 0
        realised += int(observed_overlap_cycles[s.workload])
    if available == 0:
        return Unavailable(
            "composition operator",
            ("a workload where two groups are both busy",),
            "no pair has any overlappable time, so the operator is unobservable",
        )
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


def headroom(
    source: ActivitySource,
    *,
    traits: ConcurrencyTraits | None = None,
    manifest: Mapping[str, Any] | None = None,
    grouping: Mapping[str, str] | None = None,
    observed_overlap_cycles: int | None = None,
) -> WorkloadHeadroom | Unavailable:
    """Overlap headroom for one workload: ``min(T_a, T_b)`` over every concurrency-capable pair.

    ``observed_overlap_cycles`` is the overlap the workload already realises. Left ``None`` the
    result is an upper bound and says so (``is_upper_bound``); it is never quietly assumed to be 0,
    because assuming zero overlap manufactures headroom exactly where a well-scheduled program has
    none.
    """
    tr = traits if traits is not None else concurrency_traits([source], manifest=manifest, grouping=grouping)
    if not tr.satisfied:
        return Unavailable("concurrency headroom", tr.missing, tr.evidence)

    busy, _, rationale = resource_groups(source, grouping)
    names = sorted(busy)
    pairs: list[PairHeadroom] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ceiling = min(busy[a], busy[b])
            realised = 0 if observed_overlap_cycles is None else int(observed_overlap_cycles)
            saving = max(0, ceiling - realised)
            pairs.append(
                PairHeadroom(
                    a=a,
                    b=b,
                    busy_a=busy[a],
                    busy_b=busy[b],
                    saving_cycles=saving,
                    saving_share=saving / source.total_cycles if source.total_cycles else 0.0,
                    is_upper_bound=observed_overlap_cycles is None,
                )
            )
    pairs.sort(key=lambda p: (-p.saving_cycles, p.a, p.b))
    best = pairs[0] if pairs and pairs[0].saving_cycles > 0 else None
    return WorkloadHeadroom(
        workload=source.workload, total_cycles=source.total_cycles, pairs=tuple(pairs), grouping=rationale, best=best
    )


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


def corpus_headroom(
    sources: Iterable[ActivitySource],
    *,
    only: Iterable[str] | None = None,
    traits: ConcurrencyTraits | None = None,
    manifest: Mapping[str, Any] | None = None,
    grouping: Mapping[str, str] | None = None,
    observed_overlap_cycles: Mapping[str, int] | None = None,
) -> CorpusHeadroom:
    """:func:`headroom` over a corpus. ``only`` restricts it to a named subset.

    The subset matters: summed over *every* workload the number includes ones where overlap is not
    the lever, which flatters or deflates the figure depending on the mix. Pass the set the analysis
    is about (for instance the roles classified OPTIMIZE by :mod:`merlin.analysis.perf.workload_roles`).
    """
    keep = None if only is None else set(only)
    sources = [s for s in sources if keep is None or s.workload in keep]
    tr = traits if traits is not None else concurrency_traits(sources, manifest=manifest, grouping=grouping)
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


# ---- group_headroom ----------------------------------------------------------------------------
# Per-group HEADROOM: measured cycles against the same-machine reference, and two DERIVED bounds --
# never a suggested fix, only where a group sits relative to what the target's own facts say is
# achievable.
#
# Two bounds, both roofline-simple by design (this is a per-group DIAGNOSTIC, not the tiling-search
# achievable bound :mod:`merlin.analysis.perf.bounds.per_shape_ceiling` computes for a whole workload search):
#
# * **compute bound** -- ``macs / mesh_macs_per_cycle``, where the mesh's rate is the array geometry
#   :func:`merlin.analysis.perf.bounds.per_shape_ceiling.machine_from_facts` already derives from the target's own RTL
#   facts (never a nameplate, never a literal here).
# * **data-movement bound** -- ``moved_bytes / dram_bytes_per_cycle``, the same machine's own derived
#   DMA rate. ``moved_bytes`` is the group's OWN tensors (image, weight, result) -- never the im2col
#   matrix a convolution's unit never materializes, which would over-charge a k>1 tap ninefold (see
#   :class:`merlin.analysis.perf.bounds.schedule_proxy.Contraction`'s own note on the same trap).
#
# Every rate is read off :class:`merlin.analysis.perf.bounds.per_shape_ceiling.Machine`; a term whose fact is UNKNOWN drops
# out rather than being defaulted, and the group's bound is then reported over whatever terms resolved
# -- never silently zero, never a guess. An op this module has no MAC-bearing formula for (a residual
# add, a window mean) gets no bound at all: fabricating one would be worse than omitting it.
#
# A group's shape facts are first stated as a :class:`merlin.analysis.perf.bounds.schedule_proxy.Contraction`
# (:func:`contraction_for`), so the bound above and the RANK signal beside it price the same work.
# :func:`group_rank` is the validated schedule cost proxy's answer for that contraction -- array-tile
# transactions of compute and movement, ordered against measured hardware -- reported beside the bound
# and never folded into it: the proxy establishes an order between groups, not a cycle count.

SCHEMA = "merlin_group_headroom_v1"

#: Term names, matching :mod:`merlin.analysis.perf.bounds.per_shape_ceiling`'s vocabulary so a reader of both never sees
#: the same idea spelled two ways.
COMPUTE_TERM = "compute"
TRAFFIC_TERM = "dram_traffic"


def _dtype_bytes(dtype: Any) -> int | None:
    from merlin.analysis.perf.bounds.per_shape_ceiling import _width_bits

    bits = _width_bits(dtype)
    return None if bits is None else -(-bits // 8)  # ceil: a sub-byte width still occupies one byte


def contraction_for(op: Any, facts: Mapping[str, Any], *, label: str = "") -> Contraction | None:
    """One group's shape facts as the contraction it presents, or ``None`` when ``op`` states no
    MAC-bearing contraction this estimate covers, or a needed extent/dtype is missing.

    The footprints are the group's OWN tensors. A convolution's ``m`` is its output positions and its
    ``k`` the window times the input channels, but its ``lhs`` is the image -- never the im2col matrix
    -- and its contiguous output run is one output row, which is what separates a narrow-row stage
    from an equal-MAC one on a fixed-height array."""
    from merlin.analysis.perf.bounds.schedule_proxy import Contraction

    operand_bytes = _dtype_bytes(facts.get("operand_dtype"))
    output_bytes = _dtype_bytes(facts.get("output_dtype")) or operand_bytes
    if operand_bytes is None:
        return None
    if op == "matmul":
        try:
            m, k, n = int(facts["M"]), int(facts["K"]), int(facts["N"])
        except (KeyError, TypeError, ValueError):
            return None
        if min(m, k, n) <= 0:
            return None
        return Contraction(
            m=m,
            n=n,
            k=k,
            lhs_bytes=m * k * operand_bytes,
            rhs_bytes=k * n * operand_bytes,
            result_bytes=m * n * (output_bytes or operand_bytes),
            stream_run=m,
            label=label,
        )
    if op == "conv2d":
        try:
            co, ci = int(facts["N"]), int(facts["ci"])
            himg, wimg = int(facts["Himg"]), int(facts["Wimg"])
            kh, kw = int(facts["kh"]), int(facts["kw"])
            stride = list(facts.get("stride") or (1, 1))
            padding = list(facts.get("padding") or (0, 0, 0, 0))
            sh, sw = int(stride[0]), int(stride[1])
            pad_h, pad_w = int(padding[0]) + int(padding[2]), int(padding[1]) + int(padding[3])
        except (KeyError, TypeError, ValueError, IndexError):
            return None
        if min(co, ci, himg, wimg, kh, kw, sh, sw) <= 0:
            return None
        hout = (himg - kh + pad_h) // sh + 1
        wout = (wimg - kw + pad_w) // sw + 1
        if hout <= 0 or wout <= 0:
            return None
        # The unit's OWN tensors: the image and the weight, never the im2col matrix a native
        # convolution unit forms in its own address stream and no buffer ever holds whole.
        return Contraction(
            m=hout * wout,
            n=co,
            k=ci * kh * kw,
            lhs_bytes=himg * wimg * ci * operand_bytes,
            rhs_bytes=co * ci * kh * kw * operand_bytes,
            result_bytes=hout * wout * co * (output_bytes or operand_bytes),
            stream_run=wout,
            label=label,
        )
    return None


def machine_for(target: str) -> Machine:
    """The target's derived machine facts, without consulting the circuit for a fill/drain depth --
    this diagnostic's two terms (mesh rate, DMA rate) do not need one, and deriving it is the
    expensive half of :func:`merlin.analysis.perf.bounds.per_shape_ceiling.machine_from_facts`."""
    from merlin.analysis.perf.bounds.per_shape_ceiling import machine_from_facts

    return machine_from_facts(target, measure_fill=False)


def group_bound(macs: int, moved_bytes: int, machine: Machine) -> dict[str, Any]:
    """The two derived bounds for one group's ``(macs, moved_bytes)``, each dropped when its own
    machine fact is UNKNOWN rather than defaulted. ``bound_cycles`` is the max of what resolved (the
    binding constraint), or ``None`` when neither did."""
    rows, cols, muls = machine.array_rows, machine.array_cols, machine.muls_per_element
    mesh_macs_per_cycle = (
        int(rows) * int(cols) * int(muls) if not (is_unknown(rows) or is_unknown(cols) or is_unknown(muls)) else None
    )
    compute_bound_cycles = macs / mesh_macs_per_cycle if mesh_macs_per_cycle else None
    dram_rate = machine.dram_bytes_per_cycle
    dram_bytes_per_cycle = None if is_unknown(dram_rate) or not dram_rate else dram_rate
    traffic_bound_cycles = moved_bytes / dram_bytes_per_cycle if dram_bytes_per_cycle else None
    terms = {
        name: cycles
        for name, cycles in ((COMPUTE_TERM, compute_bound_cycles), (TRAFFIC_TERM, traffic_bound_cycles))
        if cycles is not None
    }
    return {
        "macs": macs,
        "moved_bytes": moved_bytes,
        "mesh_macs_per_cycle": mesh_macs_per_cycle,
        "dram_bytes_per_cycle": dram_bytes_per_cycle,
        "compute_bound_cycles": compute_bound_cycles,
        "traffic_bound_cycles": traffic_bound_cycles,
        "bound_cycles": max(terms.values()) if terms else None,
        "limiter": max(terms, key=terms.get) if terms else None,
    }


def group_rank(work: Contraction, machine: Machine, *, target: str) -> dict[str, Any]:
    """The schedule cost proxy's RANK signal for one group, priced on ``machine``.

    ``transactions`` orders groups by what they ask the device for and is ``None`` when the machine's
    own facts do not ground a transaction (the reasons travel in ``unresolved``). It is never a cycle
    count: the proxy is validated as an order, and its magnitude error is large."""
    from merlin.analysis.perf.bounds.schedule_proxy import schedule_cost

    cost = schedule_cost(work, target=target, machine=machine, workload=work.name)
    return {
        "transactions": round(cost.transactions, 4) if cost.resolved else None,
        "terms": {name: round(value, 4) for name, value in cost.terms.items()},
        "regime": cost.regime,
        "unresolved": {name: cost.reasons[name] for name in cost.unresolved},
        "pricing": cost.pricing.provenance,
    }


def group_headroom(
    *,
    op: Any,
    shape_facts: Mapping[str, Any] | None,
    ours_cycles: Any,
    reference_cycles: Any,
    machine: Machine,
    target: str | None = None,
) -> dict[str, Any] | None:
    """One group's full headroom document, or ``None`` when its op/shape facts admit no MAC-bearing
    estimate. ``headroom_cycles`` is ``ours - reference`` (never derived, the measurement already
    states both); ``over_bound`` is ``ours / bound_cycles`` when the bound resolved. With ``target``,
    the document also carries the group's :func:`group_rank`."""
    if not isinstance(shape_facts, Mapping) or not isinstance(ours_cycles, int):
        return None
    work = contraction_for(op, shape_facts)
    if work is None:
        return None
    bound = group_bound(work.macs, work.moved_bytes, machine)
    document: dict[str, Any] = {"schema": SCHEMA, **bound}
    if isinstance(reference_cycles, int):
        document["headroom_cycles"] = ours_cycles - reference_cycles
    if bound["bound_cycles"]:
        document["over_bound"] = round(ours_cycles / bound["bound_cycles"], 4)
    if target is not None:
        document["rank"] = group_rank(work, machine, target=target)
    return document


# ---- mesh_occupancy ----------------------------------------------------------------------------
# How many cycles a systolic array is occupied issuing a program's contractions.
#
# WHY THIS EXISTS. A scoreboard built from movement bytes and launch counts cannot see mesh
# efficiency. Measured on one whole model: changing which operand is held stationary moved the
# array's issue count by 25% -- 21,383,000 cycles to 16,050,000, with idle column-slots falling from
# 25.30% to 0.48% -- while DRAM bytes per operand role stayed BIT-IDENTICAL and the launch count did
# not change at all. Every field that scoreboard tracked was unmoved. A loop blind to this term
# treats array efficiency as free and will spend it without noticing.
#
# The count is the loop the sequencer itself runs. For a tile of `rows` streamed rows against a
# `depth x cols` stationary block, an array of `array_rows x array_cols` issues
# `rows * ceil(depth/array_rows) * ceil(cols/array_cols)` -- so a partial `cols` wastes array columns
# for the whole tile while a partial `rows` costs nothing beyond the rows themselves. That asymmetry
# is the entire reason operand orientation matters, and it is invisible in any byte or launch count.
#
# WHAT IT REFUSES TO DO. A descriptor whose geometry cannot be read is COUNTED AND REPORTED, never
# skipped. Silently ignoring an unreadable descriptor under-reports occupancy and makes the metric
# look better the less the caller understands its own program -- the failure mode where a check that
# finds nothing reports success.
#
# Target-neutral: array dimensions are parameters, derived by the caller from the target's own facts.


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-int(numerator) // int(denominator))


def tile_issue_cycles(rows: int, depth: int, cols: int, *, array_rows: int, array_cols: int) -> int:
    """Issue cycles for one tile, and the ideal it would take with no partial-block waste."""
    if min(rows, depth, cols) < 0:
        raise ValueError(f"negative extent: rows={rows} depth={depth} cols={cols}")
    if min(array_rows, array_cols) < 1:
        raise ValueError(f"array dimensions must be positive: {array_rows}x{array_cols}")
    return int(rows) * _ceil_div(depth, array_rows) * _ceil_div(cols, array_cols)


def mesh_occupancy(tiles: Iterable[Mapping[str, Any]], *, array_rows: int, array_cols: int) -> dict[str, Any]:
    """Total issue cycles over `tiles`, with the idle-slot share and unreadable-tile census.

    Each tile supplies `rows`, `depth` and `cols`. `idle_slot_share` is the fraction of occupied
    array slots doing no useful work -- the direct measure of partial-block waste, and the term an
    orientation or tiling change actually moves.
    """
    total = ideal = unreadable = readable = 0
    for tile in tiles:
        try:
            rows, depth, cols = (int(tile["rows"]), int(tile["depth"]), int(tile["cols"]))
        except (KeyError, TypeError, ValueError):
            unreadable += 1
            continue
        readable += 1
        total += tile_issue_cycles(rows, depth, cols, array_rows=array_rows, array_cols=array_cols)
        # Slots that would be occupied if every block were full: the useful work, in slot-cycles.
        ideal += rows * depth * cols
    slots = total * array_rows * array_cols
    return {
        "schema": "mesh_occupancy_v1",
        "array": {"rows": array_rows, "cols": array_cols},
        "tiles_read": readable,
        # Reported, never silently dropped: an unreadable descriptor understates occupancy, which
        # would make the metric improve as the caller's understanding of its program got worse.
        "tiles_unreadable": unreadable,
        "issue_cycles": total,
        "useful_slot_cycles": ideal,
        "idle_slot_share": round(1.0 - ideal / slots, 6) if slots else None,
        "complete": unreadable == 0,
    }


__all__ = [
    "Composition",
    "ConcurrencyTraits",
    "CorpusHeadroom",
    "PairHeadroom",
    "SCHEMA",
    "WorkloadHeadroom",
    "composition_operator",
    "concurrency_traits",
    "contraction_for",
    "corpus_headroom",
    "group_bound",
    "group_headroom",
    "group_rank",
    "headroom",
    "machine_for",
    "mesh_occupancy",
    "resource_groups",
    "tile_issue_cycles",
]
