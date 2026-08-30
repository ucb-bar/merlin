"""Gap attribution: every measured cycle in a bucket, and what the gap in it implies.

A structural envelope says what a workload *cannot* beat. The measurement says what it *did*. The
difference is the gap, and a gap that is not attributed is a number nobody can act on. This module
splits the measured runtime into buckets, prices each bucket's distance from its structural bound,
and names the optimization family that distance implies -- or records UNKNOWN when the evidence does
not settle which family it is.

The bucket vocabulary is ``mlc``'s
-----------------------------------
:data:`BUCKETS` is ``compute / dma / stall / control / host``, mapped 1:1 onto the buckets
``mlc/passes/attribution.py`` already routes RTL counter events into. Mirrored rather than imported:
mlc is a separately pinned external checkout that may not be present, and a second vocabulary for
the same five time regions would be a second thing to keep in sync. :func:`buckets_match_reference`
checks the two are still identical when mlc *is* importable, so the mirror cannot drift silently.

Two invariants, both of which have already been violated by numbers in this tree
--------------------------------------------------------------------------------
**The components sum to the measured total, exactly.** Not approximately, not after dropping a
remainder. Whatever the buckets do not account for becomes the :data:`RESIDUAL` component, whose
evidence kind is ``assumed`` and which is emitted **even when it is zero**. A residual that
disappears from a report is a residual that was absorbed into a term that did not earn it. On the
corpus this module was built against the residual is a constant ``-1``: the activity buckets
partition the timeline and over-count it by exactly one cycle, a fencepost, and that is precisely
the kind of thing an "if residual: report it" would have hidden.

**A partition cannot measure overlap.** Buckets that sum to the total charge every cycle to exactly
one owner, so they report zero concurrency whether or not the hardware overlaps. Attribution
therefore never derives an overlap term from the bucket decomposition, and
:attr:`Attribution.overlap_derivable` says so on every result. The composition operator comes from
an independent observation (:func:`merlin.perf.headroom.composition_operator`) or not at all.

Which family a gap implies is DERIVED
-------------------------------------
A static bucket-to-advice table would be a lookup wearing an analysis's clothes. Each family here is
gated on evidence that is either present or not:

* a movement gap splits into *granularity* (the fixed per-transfer block, an artifact that
  amortizes away as tiles grow) and *redundancy* (bytes moved more than once, which survives
  amortization) only when a :class:`~merlin.perf.amplification.WorkloadAmplification` supplies the
  split; without one the family is UNKNOWN, not a guess between the two;
* a stall/control gap implies overlap only when a
  :class:`~merlin.perf.headroom.WorkloadHeadroom` shows a pair with a positive saving;
* a bucket whose measured cycles already equal its structural bound implies **no** family, which is
  a finding (there is nothing to win here) and not a missing one.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .decompose import UNKNOWN, ActivitySource, ResourceKind, Unavailable, _Unknown, is_unknown
from .envelope import StructuralEnvelope

__all__ = [
    "Attribution",
    "BUCKETS",
    "CorpusAttribution",
    "GapComponent",
    "OptimizationFamily",
    "RESIDUAL",
    "attribute",
    "attribute_corpus",
    "buckets_from_kinds",
    "buckets_match_reference",
]

#: The five time regions, mirroring ``mlc/passes/attribution.py``'s ``BUCKETS``. Order is theirs.
BUCKETS: tuple[str, ...] = ("compute", "dma", "stall", "control", "host")

#: The sixth component. Not a bucket -- the part no bucket accounted for. Always emitted.
RESIDUAL = "residual"

#: The two resource kinds whose bucket follows from the kind alone.
_KIND_BUCKET = {ResourceKind.COMPUTE: "compute", ResourceKind.MOVEMENT: "dma"}


class OptimizationFamily(str, Enum):
    """The family of change a gap implies. ``NONE`` is a finding; absence of a family is UNKNOWN."""

    #: Bytes moved more than once. Survives tiling; the real lever.
    TRANSFER_REDUNDANCY = "transfer_redundancy"
    #: The fixed per-transfer block padding small tiles. Amortizes away as the tile grows.
    TRANSFER_GRANULARITY = "transfer_granularity"
    #: Run two resources at the same time that currently take turns.
    OVERLAP = "overlap"
    #: The engine is busy longer than its structural bound: scheduling, operand supply, drain.
    COMPUTE_UTILIZATION = "compute_utilization"
    #: Cycles charged to no engine: issue, sequencing, command turnaround.
    ISSUE_CONTROL = "issue_control"
    #: Time on the far side of the accelerator boundary.
    HOST_BOUNDARY = "host_boundary"
    #: Measured cycles already meet the structural bound. Nothing to win in this bucket.
    NONE = "none"


def buckets_from_kinds(kinds: "Mapping[str, ResourceKind]", *, fixed_bucket: str,
                       other_bucket: "str | None" = None) -> dict[str, str]:
    """Route each resource to a bucket by its derived kind. Refuses where the kind does not decide.

    ``COMPUTE`` and ``MOVEMENT`` map to ``compute`` and ``dma``: those follow from what the kind
    *is*. ``FIXED`` does not -- cycles charged to no engine could be issue stalls (``stall``) or
    sequencer/command overhead (``control``), and occupancy alone cannot tell them apart, so
    ``fixed_bucket`` is required from the caller who knows which the target's residual bucket
    counts. ``OTHER`` likewise. This mirrors
    :func:`merlin.perf.decompose.activity_from_busy`, which raises rather than defaulting a kind:
    reading a role out of a spelling is how a register load once became "DMA".
    """
    if fixed_bucket not in BUCKETS:
        raise ValueError(f"fixed_bucket {fixed_bucket!r} is not one of {list(BUCKETS)}")
    if other_bucket is not None and other_bucket not in BUCKETS:
        raise ValueError(f"other_bucket {other_bucket!r} is not one of {list(BUCKETS)}")
    out: dict[str, str] = {}
    for name, kind in kinds.items():
        if kind in _KIND_BUCKET:
            out[name] = _KIND_BUCKET[kind]
        elif kind is ResourceKind.FIXED:
            out[name] = fixed_bucket
        elif other_bucket is not None:
            out[name] = other_bucket
        else:
            raise ValueError(
                f"resource {name!r} has kind {kind.value!r}, which does not decide a bucket. Pass "
                "other_bucket explicitly -- a resource whose role was not established must not be "
                "silently folded into one that was.")
    return out


def buckets_match_reference() -> "bool | Unavailable":
    """Whether :data:`BUCKETS` still equals mlc's, when mlc is importable.

    Returns :class:`Unavailable` -- never ``True`` -- when it is not. A check that could not run is
    "did not run", and reporting it as a pass is how a mirror drifts.
    """
    try:
        from mlc.passes.attribution import BUCKETS as REFERENCE  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 -- an external, optionally-present checkout
        return Unavailable("bucket vocabulary cross-check",
                           ("the mlc package on the import path",),
                           f"{type(exc).__name__}: {exc}")
    return tuple(REFERENCE) == BUCKETS


@dataclass(frozen=True)
class GapComponent:
    """One bucket's measured cycles, its structural floor, and what the difference implies."""

    bucket: str
    measured_cycles: int
    #: The envelope's lower bound for the resources routed here, or UNKNOWN when any is unresolved.
    structural_cycles: "float | _Unknown"
    #: ``measured - structural``. UNKNOWN propagates from the structural side.
    gap_cycles: "float | _Unknown"
    evidence_kind: str
    family: "OptimizationFamily | _Unknown"
    rationale: str
    resources: tuple[str, ...] = ()

    @property
    def gap_share(self) -> "float | _Unknown":
        if is_unknown(self.gap_cycles) or self.measured_cycles <= 0:
            return UNKNOWN
        return float(self.gap_cycles) / self.measured_cycles

    def to_dict(self) -> dict[str, Any]:
        def _s(v: Any) -> Any:
            if v is UNKNOWN:
                return "UNKNOWN"
            return v.value if isinstance(v, OptimizationFamily) else v

        return {"bucket": self.bucket, "measured_cycles": self.measured_cycles,
                "structural_cycles": _s(self.structural_cycles), "gap_cycles": _s(self.gap_cycles),
                "evidence_kind": self.evidence_kind, "family": _s(self.family),
                "rationale": self.rationale, "resources": list(self.resources)}


@dataclass(frozen=True)
class Attribution:
    """Where one workload's measured cycles went, bucket by bucket, plus the residual."""

    workload: str
    total_cycles: int
    components: tuple[GapComponent, ...]
    partitioned: "bool | None"
    provenance: str = ""

    def __post_init__(self) -> None:
        names = [c.bucket for c in self.components]
        if names != list(BUCKETS) + [RESIDUAL]:
            raise ValueError(
                f"components are {names}; every bucket plus the residual must be present in order. "
                "A bucket dropped because it was zero is a bucket a reader cannot tell from one "
                "that was never computed.")

    def component(self, bucket: str) -> GapComponent:
        for c in self.components:
            if c.bucket == bucket:
                return c
        raise KeyError(bucket)

    @property
    def residual(self) -> GapComponent:
        """Always present, and never removed when zero. It is the honesty term."""
        return self.component(RESIDUAL)

    @property
    def attributed_cycles(self) -> int:
        return sum(c.measured_cycles for c in self.components)

    @property
    def closes(self) -> bool:
        """The components sum to the measured total EXACTLY. Never approximately."""
        return self.attributed_cycles == self.total_cycles

    @property
    def overlap_derivable(self) -> bool:
        """False on a partitioned source: it reports zero overlap by construction, not by finding."""
        return not self.partitioned

    @property
    def families(self) -> dict[str, "OptimizationFamily | _Unknown"]:
        return {c.bucket: c.family for c in self.components}

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload": self.workload,
            "total_cycles": self.total_cycles,
            "attributed_cycles": self.attributed_cycles,
            "closes": self.closes,
            "partitioned": self.partitioned,
            "overlap_derivable": self.overlap_derivable,
            "components": [c.to_dict() for c in self.components],
            "provenance": self.provenance,
        }


def _movement_family(amp: Any) -> "tuple[OptimizationFamily | _Unknown, str]":
    """Split a movement gap into granularity vs redundancy, or refuse.

    Reads a :class:`~merlin.perf.amplification.WorkloadAmplification` duck-typed so this module
    does not depend on that one's construction path. Both factors must have resolved: the ratio
    alone cannot say which half of it survives tiling, and answering anyway is how someone gets sent
    to chase a win that proper tiling would have taken for free.
    """
    if amp is None:
        return UNKNOWN, ("no amplification split supplied: the moved/useful ratio alone cannot say "
                         "whether the excess is the fixed per-transfer granule (which amortizes "
                         "away as tiles grow) or genuine refetch (which does not)")
    gran = getattr(amp, "granularity_factor", UNKNOWN)
    redu = getattr(amp, "redundancy_factor", UNKNOWN)
    if is_unknown(gran) or is_unknown(redu):
        return UNKNOWN, ("the amplification split did not resolve (transfer count unknown), so the "
                         "artifact and the amortizing-resistant part cannot be told apart")
    if float(redu) >= float(gran):
        return (OptimizationFamily.TRANSFER_REDUNDANCY,
                f"redundancy {float(redu):.2f}x >= granularity {float(gran):.2f}x: the excess "
                "survives amortization, so it is refetch and not the small-tile artifact")
    return (OptimizationFamily.TRANSFER_GRANULARITY,
            f"granularity {float(gran):.2f}x > redundancy {float(redu):.2f}x: most of the excess "
            "is the fixed per-transfer block and amortizes away as the tile grows past it")


def _stall_family(head: Any) -> "tuple[OptimizationFamily | _Unknown, str]":
    if head is None:
        return UNKNOWN, ("no concurrency headroom result supplied; a partitioned activity source "
                         "cannot settle whether these cycles are overlappable")
    best = getattr(head, "best", None)
    if best is None or getattr(best, "saving_cycles", 0) <= 0:
        return (OptimizationFamily.NONE,
                "no pair of resource groups has overlappable time on this workload")
    bound = " (an upper bound until realised overlap is observed)" if getattr(
        best, "is_upper_bound", False) else ""
    return (OptimizationFamily.OVERLAP,
            f"overlapping {best.a} with {best.b} is worth up to {best.saving_cycles} cycles"
            f"{bound}")


def _family_for(bucket: str, gap: "float | _Unknown", measured: int, *, amplification: Any,
                headroom: Any) -> "tuple[OptimizationFamily | _Unknown, str]":
    if measured == 0:
        return (OptimizationFamily.NONE,
                "no cycles are charged to this bucket on this workload")
    if is_unknown(gap):
        return UNKNOWN, ("the structural bound for this bucket is UNKNOWN, so the distance to it "
                         "is not established and no family follows from it")
    if float(gap) <= 0:
        return (OptimizationFamily.NONE,
                "measured cycles already meet the structural bound for this bucket")
    if bucket == "dma":
        return _movement_family(amplification)
    if bucket == "compute":
        return (OptimizationFamily.COMPUTE_UTILIZATION,
                f"the engine is busy {float(gap):.0f} cycles beyond its structural bound: operand "
                "supply, scheduling or drain, not arithmetic")
    if bucket == "stall":
        return _stall_family(headroom)
    if bucket == "control":
        return (OptimizationFamily.ISSUE_CONTROL,
                f"{float(gap):.0f} cycles are charged to no engine: issue, sequencing or command "
                "turnaround")
    if bucket == "host":
        return (OptimizationFamily.HOST_BOUNDARY,
                f"{float(gap):.0f} cycles sit on the far side of the accelerator boundary")
    return UNKNOWN, f"no family is derivable for bucket {bucket!r}"


def attribute(source: ActivitySource, *, buckets: "Mapping[str, str]",
              envelope: "StructuralEnvelope | None" = None,
              amplification: Any = None, headroom: Any = None) -> Attribution:
    """Attribute one workload's measured cycles to buckets and price each bucket's gap.

    ``buckets`` maps every resource in ``source`` to one of :data:`BUCKETS`; an unmapped resource
    raises rather than being folded somewhere plausible. ``envelope`` supplies the structural floor
    per resource -- without it the structural side of every component is UNKNOWN, and so is every
    family, which is the correct report for a workload nobody has bounded rather than a gap of zero.
    """
    missing = sorted({r.name for r in source.resources} - set(buckets))
    if missing:
        raise ValueError(
            f"{source.workload}: no bucket declared for resource(s) {missing}. The mapping comes "
            "from the resources' derived kinds (buckets_from_kinds) or from the caller; it is "
            "never inferred from a bucket's spelling.")
    bad = sorted({b for b in buckets.values() if b not in BUCKETS})
    if bad:
        raise ValueError(f"bucket(s) {bad} are not in the vocabulary {list(BUCKETS)}")

    measured: dict[str, int] = {b: 0 for b in BUCKETS}
    members: dict[str, list[str]] = {b: [] for b in BUCKETS}
    for r in source.resources:
        measured[buckets[r.name]] += r.busy_cycles
        members[buckets[r.name]].append(r.name)

    structural: dict[str, "float | _Unknown"] = {b: 0.0 for b in BUCKETS}
    evidence: dict[str, list[str]] = {b: [] for b in BUCKETS}
    if envelope is None:
        structural = {b: UNKNOWN for b in BUCKETS}
    else:
        by_resource = {t.resource: t for t in envelope.times}
        for b in BUCKETS:
            acc: "float | _Unknown" = 0.0
            for name in members[b]:
                t = by_resource.get(name)
                if t is None or not t.known:
                    acc = UNKNOWN
                    break
                acc = float(acc) + float(t.cycles)
            structural[b] = acc
            evidence[b] = [by_resource[n].evidence_kind for n in members[b] if n in by_resource]

    components: list[GapComponent] = []
    for b in BUCKETS:
        st = structural[b]
        gap: "float | _Unknown" = UNKNOWN if is_unknown(st) else measured[b] - float(st)
        family, why = _family_for(b, gap, measured[b], amplification=amplification,
                                  headroom=headroom)
        kinds = evidence[b] or ["measured"]
        components.append(GapComponent(
            bucket=b, measured_cycles=measured[b], structural_cycles=st, gap_cycles=gap,
            evidence_kind=_weakest(kinds), family=family, rationale=why,
            resources=tuple(sorted(members[b]))))

    accounted = sum(measured.values())
    residual = source.total_cycles - accounted
    components.append(GapComponent(
        bucket=RESIDUAL, measured_cycles=residual, structural_cycles=UNKNOWN, gap_cycles=UNKNOWN,
        evidence_kind="assumed", family=UNKNOWN,
        rationale=("cycles the buckets do not account for. Emitted unconditionally, INCLUDING when "
                   "it is zero: a residual that vanishes from a report is one that was absorbed "
                   "into a term that did not earn it. A non-zero constant across a corpus is a "
                   "systematic accounting offset, not noise, and is a fact about the instrument.")))

    return Attribution(workload=source.workload, total_cycles=source.total_cycles,
                       components=tuple(components), partitioned=source.partitioned,
                       provenance=source.provenance)


def _weakest(kinds: "Sequence[str]") -> str:
    from merlin.dse_guidance.evidence import weakest_evidence

    return weakest_evidence(list(kinds))


@dataclass(frozen=True)
class CorpusAttribution:
    """Attributions over a corpus, plus what the body of them says as a whole."""

    workloads: dict[str, Attribution] = field(default_factory=dict)

    @property
    def closes(self) -> bool:
        return all(a.closes for a in self.workloads.values())

    @property
    def residual_cycles(self) -> dict[str, int]:
        return {n: a.residual.measured_cycles for n, a in self.workloads.items()}

    @property
    def residual_is_constant(self) -> "int | _Unknown":
        """The residual when every workload shares one, else UNKNOWN.

        A shared constant is a property of the instrument (a fencepost in how it charges cycles);
        a varying residual means the buckets no longer partition and the decomposition needs
        re-deriving. The two demand different work, so they get different answers.
        """
        vals = set(self.residual_cycles.values())
        if len(vals) == 1:
            return vals.pop()
        return UNKNOWN

    def bucket_cycles(self) -> dict[str, int]:
        out = {b: 0 for b in list(BUCKETS) + [RESIDUAL]}
        for a in self.workloads.values():
            for c in a.components:
                out[c.bucket] += c.measured_cycles
        return out

    def bucket_shares(self) -> dict[str, float]:
        total = sum(a.total_cycles for a in self.workloads.values())
        if not total:
            return {}
        return {b: v / total for b, v in self.bucket_cycles().items()}

    def families(self) -> dict[str, dict[str, int]]:
        """How often each family is implied per bucket, over the corpus."""
        out: dict[str, dict[str, int]] = {}
        for a in self.workloads.values():
            for c in a.components:
                key = "UNKNOWN" if is_unknown(c.family) else c.family.value
                out.setdefault(c.bucket, {})
                out[c.bucket][key] = out[c.bucket].get(key, 0) + 1
        return out


def attribute_corpus(sources: "Iterable[ActivitySource]", *, buckets: "Mapping[str, str]",
                     envelopes: "Mapping[str, StructuralEnvelope] | None" = None,
                     amplifications: "Mapping[str, Any] | None" = None,
                     headrooms: "Mapping[str, Any] | None" = None) -> CorpusAttribution:
    """:func:`attribute` over a corpus, keyed by workload."""
    out: dict[str, Attribution] = {}
    for s in sources:
        out[s.workload] = attribute(
            s, buckets=buckets,
            envelope=None if envelopes is None else envelopes.get(s.workload),
            amplification=None if amplifications is None else amplifications.get(s.workload),
            headroom=None if headrooms is None else headrooms.get(s.workload))
    return CorpusAttribution(workloads=out)
