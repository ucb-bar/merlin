"""The generalized composer: build every workload's envelope, attribute its gap, and say how much
of the answer is actually established.

:mod:`merlin.perf.envelope` holds the arithmetic of one bound and :mod:`merlin.perf.attribution`
holds the arithmetic of one split. This module is what runs them over a corpus: it derives the peaks
that *can* be derived, records the ones that cannot, composes under an operator that arrives from
evidence rather than from habit, and reports coverage in the only unit that means anything --
**measured time**.

Coverage is time-weighted, and that is not a refinement
------------------------------------------------------
The obvious coverage metric is the fraction of RTL modules whose pipeline depth resolved: 43 of 84
on one target, 31 of 116 on the other. Reported as confidence it is worse than useless, because the
walk resolves the cheap combinational leaves -- lane boxes, adders, muxes, rounders -- and refuses
the sequenced units, which is exactly where the time is. Both archetypes refuse their own dominant
resource: one target's mesh feeds back on 36 of 36 outputs; the other's movement engine refuses on
21 of 21 while movement is 60-93.7% of every cycle count. So a module-count share of 51% can sit
next to a time-weighted share of a few percent.

:class:`Coverage` therefore carries both, under names that cannot be confused
(``module_count_share`` vs ``time_weighted_resolved_share``), and deliberately exposes no field
called ``confidence``. The module count is a count of modules. It is never the confidence in a
prediction, and the only defensible weighting is each resource's share of the measured cycles.

What this module refuses to do
------------------------------
* **Pick a composition operator.** :func:`compose_corpus` takes one. Getting it from
  :func:`merlin.perf.headroom.composition_operator` returns :class:`Unavailable` on a partitioned
  activity source, and that refusal is the correct outcome, not an obstacle to route around.
* **Fit the point that disagrees.** Where a structural law leaves its validated regime the
  resource's time is UNKNOWN and the workload's bound is partial. Two points that show a law is
  wrong do not show what is right, and a correction fitted to one of them is indistinguishable from
  the law having been right all along.
* **Report a prediction that beats the measurement as a success.** Every prediction here is a lower
  bound; one that exceeds the measured cycles falsifies an input, and
  :attr:`CorpusPrediction.bound_violations` surfaces it instead of the mean absolute error hiding
  it.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .attribution import Attribution, CorpusAttribution, attribute_corpus
from .decompose import UNKNOWN, ActivitySource, ResourceKind, Unavailable, _Unknown, is_unknown
from .envelope import (
    Basis,
    Composed,
    FixedTerm,
    Peak,
    ResourceDemand,
    ResourceTime,
    StructuralEnvelope,
    compose,
    envelope,
    resource_time,
)
from .headroom import Composition

__all__ = [
    "Coverage",
    "CorpusPrediction",
    "OperatorSensitivity",
    "Prediction",
    "ResourceDemand",
    "resource_time",
    "compose_corpus",
    "coverage",
    "emit_envelope_report",
    "fixed_terms_from_timing",
    "operator_sensitivity",
    "peaks_from_observations",
    "structural_unit_time",
]


# ------------------------------------------------------------------------------------------------
# deriving the inputs
# ------------------------------------------------------------------------------------------------


def peaks_from_observations(demands: "Mapping[str, Mapping[str, ResourceDemand]]",
                            sources: "Iterable[ActivitySource]", *,
                            units: "Mapping[str, str]",
                            provenance: str) -> dict[str, Peak]:
    """One :class:`Peak` per resource, as the highest rate the corpus ever achieved.

    A ceiling on the achieved rate, not a nameplate peak -- which is the useful direction: every
    time derived from it is then a genuine lower bound on that resource's occupancy, on every
    workload, by construction rather than by luck.

    :meth:`Peak.observed_ceiling` falsifies the ceiling against every observation before returning
    it, so a resource whose declared demand does not actually drive its occupancy comes back
    UNKNOWN. That refusal is the point: it is how a plausible demand proxy gets caught being wrong
    instead of quietly predicting more cycles than the hardware spent.
    """
    by_resource: dict[str, list[tuple[float, float]]] = {}
    for s in sources:
        per = demands.get(s.workload) or {}
        for r in s.resources:
            d = per.get(r.name)
            if d is None:
                continue
            by_resource.setdefault(r.name, []).append((float(d.amount), float(r.busy_cycles)))
    out: dict[str, Peak] = {}
    for name, samples in by_resource.items():
        unit = units.get(name)
        if unit is None:
            raise ValueError(f"no demand unit declared for resource {name!r}; a rate with no unit "
                             "is not a rate")
        out[name] = Peak.observed_ceiling(name, samples, unit=unit, provenance=provenance)
    return out


def fixed_terms_from_timing(records: "Sequence[Mapping[str, Any]] | None",
                            modules: "Mapping[str, str]") -> tuple[dict[str, FixedTerm],
                                                                   dict[str, Unavailable]]:
    """Per-resource pipeline-fill intercepts from the RTL timing walk, and the refusals.

    ``records`` is ``facts["timing"]``; ``modules`` maps each resource to the RTL module that
    implements it (a target-specific fact, supplied by the caller, never guessed from a name).
    ``records is None`` means the RTL was not reachable or the fact cache was never built on this
    host -- **uncached, not absent** -- and every resource gets that refusal rather than a fill of
    zero.
    """
    index: "dict[str, Mapping[str, Any]] | None" = None
    if records is not None:
        index = {str(r["module"]): r for r in records}
    ok: dict[str, FixedTerm] = {}
    bad: dict[str, Unavailable] = {}
    for resource, module in modules.items():
        record = None if index is None else index.get(module)
        if index is not None and record is None:
            bad[resource] = Unavailable(
                f"pipeline fill for {resource}", (f"a timing record for module {module!r}",),
                f"the walk covered {len(index)} module(s) and {module!r} was not among them")
            continue
        got = FixedTerm.from_pipeline_depth(record, name=f"{resource}_fill", resource=resource)
        if isinstance(got, Unavailable):
            bad[resource] = got
        else:
            ok[resource] = got
    return ok, bad


def structural_unit_time(resource: str, kind: ResourceKind, composed_busy: Any, *,
                         provenance: str) -> ResourceTime:
    """Lift a :class:`merlin.perf.record.ComposedBusy` into a :class:`ResourceTime`.

    A unit whose cost is an intercept plus the delays its own program schedules has no useful
    ``demand / peak`` form -- the schedule *is* the demand -- so its time is composed structurally
    and enters the envelope alongside the rate-derived ones. When the program leaves the law's
    validated regime the composed value is ``None`` and this propagates UNKNOWN with the law's own
    reason, keeping the lower bound it *did* establish visible as the resource's floor.
    """
    if composed_busy.cycles is None:
        return ResourceTime(
            resource=resource, kind=kind, cycles=UNKNOWN, unit="cycles", basis=Basis.MOVED,
            fixed_cycles=int(composed_busy.lower_bound), evidence_kind="structural_bound",
            provenance=provenance, reason=str(composed_busy.reason))
    return ResourceTime(
        resource=resource, kind=kind, cycles=float(composed_busy.cycles), unit="cycles",
        basis=Basis.MOVED, fixed_cycles=int(composed_busy.lower_bound),
        evidence_kind="structural_bound", provenance=provenance)


# ------------------------------------------------------------------------------------------------
# coverage -- weighted by TIME, never by module count
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Coverage:
    """How much of the answer is established, in three units that must not be confused.

    ``module_count_share`` counts modules. ``structurally_resolved_time_share`` weights the same
    question by measured cycles, and is the one that belongs next to it. ``time_weighted_resolved_
    share`` is broader still: the runtime bounded by ANY evidence, measurement included.

    There is deliberately no ``confidence`` field. A module count is a count of modules; weighting
    it into a confidence would report "half the design is understood" while the unresolved half
    carries nearly all the time, which is the specific mistake this class exists to prevent.
    """

    #: Modules whose ``pipeline_depth`` resolved, out of those walked. A COUNT OF MODULES.
    resolved_module_count: int
    walked_module_count: int
    #: Resources whose time the envelope resolved, out of those in the activity graph.
    resolved_resource_count: int
    resource_count: int
    #: Share of MEASURED cycles belonging to resources whose time the envelope resolved -- by ANY
    #: evidence, structural or measured. The load-bearing number for "how much of the runtime is
    #: bounded at all".
    time_weighted_resolved_share: "float | _Unknown"
    #: Share of MEASURED cycles belonging to resources whose peak is STRUCTURALLY derivable from the
    #: RTL. This is the one to compare against :attr:`module_count_share`, and the one that shows
    #: why a module count must never stand in for confidence: the depth walk resolves combinational
    #: leaves and refuses sequenced units, so a majority of modules can sit next to a few percent of
    #: the time. UNKNOWN -- never 0.0 -- when nobody supplied which resources resolved structurally.
    structurally_resolved_time_share: "float | _Unknown" = UNKNOWN
    #: Share of measured cycles belonging to resources that did not resolve, by resource.
    unresolved_time_share: dict[str, float] = field(default_factory=dict)
    unresolved_reasons: dict[str, str] = field(default_factory=dict)
    note: str = ""

    @property
    def module_count_share(self) -> "float | _Unknown":
        """Resolved modules over walked modules. **A module count, not a confidence.**

        The walk resolves combinational leaves and refuses sequenced units, so this share is biased
        away from where the time goes -- and on both archetypes measured here the refusal includes
        the dominant resource. Compare it with
        :attr:`time_weighted_resolved_share`; where they disagree, the time-weighted one is the
        answer to "how much of the runtime is bounded".
        """
        if self.walked_module_count <= 0:
            return UNKNOWN
        return self.resolved_module_count / self.walked_module_count

    def to_dict(self) -> dict[str, Any]:
        def _s(v: Any) -> Any:
            return "UNKNOWN" if v is UNKNOWN else v

        return {
            "resolved_module_count": self.resolved_module_count,
            "walked_module_count": self.walked_module_count,
            "module_count_share": _s(self.module_count_share),
            "module_count_share_note": (
                "A COUNT OF MODULES. Not a confidence and not a coverage of runtime: the depth "
                "walk resolves combinational leaves and refuses sequenced units, so it is biased "
                "away from where the cycles are."),
            "resolved_resource_count": self.resolved_resource_count,
            "resource_count": self.resource_count,
            "time_weighted_resolved_share": _s(self.time_weighted_resolved_share),
            "structurally_resolved_time_share": _s(self.structurally_resolved_time_share),
            "unresolved_time_share": dict(self.unresolved_time_share),
            "unresolved_reasons": dict(self.unresolved_reasons),
            "note": self.note,
        }


def coverage(sources: "Sequence[ActivitySource]",
             envelopes: "Mapping[str, StructuralEnvelope]", *,
             timing_records: "Sequence[Mapping[str, Any]] | None" = None,
             structural_resources: "Iterable[str] | None" = None) -> Coverage:
    """Coverage of a corpus, weighted by the measured cycles each resource actually carries.

    ``structural_resources`` names the resources whose peak or fill the RTL walk resolved (the keys
    of :func:`fixed_terms_from_timing`'s first return value). Supplying it produces the number that
    belongs next to the module count; omitting it leaves that share UNKNOWN rather than zero,
    because "nobody said" and "none resolved" are different claims.
    """
    total = sum(s.total_cycles for s in sources)
    busy: dict[str, int] = {}
    for s in sources:
        for r in s.resources:
            busy[r.name] = busy.get(r.name, 0) + r.busy_cycles

    unresolved: set[str] = set()
    reasons: dict[str, str] = {}
    for s in sources:
        env = envelopes.get(s.workload)
        if env is None:
            continue
        for t in env.times:
            if not t.known:
                unresolved.add(t.resource)
                reasons.setdefault(t.resource, t.reason)

    all_resources = set(busy)
    unresolved_busy = sum(busy.get(n, 0) for n in unresolved)
    denom = sum(busy.values())
    share: "float | _Unknown" = UNKNOWN if denom <= 0 else 1.0 - unresolved_busy / denom

    structural_share: "float | _Unknown" = UNKNOWN
    if structural_resources is not None and denom > 0:
        structural_share = sum(busy.get(n, 0) for n in set(structural_resources)) / denom

    walked = 0 if timing_records is None else len(timing_records)
    resolved_modules = 0 if timing_records is None else sum(
        1 for r in timing_records if r.get("pipeline_depth") is not None)

    note = ("time weighting uses each resource's share of measured busy cycles over the corpus "
            f"({denom} busy cycles across {total} measured cycles)")
    if timing_records is None:
        note += ("; no timing records were supplied, so the module count is 0/0 -- UNCACHED, not a "
                 "design with no sequenced logic")
    return Coverage(
        resolved_module_count=resolved_modules, walked_module_count=walked,
        resolved_resource_count=len(all_resources - unresolved), resource_count=len(all_resources),
        time_weighted_resolved_share=share, structurally_resolved_time_share=structural_share,
        unresolved_time_share={n: busy.get(n, 0) / denom for n in sorted(unresolved)} if denom
        else {},
        unresolved_reasons=reasons, note=note)


# ------------------------------------------------------------------------------------------------
# predictions
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Prediction:
    """One workload's structural bound set against what it measured."""

    workload: str
    measured_cycles: int
    envelope: StructuralEnvelope
    attribution: "Attribution | None" = None

    @property
    def predicted_cycles(self) -> "float | _Unknown":
        return self.envelope.lower_bound_cycles

    @property
    def recovered_share(self) -> "float | _Unknown":
        """Predicted over measured. A bound recovers *at most* 1.0; more than 1.0 falsifies it."""
        if is_unknown(self.predicted_cycles) or self.measured_cycles <= 0:
            return UNKNOWN
        return float(self.predicted_cycles) / self.measured_cycles

    @property
    def partial_recovered_share(self) -> float:
        """The same ratio over the RESOLVED subset. Weaker, and named so it cannot be mistaken."""
        if self.measured_cycles <= 0:
            return 0.0
        return self.envelope.partial_lower_bound_cycles / self.measured_cycles

    @property
    def respects_floor(self) -> bool:
        """No prediction may fall below the busiest single resource plus the workload intercepts."""
        c = self.envelope.composed
        value = c.partial_cycles if is_unknown(c.cycles) else float(c.cycles)
        return value >= c.floor_cycles - 1e-9

    @property
    def is_valid_lower_bound(self) -> "bool | _Unknown":
        """Whether the bound is below the measurement. UNKNOWN when the bound is UNKNOWN."""
        if is_unknown(self.predicted_cycles):
            return UNKNOWN
        return float(self.predicted_cycles) <= self.measured_cycles

    def to_dict(self) -> dict[str, Any]:
        def _s(v: Any) -> Any:
            return "UNKNOWN" if v is UNKNOWN else v

        out = {"workload": self.workload, "measured_cycles": self.measured_cycles,
               "predicted_cycles": _s(self.predicted_cycles),
               "recovered_share": _s(self.recovered_share),
               "partial_recovered_share": self.partial_recovered_share,
               "respects_floor": self.respects_floor,
               "is_valid_lower_bound": _s(self.is_valid_lower_bound),
               "envelope": self.envelope.to_dict()}
        if self.attribution is not None:
            out["attribution"] = self.attribution.to_dict()
        return out


@dataclass(frozen=True)
class CorpusPrediction:
    """Predictions over a corpus, with the failures kept visible rather than averaged away."""

    predictions: dict[str, Prediction] = field(default_factory=dict)
    coverage: "Coverage | None" = None
    attribution: "CorpusAttribution | None" = None
    operator: "Composition | None" = None
    eta: float = 0.0

    @property
    def resolved(self) -> dict[str, Prediction]:
        return {n: p for n, p in self.predictions.items() if not is_unknown(p.predicted_cycles)}

    @property
    def unresolved(self) -> dict[str, tuple[str, ...]]:
        return {n: p.envelope.unresolved for n, p in self.predictions.items()
                if p.envelope.unresolved}

    @property
    def floor_violations(self) -> tuple[str, ...]:
        """Workloads whose prediction fell below the structural bound. Must always be empty."""
        return tuple(sorted(n for n, p in self.predictions.items() if not p.respects_floor))

    @property
    def bound_violations(self) -> tuple[str, ...]:
        """Workloads where the lower bound exceeds the measurement -- an input is falsified."""
        return tuple(sorted(n for n, p in self.predictions.items()
                            if p.is_valid_lower_bound is False))

    @property
    def limiters(self) -> dict[str, "str | _Unknown"]:
        return {n: p.envelope.limiter for n, p in self.predictions.items()}

    def corpus_recovered_share(self) -> "float | _Unknown":
        """Summed predicted over summed measured, on the workloads whose bound resolved."""
        res = self.resolved
        if not res:
            return UNKNOWN
        measured = sum(p.measured_cycles for p in res.values())
        if measured <= 0:
            return UNKNOWN
        return sum(float(p.predicted_cycles) for p in res.values()) / measured

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": None if self.operator is None else self.operator.value,
            "eta": self.eta,
            "n_workloads": len(self.predictions),
            "n_resolved": len(self.resolved),
            "corpus_recovered_share": ("UNKNOWN" if is_unknown(self.corpus_recovered_share())
                                       else self.corpus_recovered_share()),
            "floor_violations": list(self.floor_violations),
            "bound_violations": list(self.bound_violations),
            "unresolved": {k: list(v) for k, v in self.unresolved.items()},
            "coverage": None if self.coverage is None else self.coverage.to_dict(),
            "predictions": {n: p.to_dict() for n, p in sorted(self.predictions.items())},
            "corpus_attribution": None if self.attribution is None else {
                "closes": self.attribution.closes,
                "residual_cycles": self.attribution.residual_cycles,
                "residual_is_constant": ("UNKNOWN" if is_unknown(
                    self.attribution.residual_is_constant)
                    else self.attribution.residual_is_constant),
                "bucket_cycles": self.attribution.bucket_cycles(),
                "families": self.attribution.families(),
            },
        }


def compose_corpus(sources: "Sequence[ActivitySource]", *,
                   times: "Mapping[str, Sequence[ResourceTime]]",
                   operator: Composition, eta: float,
                   fixed: "Mapping[str, Sequence[FixedTerm]] | None" = None,
                   buckets: "Mapping[str, str] | None" = None,
                   amplifications: "Mapping[str, Any] | None" = None,
                   headrooms: "Mapping[str, Any] | None" = None,
                   timing_records: "Sequence[Mapping[str, Any]] | None" = None,
                   structural_resources: "Iterable[str] | None" = None) -> CorpusPrediction:
    """Build every workload's envelope under one derived operator, then attribute its gap.

    ``operator`` and ``eta`` are required and are not defaulted anywhere on this path. Textbook
    roofline's ``max`` is one of three admissible answers here, and picking it without an overlap
    observation is the error the whole module exists to make impossible to commit silently.
    """
    envelopes: dict[str, StructuralEnvelope] = {}
    for s in sources:
        envelopes[s.workload] = envelope(
            s.workload, list(times.get(s.workload, ())), operator=operator, eta=eta,
            fixed=list((fixed or {}).get(s.workload, ())))

    corpus_attr = None
    if buckets is not None:
        corpus_attr = attribute_corpus(sources, buckets=buckets, envelopes=envelopes,
                                       amplifications=amplifications, headrooms=headrooms)

    preds = {
        s.workload: Prediction(
            workload=s.workload, measured_cycles=s.total_cycles, envelope=envelopes[s.workload],
            attribution=None if corpus_attr is None else corpus_attr.workloads.get(s.workload))
        for s in sources
    }
    return CorpusPrediction(predictions=preds,
                            coverage=coverage(sources, envelopes,
                                              timing_records=timing_records,
                                              structural_resources=structural_resources),
                            attribution=corpus_attr, operator=operator, eta=eta)


# ------------------------------------------------------------------------------------------------
# what defaulting the operator would cost
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatorSensitivity:
    """The same terms composed under each admissible operator, so the choice is priced.

    ``understatement_vs_sum`` is what a textbook roofline gives up by asserting perfect overlap on a
    target that does not overlap: ``1 - T_max / T_sum``. It is reported over the workloads whose
    terms all resolved, because an operator comparison over partially-resolved terms compares two
    different sets of terms.
    """

    n_workloads: int
    by_operator: dict[str, float]
    per_workload: dict[str, dict[str, float]]
    understatement_vs_sum: float
    worst_workload: str
    worst_understatement: float


def operator_sensitivity(sources: "Sequence[ActivitySource]", *,
                         times: "Mapping[str, Sequence[ResourceTime]]",
                         fixed: "Mapping[str, Sequence[FixedTerm]] | None" = None,
                         eta: float = 0.5) -> "OperatorSensitivity | Unavailable":
    """Compose each workload under SUM, MAX and PARTIAL and report what the choice is worth."""
    totals: dict[str, float] = {c.value: 0.0 for c in Composition}
    per: dict[str, dict[str, float]] = {}
    n = 0
    for s in sources:
        ts = list(times.get(s.workload, ()))
        if not ts or any(not t.known for t in ts):
            continue
        wf = [f for f in (fixed or {}).get(s.workload, ()) if not f.resource]
        row: dict[str, float] = {}
        for op in Composition:
            c: Composed = compose(ts, operator=op, eta=eta, workload_fixed=wf)
            row[op.value] = float(c.cycles)
            totals[op.value] += float(c.cycles)
        per[s.workload] = row
        n += 1
    if not n:
        return Unavailable("operator sensitivity",
                           ("at least one workload whose resource times all resolved",),
                           "an operator comparison over partially-resolved terms would compare "
                           "two different sets of terms")
    under = 0.0 if totals["sum"] <= 0 else 1.0 - totals["max"] / totals["sum"]
    worst_name, worst = max(
        ((w, 0.0 if r["sum"] <= 0 else 1.0 - r["max"] / r["sum"]) for w, r in per.items()),
        key=lambda p: p[1])
    return OperatorSensitivity(n_workloads=n, by_operator=totals, per_workload=per,
                               understatement_vs_sum=under, worst_workload=worst_name,
                               worst_understatement=worst)


# ------------------------------------------------------------------------------------------------
# emission
# ------------------------------------------------------------------------------------------------


def emit_envelope_report(prediction: CorpusPrediction, *, target: str, version: int = 1,
                         sources: "Sequence[str]" = (), notes: str = "") -> Any:
    """Write the corpus prediction as a versioned product under the single generated-output root."""
    from merlin.common.artifacts import new_product

    pd = new_product("perf-envelope", version=version, target=target, sources=list(sources),
                     notes=notes or ("structural envelope, generalized ridge point and gap "
                                     "attribution per workload; coverage is weighted by measured "
                                     "time, and the module count is reported separately because it "
                                     "is biased away from where the cycles are"))
    body = prediction.to_dict()
    pd.add_artifact("envelope.json").write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.write_manifest()
    return pd.path
