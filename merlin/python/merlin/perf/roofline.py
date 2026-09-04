"""Empirical roofline points built only from explicit measured inputs.

This module is a reporting edge over :mod:`merlin.perf.envelope`: it does not
derive geometry, translate MACs into operations, invent bandwidth, or choose a
composition operator.  The caller supplies measured demands, measured peaks,
fixed terms, and the measured composition evidence.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import string
from typing import Any

from merlin.dse_guidance.evidence import EVIDENCE_TYPES

from .composer import Prediction
from .decompose import UNKNOWN, ResourceKind, Unavailable, is_unknown
from .envelope import Basis, FixedTerm, Peak, ResourceDemand, ResourceTime, envelope, resource_time
from .headroom import Composition

__all__ = [
    "EmpiricalObservation",
    "EvidenceReceipt",
    "RooflineCoverage",
    "RooflinePoint",
    "RooflineReport",
    "empirical_roofline",
]

_EMPIRICAL_EVIDENCE_KINDS = frozenset(EVIDENCE_TYPES[:3])
_MIN_FIT_SAMPLES = 4


@dataclass(frozen=True)
class EvidenceReceipt:
    """Content-addressed raw evidence retained behind a derived roofline quantity."""

    artifact_sha256: str
    source_kind: str
    sample_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"artifact_sha256": self.artifact_sha256, "source_kind": self.source_kind,
                "sample_ids": list(self.sample_ids)}


def _receipt_problem(receipt: Any, *, source_kinds: tuple[str, ...], min_samples: int) -> str | None:
    if not isinstance(receipt, EvidenceReceipt):
        return "a typed EvidenceReceipt is absent"
    digest = receipt.artifact_sha256
    if (len(digest) != 64 or any(char not in string.hexdigits for char in digest)):
        return "the evidence artifact SHA-256 is malformed"
    if receipt.source_kind not in source_kinds:
        return f"source kind {receipt.source_kind!r} is not one of {list(source_kinds)}"
    if (len(receipt.sample_ids) < min_samples or len(set(receipt.sample_ids)) != len(receipt.sample_ids)
            or any(not str(sample).strip() for sample in receipt.sample_ids)):
        return f"at least {min_samples} distinct raw sample id(s) are required"
    return None


def _serial(value: Any) -> Any:
    if isinstance(value, Unavailable):
        return {"what": value.what, "missing": list(value.missing), "detail": value.detail}
    if is_unknown(value):
        return "UNKNOWN"
    return value


@dataclass(frozen=True)
class EmpiricalObservation:
    """Measured runtime, work, and actual traffic for one workload.

    ``work`` carries its own unit (for example instructions, elements, or
    operations); no operation conversion is implied. ``moved_bytes`` carries
    one :class:`ResourceDemand` for each observed memory level.
    """

    workload: str
    cycles: int
    work: ResourceDemand | None
    moved_bytes: tuple[ResourceDemand, ...] = ()
    provenance: str = ""


@dataclass(frozen=True)
class RooflineCoverage:
    expected: tuple[str, ...]
    observed: tuple[str, ...]
    resolved: tuple[str, ...]
    missing: tuple[str, ...] = ()
    unexpected: tuple[str, ...] = ()
    cycle_weighted_resolved_share: Any = UNKNOWN

    @property
    def complete(self) -> bool:
        return (not self.missing and not self.unexpected
                and self.expected == self.observed and self.expected == self.resolved)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected": list(self.expected),
            "observed": list(self.observed),
            "resolved": list(self.resolved),
            "missing": list(self.missing),
            "unexpected": list(self.unexpected),
            "complete": self.complete,
            "cycle_weighted_resolved_share": _serial(self.cycle_weighted_resolved_share),
        }


@dataclass(frozen=True)
class RooflinePoint:
    workload: str
    measured_cycles: Any
    work: Any
    work_unit: str
    moved_bytes: dict[str, float]
    measured_rate: Any
    intensity_by_level: dict[str, Any]
    bound_cycles: Any
    is_valid_lower_bound: Any
    efficiency: Any
    limiter: Any
    margin_to_second: Any
    margin_share: Any
    envelope: Any
    provenance: dict[str, Any]
    refusals: tuple[Unavailable, ...] = ()

    @property
    def resolved(self) -> bool:
        return not self.refusals and not is_unknown(self.bound_cycles)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload": self.workload,
            "measured_cycles": self.measured_cycles,
            "work": _serial(self.work),
            "work_unit": self.work_unit,
            "measured_rate": _serial(self.measured_rate),
            "measured_rate_unit": f"{self.work_unit}/cycle",
            "moved_bytes": dict(self.moved_bytes),
            "intensity_by_level": {name: _serial(value)
                                   for name, value in sorted(self.intensity_by_level.items())},
            "intensity_unit": f"{self.work_unit}/byte",
            "bound_cycles": _serial(self.bound_cycles),
            "is_valid_lower_bound": _serial(self.is_valid_lower_bound),
            "efficiency": _serial(self.efficiency),
            "limiter": _serial(self.limiter),
            "margin_to_second": _serial(self.margin_to_second),
            "margin_share": _serial(self.margin_share),
            "resolved": self.resolved,
            "provenance": self.provenance,
            "refusals": [_serial(refusal) for refusal in self.refusals],
            "envelope": None if self.envelope is None else self.envelope.to_dict(),
        }


@dataclass(frozen=True)
class RooflineReport:
    points: dict[str, RooflinePoint] = field(default_factory=dict)
    coverage: RooflineCoverage | Unavailable = field(
        default_factory=lambda: Unavailable("roofline coverage", ("expected workloads",)))
    composition: Composition | None = None
    composition_eta: Any = UNKNOWN
    composition_provenance: str = ""
    refusals: tuple[Unavailable, ...] = ()

    @property
    def complete(self) -> bool:
        return (not self.refusals and isinstance(self.coverage, RooflineCoverage)
                and self.coverage.complete
                and all(point.resolved for point in self.points.values()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "empirical_roofline_v1",
            "status": "resolved" if self.complete else "refused",
            "composition": (self.composition.value
                            if isinstance(self.composition, Composition) else None),
            "composition_eta": _serial(self.composition_eta),
            "composition_provenance": self.composition_provenance,
            "coverage": (self.coverage.to_dict() if isinstance(self.coverage, RooflineCoverage)
                         else _serial(self.coverage)),
            "refusals": [_serial(refusal) for refusal in self.refusals],
            "points": {name: point.to_dict() for name, point in sorted(self.points.items())},
        }


def empirical_roofline(
        observations: Sequence[EmpiricalObservation], *,
        peaks: Mapping[str, Peak] | None = None,
        fixed_terms: Mapping[str, Sequence[FixedTerm]] | None = None,
        composition: Composition | None = None,
        composition_eta: float | None = None,
        composition_provenance: str = "",
        evidence_receipts: Mapping[str, EvidenceReceipt] | None = None,
        expected_workloads: Sequence[str] | None = None) -> RooflineReport:
    """Compose empirical roofline points without filling any absent input."""
    global_refusals: list[Unavailable] = []
    if not isinstance(composition, Composition):
        global_refusals.append(Unavailable(
            "roofline composition", ("an explicitly measured Composition",),
            "the empirical report never defaults to textbook max"))
    if (isinstance(composition_eta, bool) or not isinstance(composition_eta, (int, float))
            or not 0.0 <= float(composition_eta) <= 1.0):
        global_refusals.append(Unavailable(
            "roofline composition eta", ("a measured eta in [0, 1]",)))
    if not str(composition_provenance or "").strip():
        global_refusals.append(Unavailable(
            "roofline composition", ("composition provenance",)))
    receipts = evidence_receipts or {}
    composition_receipt_problem = _receipt_problem(
        receipts.get("composition"), source_kinds=("rtl_counter_partition",), min_samples=1)
    if composition_receipt_problem:
        global_refusals.append(Unavailable(
            "roofline composition evidence", ("RTL-counter partition receipt",),
            composition_receipt_problem))
    if isinstance(composition, Composition) and isinstance(composition_eta, (int, float)) \
            and not isinstance(composition_eta, bool) and 0.0 <= float(composition_eta) <= 1.0:
        eta = float(composition_eta)
        derived = (Composition.SUM if eta <= 0.05 else
                   Composition.MAX if eta >= 0.95 else Composition.PARTIAL)
        if composition is not derived:
            global_refusals.append(Unavailable(
                "roofline composition", ("an operator consistent with measured eta",),
                f"eta={eta:g} derives {derived.value!r}, not {composition.value!r}"))
    names = [observation.workload for observation in observations]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        global_refusals.append(Unavailable(
            "roofline observations", ("unique workload identities",),
            f"duplicate observations: {duplicates}"))
    reported_composition = composition if isinstance(composition, Composition) else None
    reported_eta: Any = (
        float(composition_eta)
        if (isinstance(composition_eta, (int, float))
            and not isinstance(composition_eta, bool)
            and 0.0 <= float(composition_eta) <= 1.0)
        else UNKNOWN)

    points: dict[str, RooflinePoint] = {}
    report_refusals = list(global_refusals)
    for observation in observations:
        point_refusals = list(global_refusals)
        provenance_invalid = False
        traffic_invalid = False
        work = observation.work if isinstance(observation.work, ResourceDemand) else None
        cycles_valid = (isinstance(observation.cycles, int)
                        and not isinstance(observation.cycles, bool)
                        and observation.cycles > 0)
        if work is None:
            point_refusals.append(Unavailable(
                f"roofline point {observation.workload!r}",
                ("an explicit measured work quantity and unit",),
                "work is never inferred from shape, geometry, or a MAC conversion"))
        if not cycles_valid:
            point_refusals.append(Unavailable(
                f"roofline point {observation.workload!r}",
                ("a positive integer measured cycle count",)))
        if not str(observation.provenance or "").strip():
            point_refusals.append(Unavailable(
                f"roofline point {observation.workload!r}",
                ("cycle-measurement provenance",)))
            provenance_invalid = True
        observation_receipt_problem = _receipt_problem(
            receipts.get(f"observation:{observation.workload}"),
            source_kinds=("rtl_cycle_measurement",), min_samples=1)
        if observation_receipt_problem:
            point_refusals.append(Unavailable(
                f"roofline point {observation.workload!r}",
                ("a content-addressed RTL cycle receipt",), observation_receipt_problem))
            provenance_invalid = True
        if work is not None and not str(work.provenance or "").strip():
            point_refusals.append(Unavailable(
                f"roofline work for {observation.workload!r}",
                ("work-measurement provenance",)))
            provenance_invalid = True
        if work is not None:
            work_receipt_problem = _receipt_problem(
                receipts.get(f"work:{observation.workload}:{work.resource}"),
                source_kinds=("compiler_ir",), min_samples=1)
            if work_receipt_problem:
                point_refusals.append(Unavailable(
                    f"roofline work for {observation.workload!r}",
                    ("a content-addressed compiler-IR work receipt",), work_receipt_problem))
                provenance_invalid = True
        for demand in observation.moved_bytes:
            if (not isinstance(demand, ResourceDemand)
                    or demand.kind is not ResourceKind.MOVEMENT
                    or demand.basis is not Basis.MOVED
                    or demand.unit != "bytes"
                    or demand.amount <= 0):
                resource = getattr(demand, "resource", "unknown")
                point_refusals.append(Unavailable(
                    f"roofline traffic for {observation.workload!r}",
                    (f"positive actual moved bytes for {resource}",),
                    "transaction counts need an explicit measured byte conversion; algorithmic "
                    "bytes and measured zero are not actual moved-byte observations"))
                traffic_invalid = True
                continue
            if not str(demand.provenance or "").strip():
                point_refusals.append(Unavailable(
                    f"roofline traffic for {observation.workload!r}",
                    (f"moved-byte provenance for {demand.resource}",)))
                provenance_invalid = True
            traffic_receipt_problem = _receipt_problem(
                receipts.get(f"traffic:{observation.workload}:{demand.resource}"),
                source_kinds=("physical_counter",), min_samples=1)
            if traffic_receipt_problem:
                point_refusals.append(Unavailable(
                    f"roofline traffic for {observation.workload!r}",
                    (f"a content-addressed physical-counter receipt for {demand.resource}",),
                    traffic_receipt_problem))
                provenance_invalid = True
        if fixed_terms is None or observation.workload not in fixed_terms:
            point_refusals.append(Unavailable(
                f"roofline fixed terms for {observation.workload!r}",
                ("an explicit fixed-term measurement set",),
                "an explicit empty sequence means measured zero terms; an absent entry is UNKNOWN"))
        workload_fixed = tuple((fixed_terms or {}).get(observation.workload, ()))
        if fixed_terms is not None and observation.workload in fixed_terms:
            fixed_receipt_problem = _receipt_problem(
                receipts.get(f"fixed:{observation.workload}"),
                source_kinds=("calibration_fit",), min_samples=_MIN_FIT_SAMPLES)
            if fixed_receipt_problem:
                point_refusals.append(Unavailable(
                    f"roofline fixed terms for {observation.workload!r}",
                    ("a rate/intercept calibration-fit receipt",), fixed_receipt_problem))
        demand_resources = ([work.resource] if work is not None else []) + [
            demand.resource for demand in observation.moved_bytes
            if isinstance(demand, ResourceDemand)]
        duplicate_resources = sorted(
            name for name, count in Counter(demand_resources).items() if count > 1)
        if duplicate_resources:
            point_refusals.append(Unavailable(
                f"roofline demands for {observation.workload!r}",
                ("unique resource demands",),
                f"duplicate resource demands: {duplicate_resources}"))
        unmatched_fixed = sorted(
            term.resource for term in workload_fixed
            if isinstance(term, FixedTerm) and term.resource
            and term.resource not in demand_resources)
        if unmatched_fixed:
            point_refusals.append(Unavailable(
                f"roofline fixed terms for {observation.workload!r}",
                ("a matching resource demand for every resource-owned fixed term",),
                f"unmatched fixed-term resources: {unmatched_fixed}"))
        for term in workload_fixed:
            if not isinstance(term, FixedTerm):
                point_refusals.append(Unavailable(
                    f"roofline fixed terms for {observation.workload!r}",
                    ("measured FixedTerm objects",),
                    f"got {type(term).__name__}"))
                continue
            if term.evidence_kind not in _EMPIRICAL_EVIDENCE_KINDS:
                point_refusals.append(Unavailable(
                    f"roofline fixed terms for {observation.workload!r}",
                    (f"measured evidence for fixed term {term.name}",),
                    f"evidence kind {term.evidence_kind!r} is not empirical"))
            if not str(term.provenance or "").strip():
                point_refusals.append(Unavailable(
                    f"roofline fixed terms for {observation.workload!r}",
                    (f"fixed-term provenance for {term.name}",)))
                provenance_invalid = True
        if not observation.moved_bytes:
            point_refusals.append(Unavailable(
                f"roofline point {observation.workload!r}",
                ("actual moved bytes at one or more memory levels",),
                "algorithmic bytes and zero are not substitutes for an absent bus measurement"))
        if point_refusals:
            report_refusals.extend(point_refusals[len(global_refusals):])
            work_amount: Any = UNKNOWN if work is None else float(work.amount)
            work_unit = "" if work is None else work.unit
            valid_moved = tuple(
                demand for demand in observation.moved_bytes
                if isinstance(demand, ResourceDemand))
            demand_resources = ([work.resource] if work is not None else []) + [
                demand.resource for demand in valid_moved]
            points[observation.workload] = RooflinePoint(
                workload=observation.workload,
                measured_cycles=observation.cycles,
                work=work_amount,
                work_unit=work_unit,
                moved_bytes={d.resource: float(d.amount) for d in valid_moved},
                measured_rate=(UNKNOWN if provenance_invalid or work is None or not cycles_valid
                               else float(work.amount) / observation.cycles),
                intensity_by_level={
                    d.resource: (UNKNOWN
                                 if provenance_invalid or traffic_invalid or work is None
                                 else float(work.amount) / float(d.amount))
                    for d in valid_moved},
                bound_cycles=UNKNOWN, efficiency=UNKNOWN, limiter=UNKNOWN,
                is_valid_lower_bound=UNKNOWN,
                margin_to_second=UNKNOWN, margin_share=UNKNOWN, envelope=None,
                provenance={
                    "cycles": observation.provenance,
                    "work": "" if work is None else work.provenance,
                    "moved_bytes": {
                        demand.resource: demand.provenance
                        for demand in observation.moved_bytes
                        if isinstance(demand, ResourceDemand)},
                    "peaks": {
                        resource: peak.provenance
                        for resource in demand_resources
                        if isinstance((peak := (peaks or {}).get(resource)), Peak)},
                    "fixed_terms": {
                        term.name: term.provenance for term in workload_fixed
                        if isinstance(term, FixedTerm)},
                },
                refusals=tuple(point_refusals),
            )
            continue
        assert work is not None and cycles_valid
        demands = (work, *observation.moved_bytes)
        terms: list[ResourceTime] = []
        for demand in demands:
            peak = (peaks or {}).get(demand.resource)
            if not isinstance(peak, Peak):
                reason = f"no measured Peak was supplied for resource {demand.resource!r}"
                point_refusals.append(Unavailable(
                    f"roofline peak for {demand.resource}", ("a measured Peak",), reason))
                terms.append(ResourceTime(
                    demand.resource, demand.kind, UNKNOWN, demand.unit, demand.basis,
                    reason=reason))
                continue
            if peak.resource != demand.resource or peak.unit != demand.unit:
                reason = (f"demand {demand.resource!r}/{demand.unit!r} does not match measured "
                          f"peak {peak.resource!r}/{peak.unit!r}")
                point_refusals.append(Unavailable(
                    f"roofline peak for {demand.resource}",
                    ("exact peak identity and unit",), reason))
                terms.append(ResourceTime(
                    demand.resource, demand.kind, UNKNOWN, demand.unit, demand.basis,
                    reason=reason))
                continue
            if not str(peak.provenance or "").strip():
                reason = f"the Peak for resource {demand.resource!r} has no provenance"
                point_refusals.append(Unavailable(
                    f"roofline peak for {demand.resource}", ("peak provenance",), reason))
                terms.append(ResourceTime(
                    demand.resource, demand.kind, UNKNOWN, demand.unit, demand.basis,
                    reason=reason))
                continue
            if not peak.known:
                term = resource_time(demand, peak, workload_fixed)
                terms.append(term)
                point_refusals.append(Unavailable(
                    f"roofline peak for {demand.resource}",
                    ("a resolved measured Peak",), term.reason))
                continue
            peak_receipt_problem = _receipt_problem(
                receipts.get(f"peak:{demand.resource}"),
                source_kinds=("calibration_fit",), min_samples=_MIN_FIT_SAMPLES)
            if peak_receipt_problem:
                reason = f"the Peak for {demand.resource!r} lacks raw fit evidence: {peak_receipt_problem}"
                point_refusals.append(Unavailable(
                    f"roofline peak for {demand.resource}",
                    ("a content-addressed rate/intercept fit receipt",), reason))
                terms.append(ResourceTime(
                    demand.resource, demand.kind, UNKNOWN, demand.unit, demand.basis,
                    reason=reason))
                continue
            if peak.n_samples < _MIN_FIT_SAMPLES or peak.is_ceiling is not True:
                reason = (f"the Peak for {demand.resource!r} has n_samples={peak.n_samples} and "
                          f"is_ceiling={peak.is_ceiling}; a rate+intercept fit needs at least "
                          f"{_MIN_FIT_SAMPLES} samples and a falsified ceiling")
                point_refusals.append(Unavailable(
                    f"roofline peak for {demand.resource}",
                    ("a calibrated empirical ceiling",), reason))
                terms.append(ResourceTime(
                    demand.resource, demand.kind, UNKNOWN, demand.unit, demand.basis,
                    reason=reason))
                continue
            if peak.evidence_kind not in _EMPIRICAL_EVIDENCE_KINDS:
                reason = (f"Peak evidence kind {peak.evidence_kind!r} is not an empirical "
                          "measurement")
                point_refusals.append(Unavailable(
                    f"roofline peak for {demand.resource}",
                    ("measured evidence for the peak",), reason))
                terms.append(ResourceTime(
                    demand.resource, demand.kind, UNKNOWN, demand.unit, demand.basis,
                    reason=reason))
                continue
            term = resource_time(
                demand, peak, workload_fixed)
            terms.append(term)
            if not term.known:
                point_refusals.append(Unavailable(
                    f"roofline peak for {demand.resource}", ("a resolved measured Peak",),
                    term.reason))
        env = envelope(observation.workload, terms, operator=composition,
                       eta=composition_eta,
                       fixed=workload_fixed)
        prediction = Prediction(observation.workload, observation.cycles, env)
        bound = prediction.predicted_cycles
        valid_bound = prediction.is_valid_lower_bound
        if valid_bound is False:
            point_refusals.append(Unavailable(
                f"roofline bound for {observation.workload!r}",
                ("consistent measured demands, peaks, fixed terms, and cycles",),
                (f"composed lower bound {float(bound)} exceeds measured "
                 f"{observation.cycles} cycles")))
        efficiency = (prediction.recovered_share if valid_bound is True else UNKNOWN)
        point = RooflinePoint(
            workload=observation.workload,
            measured_cycles=observation.cycles,
            work=float(work.amount),
            work_unit=work.unit,
            moved_bytes={d.resource: float(d.amount) for d in observation.moved_bytes},
            measured_rate=float(work.amount) / observation.cycles,
            intensity_by_level={d.resource: float(work.amount) / float(d.amount)
                                for d in observation.moved_bytes},
            bound_cycles=bound,
            is_valid_lower_bound=valid_bound,
            efficiency=efficiency,
            limiter=env.limiter,
            margin_to_second=env.margin_to_second,
            margin_share=env.margin_share,
            envelope=env,
            provenance={
                "cycles": observation.provenance,
                "work": work.provenance,
                "moved_bytes": {d.resource: d.provenance for d in observation.moved_bytes},
                "peaks": {
                    demand.resource: peak.provenance
                    for demand in demands
                    if isinstance((peak := (peaks or {}).get(demand.resource)), Peak)},
                "fixed_terms": {term.name: term.provenance for term in workload_fixed},
            },
            refusals=tuple(point_refusals),
        )
        points[observation.workload] = point
        report_refusals.extend(point_refusals[len(global_refusals):])

    observed = tuple(sorted(points))
    resolved = tuple(sorted(name for name, point in points.items() if point.resolved))
    if expected_workloads is None:
        coverage: RooflineCoverage | Unavailable = Unavailable(
            "roofline coverage", ("explicit expected workload identities",),
            "the observations cannot define their own denominator")
        report_refusals.append(coverage)
    else:
        raw_expected = tuple(expected_workloads)
        repeated_expected = sorted(
            name for name, count in Counter(raw_expected).items() if count > 1)
        if not raw_expected or repeated_expected:
            coverage = Unavailable(
                "roofline coverage", ("nonempty unique expected workload identities",),
                ("the declared set is empty" if not raw_expected
                 else f"duplicate expected identities: {repeated_expected}"))
            report_refusals.append(coverage)
            return RooflineReport(
                points=points, coverage=coverage, composition=reported_composition,
                composition_eta=reported_eta,
                composition_provenance=composition_provenance,
                refusals=tuple(report_refusals))
        expected = tuple(sorted(raw_expected))
        missing = tuple(sorted(set(expected) - set(observed)))
        unexpected = tuple(sorted(set(observed) - set(expected)))
        total_cycles = sum(point.measured_cycles for point in points.values()
                           if isinstance(point.measured_cycles, int)
                           and not isinstance(point.measured_cycles, bool)
                           and point.measured_cycles > 0)
        resolved_cycles = sum(points[name].measured_cycles for name in resolved)
        share: Any = (resolved_cycles / total_cycles if total_cycles else UNKNOWN)
        if missing or unexpected:
            share = UNKNOWN
            report_refusals.append(Unavailable(
                "roofline coverage", ("the exact expected workload set",),
                f"missing={list(missing)}, unexpected={list(unexpected)}"))
        coverage = RooflineCoverage(
            expected=expected, observed=observed, resolved=resolved,
            missing=missing, unexpected=unexpected,
            cycle_weighted_resolved_share=share,
        )
    return RooflineReport(
        points=points, coverage=coverage, composition=reported_composition,
        composition_eta=reported_eta,
        composition_provenance=composition_provenance, refusals=tuple(report_refusals))
