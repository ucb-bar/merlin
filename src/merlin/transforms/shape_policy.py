"""Empirical shape/resource guards over real registered pipeline alternatives.

This optional selector never establishes legality, executes a simulator or
reuses a training kernel for another source. It retains incumbent pass paths
until exact paired feedback covers the declared training, heldout and cutoff
boundaries. Ordinary current-IR pass checks still govern every selected path.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from itertools import combinations

from .evaluated_exploration import CompletedObservation, ExplorationIdentity, _pin, exploration_context_identity
from .pass_exploration import ExplorationGeometry, ObservedMetrics, ValidationPartition, _identity
from .pipeline_policy import (
    POLICY_PREDICATE_TYPES,
    FeaturePredicate,
    GraphPredicate,
    PipelinePolicy,
    PipelineVariant,
    PolicyFacts,
    ResourcePredicate,
    RatioFeature,
    DimensionRef,
    ShapePredicate,
    TileFeature,
    execute_policy,
)

_PREDICATES = POLICY_PREDICATE_TYPES


def _match(predicates, facts):
    result = tuple(p.match(facts) for p in predicates)
    return False if False in result else None if None in result else True


@dataclass(frozen=True)
class PerformanceDomain:
    provider_sha256: str
    facts_sha256: str
    library_sha256: str
    scope_sha256: str
    basis: str
    implementation_sha256: str

    def __post_init__(self):
        if (not all(_pin(value) for value in (self.provider_sha256, self.facts_sha256,
                self.library_sha256, self.scope_sha256, self.implementation_sha256))
                or self.basis not in ("modeled_roi", "measured_roi")):
            raise ValueError("performance domain requires selected provider/facts/library/scope and cycle basis")

    @classmethod
    def from_observation(cls, identity, *, implementation_sha256):
        if type(identity) is not ExplorationIdentity:
            raise ValueError("performance domain requires original completed identity")
        identity.__post_init__()
        return cls(identity.provider_sha256, identity.facts_sha256, identity.library_sha256,
                   identity.scope_sha256, identity.basis, implementation_sha256)


@dataclass(frozen=True)
class ShapeScheduleClass:
    """An empirical applicability guard, separate from target legality guards.

    The original PipelineVariant predicates remain independently authoritative.
    Names label reports/registered alternatives, never source models or capsules.
    A missing design keeps this class's incumbent. Alignment belongs to the
    variant's legality predicate, not an empirical cutoff learned here.
    """
    name: str
    predicates: tuple
    incumbent: str
    design: ValidationPartition | None = None

    def __post_init__(self):
        if (type(self.name) is not str or not self.name or type(self.incumbent) is not str or not self.incumbent
                or type(self.predicates) is not tuple or not 1 <= len(self.predicates) <= 16
                or any(type(p) not in _PREDICATES for p in self.predicates)
                or any(type(p) is ShapePredicate and p.multiple_of != 1 for p in self.predicates)
                or self.design is not None and type(self.design) is not ValidationPartition):
            raise ValueError("shape classes need typed performance guards, incumbent and optional frozen design")
        if self.design is not None:
            self.design.__post_init__()


@dataclass(frozen=True)
class ShapeCaseFeedback:
    """Paired observations plus the ordinary selected full-compiler source pin.

    implementation_sha256 binds authored registry/target callback implementations,
    beyond the generic library pin. It must come from the original source
    selection; changing an override requires a new current execution domain.
    Optional diagnostics use the exact observation identity digest as their
    domain. They retain owner-reported traffic, utilization and host costs for
    explanation; complete paired cycles remain the only selection objective.
    """
    geometry: ExplorationGeometry
    facts: PolicyFacts
    observations: tuple[tuple[str, CompletedObservation], ...]
    implementation_sha256: str
    diagnostics: tuple[tuple[str, ObservedMetrics], ...] = ()

    def __post_init__(self):
        if (type(self.geometry) is not ExplorationGeometry or type(self.facts) is not PolicyFacts
                or self.facts.shapes != self.geometry.operand_shapes or not _pin(self.implementation_sha256)
                or type(self.observations) is not tuple or not 1 <= len(self.observations) <= 16
                or any(type(row) is not tuple or len(row) != 2 or type(row[0]) is not str or not row[0]
                       or type(row[1]) is not CompletedObservation for row in self.observations)
                or len({name for name, _ in self.observations}) != len(self.observations)):
            raise ValueError("shape feedback needs actual geometry/facts and distinct exact alternative observations")
        identities = set()
        for _, observation in self.observations:
            observation.__post_init__()
            if (observation.identity.context_sha256 != exploration_context_identity(self.facts.stamp.context)
                or observation.identity.source_sha256 != self.facts.stamp.source_sha256):
                raise ValueError("feedback facts differ from original observed source/context stamp")
            identities.add(observation.identity)
        if len(identities) != 1:
            raise ValueError("paired shape feedback source/inputs/reference/provider/scope differ")
        paired = dict(self.observations)
        if (type(self.diagnostics) is not tuple or len(self.diagnostics) > len(paired)
                or any(type(row) is not tuple or len(row) != 2 or type(row[0]) is not str
                       or row[0] not in paired or type(row[1]) is not ObservedMetrics
                       for row in self.diagnostics)
                or len({name for name, _ in self.diagnostics}) != len(self.diagnostics)):
            raise ValueError("shape diagnostics require distinct observed alternatives and typed metrics")
        for name, metrics in self.diagnostics:
            metrics.__post_init__()
            observed = paired[name]
            if metrics.basis != observed.identity.basis or metrics.domain != observed.identity.digest():
                raise ValueError("shape diagnostics differ from exact source/provider/ROI observation domain")
            if "cycles" in dict(metrics.values) and dict(metrics.values)["cycles"] != observed.cycles:
                raise ValueError("diagnostic cycles differ from the original completed observation")


def _diagnostics(case, name):
    """Forward owner-supplied same-window explanations, without inferring costs.

    ObservedMetrics grants no new measurement authority. The ordinary feedback
    producer still owns resource/traffic provenance and denominators. In
    particular, instruction counts never become host cycles here.
    """
    metrics = dict(case.diagnostics).get(name)
    if metrics is None:
        return None
    return dict(basis=metrics.basis, domain=metrics.domain, values=dict(metrics.values),
                unknown=tuple(key for key, value in metrics.values if value is None),
                use="explanation_only; not a selection objective or qualification")


def _boundaries(predicates):
    required = []
    for index, predicate in enumerate(predicates):
        if type(predicate) is ShapePredicate:
            if predicate.at_least > 1:
                required.append((index, "lower", predicate.at_least))
            if predicate.at_most is not None:
                required.append((index, "upper", predicate.at_most))
        elif type(predicate) in (ResourcePredicate, GraphPredicate):
            if predicate.at_least > 0:
                required.append((index, "lower", predicate.at_least))
            if type(predicate) is GraphPredicate and predicate.at_most is not None:
                required.append((index, "upper", predicate.at_most))
        elif type(predicate) is FeaturePredicate:
            if predicate.at_least[0] > 0:
                required.append((index, "lower", Fraction(*predicate.at_least)))
            if predicate.at_most is not None:
                required.append((index, "upper", Fraction(*predicate.at_most)))
    return tuple(required)


def _value(predicate, facts):
    if type(predicate) is FeaturePredicate:
        return predicate.feature.value(facts)
    if type(predicate) is ShapePredicate:
        if predicate.operand >= len(facts.shapes) or predicate.axis >= len(facts.shapes[predicate.operand]):
            return None
        return facts.shapes[predicate.operand][predicate.axis]
    if type(predicate) is ResourcePredicate:
        return dict(facts.resources).get(predicate.resource)
    return dict(facts.graph).get(predicate.metric)


@dataclass(frozen=True)
class AssessedShapePolicy:
    classes: tuple[ShapeScheduleClass, ...]
    alternatives: tuple[PipelineVariant, ...]
    fallback: str
    selected: tuple[tuple[str, str, PerformanceDomain | None], ...]
    report: dict

    def execute(self, session, registry, facts, *, output, domain=None):
        """Run selected real passes on current IR; never return cached source bytes.

        Missing/stale performance domains or ambiguous classes retain an
        incumbent/fallback, without blocking authoring. Target legality predicates
        and every ordinary pipeline verification still run independently.
        """
        paths = {v.name: v for v in self.alternatives}
        matches = tuple(c for c in self.classes if _match(c.predicates, facts) is True)
        selected = self.fallback
        if len(matches) == 1:
            group = matches[0]
            name, expected = next((name, expected) for key, name, expected in self.selected if key == group.name)
            selected = name if expected is not None and domain == expected else group.incumbent
        # Performance applicability and original legality are distinct checks.
        # Always retain the original unconditional fallback, even when it is the
        # selected incumbent. No retry occurs after a mutating pass starts.
        candidates = []
        if len(matches) == 1:
            group = matches[0]
            for name in dict.fromkeys((selected, group.incumbent)):
                if name != self.fallback:
                    path = paths[name]
                    candidates.append(PipelineVariant(name, path.passes, (*group.predicates, *path.predicates)))
        candidates.append(paths[self.fallback])
        policy = PipelinePolicy(tuple(candidates), self.fallback)
        return execute_policy(session, policy, registry, facts, output=output)


def assess_shape_policy(classes, alternatives, cases, *, fallback, max_regression_ratio=1.0):
    """Choose guarded pass paths only from exact paired complete-cycle evidence.

    Finite cutoffs require BOTH the exact boundary and its adjacent outside
    value, with paired feedback, in the frozen design. At least one matched
    training and one matched heldout geometry must be complete. This reports
    empirical support for that panel, not a universal threshold or physical
    qualification. Exact-source winners remain reported independently.
    """
    if (type(classes) is not tuple or not 1 <= len(classes) <= 16
            or any(type(c) is not ShapeScheduleClass for c in classes)
            or len({c.name for c in classes}) != len(classes)
            or type(alternatives) is not tuple or not 1 <= len(alternatives) <= 16
            or any(type(v) is not PipelineVariant for v in alternatives)
            or len({v.name for v in alternatives}) != len(alternatives)
            or type(cases) is not tuple or len(cases) > 256
            or any(type(c) is not ShapeCaseFeedback for c in cases)
            or len({c.geometry.identity for c in cases}) != len(cases)
            or type(max_regression_ratio) not in (int, float) or not math.isfinite(max_regression_ratio)
            or max_regression_ratio < 1):
        raise ValueError("shape policy needs bounded typed classes, alternatives and paired evidence")
    paths = {v.name: v for v in alternatives}
    if fallback not in paths or paths[fallback].predicates or any(c.incumbent not in paths for c in classes):
        raise ValueError("shape policy needs original unconditional fallback and per-class incumbents")
    for case in cases:
        case.__post_init__()
        if any(name not in paths for name, _ in case.observations):
            raise ValueError("feedback names an unregistered alternative")
    by_geometry = {c.geometry.identity: c for c in cases}
    selected, reports = [], []
    exact = tuple(dict(geometry_sha256=c.geometry.identity,
        paired_identity_sha256=c.observations[0][1].identity.digest(),
        implementation_sha256=c.implementation_sha256,
        observations=[dict(alternative=n, program_sha256=o.program_sha256, feedback_sha256=o.feedback_sha256,
                           eligible=o.eligible, cycles=o.cycles, diagnostics=_diagnostics(c, n))
                      for n, o in c.observations],
        winner=min(((o.cycles, n) for n, o in c.observations
                    if o.eligible and _match(paths[n].predicates, c.facts) is True),
                   default=(None, None))[1]) for c in cases)
    for group in classes:
        group.__post_init__()
        name, domain, best = group.incumbent, None, 1.0
        records = []
        if group.design is not None:
            training = {g.identity for g in group.design.training}
            held = {g.identity for g in group.design.held_out}
            roster = training | held
            panel = [by_geometry[g] for g in sorted(roster) if g in by_geometry]
            matched = [c for c in panel if _match(group.predicates, c.facts) is True]
            missing = sorted(roster-by_geometry.keys())
            boundary_missing = []
            for index, side, edge in _boundaries(group.predicates):
                predicate = group.predicates[index]
                others = group.predicates[:index] + group.predicates[index+1:]
                values = {_value(predicate, c.facts) for c in panel
                          if _match(others, c.facts) is True and predicate.match(c.facts) is not None}
                integer = _integer_boundary(predicate, edge)
                if integer:
                    outside = edge-1 if side == "lower" else edge+1
                    if edge not in values or outside not in values:
                        boundary_missing.append((index, side, int(edge), int(outside)))
                else:
                    # Rational ratios have no universal adjacent value. Require
                    # the exact boundary and a source-observed strict outside
                    # neighbor in the frozen panel; never truncate to an int.
                    outside = [v for v in values if v is not None and (v < edge if side == "lower" else v > edge)]
                    if edge not in values or not outside:
                        boundary_missing.append((index, side, (edge.numerator, edge.denominator),
                                                 "nearest_observed_below" if side == "lower"
                                                 else "nearest_observed_above"))
            eligible = (not missing and not boundary_missing
                and any(c.geometry.identity in training for c in matched)
                and any(c.geometry.identity in held for c in matched))
            for candidate in alternatives:
                ratios, training_ratios, reasons, domains = [], [], [], set()
                for case in panel:
                    match = _match(group.predicates, case.facts)
                    # The incumbent executes outside this class. Neither an
                    # unsupported candidate there nor its missing cycles can
                    # veto a legal, supported path inside the guard.
                    if match is False:
                        continue
                    if (match is None
                        or _match(candidate.predicates, case.facts) is not True
                        or _match(paths[group.incumbent].predicates, case.facts) is not True):
                        reasons.append((case.geometry.identity,
                                        "UNKNOWN class or unadmitted original legality predicates"))
                        continue
                    paired = dict(case.observations)
                    control, observed = paired.get(group.incumbent), paired.get(candidate.name)
                    if control is None or observed is None or not control.eligible or not observed.eligible:
                        reasons.append((case.geometry.identity, "missing/UNKNOWN/failed complete paired cycles"))
                        continue
                    domains.add(PerformanceDomain.from_observation(observed.identity,
                        implementation_sha256=case.implementation_sha256))
                    if _match(group.predicates, case.facts) is True:
                        ratio = observed.cycles/control.cycles
                        ratios.append(ratio)
                        if case.geometry.identity in training:
                            training_ratios.append(ratio)
                support = (eligible and not reasons and len(domains) == 1 and ratios
                           and max(ratios) <= max_regression_ratio)
                # Training chooses; heldout/boundaries can veto, never tune the score.
                score = (math.exp(sum(math.log(v) for v in training_ratios)/len(training_ratios))
                         if training_ratios else None)
                records.append(dict(alternative=candidate.name, supported=bool(support),
                    matched_cycle_ratios=ratios, training_geometric_mean_ratio=score,
                    reasons=reasons, missing_geometries=missing, missing_boundaries=boundary_missing,
                    mixed_performance_domains=len(domains) > 1))
                if support and score < best:
                    name, domain, best = candidate.name, next(iter(domains)), score
        selected.append((group.name, name, domain))
        reports.append(dict(shape_class=group.name, incumbent=group.incumbent, selected=name,
            status="empirically_supported_declared_panel" if domain is not None else "incumbent_retained",
            design=None if group.design is None else group.design.manifest(), alternatives=records))
    return AssessedShapePolicy(classes, alternatives, fallback, tuple(selected),
        dict(schema="merlin.shape_schedule_policy.v1", classes=reports, exact_source_winners=exact,
            max_regression_ratio=max_regression_ratio, policy_sha256=_identity([
                [(c.name, [asdict(p) for p in c.predicates], c.incumbent) for c in classes],
                [(v.name, v.passes, [asdict(p) for p in v.predicates]) for v in alternatives],
                [(key, name, None if domain is None else asdict(domain)) for key, name, domain in selected],
                reports, exact]),
            decision="complete paired cycles, per-class regression retention; no static score fallback",
            decision_metric="complete_same_roi_cycles",
            diagnostic_metrics_policy="exact observation domain; explanations only; UNKNOWN stays unknown",
            legality="original variant predicates and current-IR registered pass checks remain required",
            qualification="provisional empirical panel support, not universal or physical qualification"))


def _feedback_identity(case):
    facts = asdict(case.facts)
    for key in ("operations", "numerical_semantics"):
        if facts[key] is not None:
            facts[key] = sorted(facts[key])
    return _identity((case.geometry.identity, facts, case.implementation_sha256,
                      [(name, asdict(observation)) for name, observation in case.observations]))


def _geometry_value(predicate, geometry, facts=None):
    """Resource values require original facts; geometry never invents capacity."""
    if facts is not None:
        return _value(predicate, facts)
    def dimension(ref):
        return (geometry.operand_shapes[ref.operand][ref.axis]
                if ref.operand < len(geometry.operand_shapes)
                and ref.axis < len(geometry.operand_shapes[ref.operand]) else None)
    if type(predicate) is ShapePredicate:
        return dimension(DimensionRef(predicate.operand, predicate.axis))
    feature = predicate.feature
    if type(feature) is RatioFeature and type(feature.denominator) is DimensionRef:
        numerator, denominator = dimension(feature.numerator), dimension(feature.denominator)
        return Fraction(numerator, denominator) if numerator is not None and denominator else None
    return None


def _geometry_match(predicates, geometry, facts=None):
    matches = []
    for p in predicates:
        value = _geometry_value(p, geometry, facts)
        lower = Fraction(*p.at_least) if type(p) is FeaturePredicate else p.at_least
        upper = Fraction(*p.at_most) if type(p) is FeaturePredicate and p.at_most is not None else p.at_most
        matches.append(None if value is None else value >= lower and (upper is None or value <= upper))
    return False if False in matches else None if None in matches else True


def _integer_boundary(predicate, edge):
    return (type(predicate) is not FeaturePredicate or type(predicate.feature) is TileFeature
            and predicate.feature.kind in ("count", "tail") and edge.denominator == 1)


def _request_matches(request, predicates, geometry, facts=None):
    index = request["predicate_index"]
    if _geometry_match(predicates[:index] + predicates[index+1:], geometry, facts) is not True:
        return False
    value, edge = _geometry_value(predicates[index], geometry, facts), request["value"]
    if type(edge) is dict:
        edge = Fraction(edge["numerator"], edge["denominator"])
    if value is None:
        return False
    comparison = request.get("comparison", "equal")
    return value == edge if comparison == "equal" else value < edge if comparison == "below" else value > edge


@dataclass(frozen=True)
class ShapePolicyProposal:
    """Training-selected guard/path; independent feedback may only veto it.

    Boundary requests describe desired source geometry, not fabricated IR or
    permission to change one operand while ignoring an operation's other axes.
    The ordinary caller materializes legal sources and freezes those geometries
    with ``freeze_boundaries`` before collecting their observations.
    """
    shape_class: ShapeScheduleClass
    alternatives: tuple[PipelineVariant, ...]
    candidate: str
    domain: PerformanceDomain
    training_feedback: tuple[tuple[str, str], ...]
    report: dict
    max_regression_ratio: float = 1.0
    geometry_facts: tuple[PolicyFacts, ...] = ()

    def __post_init__(self):
        if (type(self.shape_class) is not ShapeScheduleClass or self.shape_class.design is None
                or any(type(p) not in (ShapePredicate, FeaturePredicate) for p in self.shape_class.predicates)
                or type(self.alternatives) is not tuple or len(self.alternatives) != 2
                or any(type(v) is not PipelineVariant for v in self.alternatives)
                or {v.name for v in self.alternatives} != {self.shape_class.incumbent, self.candidate}
                or type(self.domain) is not PerformanceDomain or type(self.training_feedback) is not tuple
                or any(type(row) is not tuple or len(row) != 2 or not all(_pin(v) for v in row)
                       for row in self.training_feedback)
                or type(self.report) is not dict or not _pin(self.report.get("proposal_sha256"))):
            raise ValueError("proposal requires its frozen class, typed paths, domain and training identities")
        roster = {g.identity for g in (*self.shape_class.design.training, *self.shape_class.design.held_out)}
        if (type(self.geometry_facts) is not tuple or any(type(f) is not PolicyFacts for f in self.geometry_facts)
                or len({_identity(f.shapes) for f in self.geometry_facts}) != len(self.geometry_facts)
                or any(_identity(f.shapes) not in roster for f in self.geometry_facts)):
            raise ValueError("proposal facts must bind distinct original frozen geometries")
        if (type(self.max_regression_ratio) not in (int, float) or not math.isfinite(self.max_regression_ratio)
                or self.max_regression_ratio < 1):
            raise ValueError("proposal needs its frozen finite training/validation regression limit")

    def requests(self, cases=()):
        """Request only absent/stale paired feedback, reusing exact valid rows."""
        if type(cases) is not tuple or any(type(c) is not ShapeCaseFeedback for c in cases):
            raise ValueError("proposal requests require typed existing case feedback")
        if len({c.geometry.identity for c in cases}) != len(cases):
            raise ValueError("proposal feedback repeats a geometry")
        by_geometry = {c.geometry.identity: c for c in cases}
        facts = {_identity(f.shapes): f for f in self.geometry_facts}
        facts.update({c.geometry.identity: c.facts for c in cases})
        group, paths = self.shape_class, {v.name: v for v in self.alternatives}
        design = group.design
        pending, boundaries = [], []
        for geometry in (*design.training, *design.held_out):
            case = by_geometry.get(geometry.identity)
            needed = (group.incumbent, self.candidate) if _geometry_match(
                group.predicates, geometry, facts.get(geometry.identity)) is not False else (
                group.incumbent,)
            missing = []
            for name in needed:
                observed = None if case is None else dict(case.observations).get(name)
                if (case is None or observed is None or not observed.eligible
                        or _match(paths[name].predicates, case.facts) is not True
                        or PerformanceDomain.from_observation(observed.identity,
                            implementation_sha256=case.implementation_sha256) != self.domain):
                    missing.append(name)
            if missing:
                pending.append(dict(geometry_sha256=geometry.identity, family=geometry.family,
                                    operand_shapes=geometry.operand_shapes, alternatives=missing,
                                    partition="training" if geometry in design.training else "held_out"))
        panel = (*design.training, *design.held_out)
        for index, side, edge in _boundaries(group.predicates):
            predicate = group.predicates[index]
            others = group.predicates[:index] + group.predicates[index+1:]
            integer = _integer_boundary(predicate, edge)
            targets = ((edge, "equal"), (edge-1 if side == "lower" else edge+1, "equal")) if integer else (
                (edge, "equal"), (edge, "below" if side == "lower" else "above"))
            for value, comparison in targets:
                request = dict(predicate_index=index, side=side,
                    value=int(value) if integer else dict(numerator=value.numerator, denominator=value.denominator),
                    other_guards=[asdict(p) for p in others],
                    reason="materialize a legal source with this boundary value, then freeze its geometry")
                if type(predicate) is ShapePredicate:
                    request.update(operand=predicate.operand, axis=predicate.axis)
                else:
                    request.update(feature=asdict(predicate.feature), comparison=comparison,
                        source_facts_required="original selected resource facts for tile/resource features")
                if not any(_request_matches(request, group.predicates, g, facts.get(g.identity)) for g in panel):
                    boundaries.append(request)
        return dict(schema="merlin.shape_policy_requests.v1", proposal_sha256=self.report["proposal_sha256"],
                    pending_evaluations=pending, missing_boundary_sources=boundaries,
                    no_execution_performed=True, selected=False)

    def freeze_boundaries(self, geometries, *, facts=()):
        """Extend the heldout roster before timing, without constructing source.

        Tile/resource feature boundaries also require the new source's typed
        ``PolicyFacts``. A geometry alone cannot establish its selected resource
        extent. Retain those facts for requests and recheck them at assessment.
        """
        if type(geometries) is not tuple or any(type(g) is not ExplorationGeometry for g in geometries):
            raise ValueError("boundary freeze takes source geometries, never observations or scores")
        if (type(facts) is not tuple or any(type(f) is not PolicyFacts for f in facts)
                or len({_identity(f.shapes) for f in facts}) != len(facts)
                or any(f.shapes not in {g.operand_shapes for g in geometries} for f in facts)):
            raise ValueError("boundary facts must describe original new source geometries")
        by_shape = {_identity(f.shapes): f for f in facts}
        required = self.requests()["missing_boundary_sources"]
        existing = {g.identity for g in (*self.shape_class.design.training, *self.shape_class.design.held_out)}
        if len({g.identity for g in geometries}) != len(geometries) or any(g.identity in existing for g in geometries):
            raise ValueError("new boundary geometries must be distinct from the frozen panel")
        for geometry in geometries:
            if not any(_request_matches(r, self.shape_class.predicates, geometry,
                                         by_shape.get(geometry.identity)) for r in required):
                raise ValueError("boundary geometry needs original source/resource facts satisfying a fixed proposal request")
        old = self.shape_class.design
        design = ValidationPartition(old.training, (*old.held_out, *geometries), old.withheld_families)
        return replace(self, shape_class=replace(self.shape_class, design=design),
                       geometry_facts=(*self.geometry_facts, *facts))

    def assess(self, cases):
        """Assess this frozen winner only; do not retune alternatives on heldout."""
        if type(cases) is not tuple or any(type(c) is not ShapeCaseFeedback for c in cases):
            raise ValueError("proposal assessment requires typed current feedback")
        before = dict(self.training_feedback)
        frozen = {_identity(f.shapes): f for f in self.geometry_facts}
        for case in cases:
            if case.geometry.identity in frozen and case.facts != frozen[case.geometry.identity]:
                raise ValueError("frozen source/resource facts changed after proposal")
            if case.geometry.identity in before and _feedback_identity(case) != before[case.geometry.identity]:
                raise ValueError("training feedback changed after guard/path proposal")
        names = {v.name for v in self.alternatives}
        paired = tuple(replace(c, observations=tuple(r for r in c.observations if r[0] in names))
                       for c in cases if any(r[0] in names for r in c.observations))
        return assess_shape_policy((self.shape_class,), self.alternatives, paired,
                                   fallback=self.shape_class.incumbent,
                                   max_regression_ratio=self.max_regression_ratio)


def propose_shape_policies(training, alternatives, *, design, incumbent, axes=(), features=(),
                           max_proposals=4, max_thresholds=4, max_search_steps=2048,
                           min_support=2, max_regression_ratio=1.0):
    """Bounded coarse-to-fine 1D cutoffs and 2D conjunctions from TRAINING only.

    The objective is an equal-weight geometric mean of paired cycle ratios over
    matched training geometries. Complete numerical/ROI cycles and the original
    variant legality are prerequisites. UNKNOWN, mixed domains and regressions
    cannot supply a score. Names identify typed pass paths, never workloads.

    ``axes`` selects raw operand dimensions; ``features`` selects existing exact
    ratios/tile counts or tails. At most four facets share the same search budget.
    Missing dimensions/resources remain UNKNOWN. Rational boundary requests use
    the exact fraction and a strict outside neighbor, never integer rounding.

    No proposal is selected here. Freeze a disjoint validation design before
    tuning, satisfy ``requests``, then call ``proposal.assess``. Assessments may
    veto this training-chosen path; trying another proposal after seeing heldout
    results requires new independent validation, not reuse as tuning data.
    """
    if (type(training) is not tuple or not 1 <= len(training) <= 256
            or any(type(c) is not ShapeCaseFeedback for c in training)
            or len({c.geometry.identity for c in training}) != len(training)
            or type(design) is not ValidationPartition
            or type(alternatives) is not tuple or not 2 <= len(alternatives) <= 16
            or any(type(v) is not PipelineVariant for v in alternatives)
            or len({v.name for v in alternatives}) != len(alternatives)
            or type(axes) is not tuple or type(features) is not tuple or not 1 <= len(axes)+len(features) <= 4
            or any(type(f) not in (RatioFeature, TileFeature) for f in features)
            or len(set(features)) != len(features)
            or any(type(a) is not tuple or len(a) != 2 or any(type(i) is not int or i < 0 for i in a) for a in axes)
            or len(set(axes)) != len(axes)
            or any(type(v) is not int or not lo <= v <= hi for v, lo, hi in (
                (max_proposals, 1, 16), (max_thresholds, 2, 8), (max_search_steps, 1, 4096), (min_support, 2, 256)))
            or type(max_regression_ratio) not in (int, float) or not math.isfinite(max_regression_ratio)
            or max_regression_ratio < 1):
        raise ValueError("shape proposals require bounded training, frozen design, typed paths and dimension axes")
    design.__post_init__()
    paths = {v.name: v for v in alternatives}
    if incumbent not in paths or paths[incumbent].predicates:
        raise ValueError("proposal incumbent must be an original unconditional path")
    train_ids = {g.identity for g in design.training}
    if not {c.geometry.identity for c in training} <= train_ids:
        raise ValueError("proposal may consume TRAINING feedback only; heldout remains independent")
    for case in training:
        case.__post_init__()
        if any(n not in paths for n, _ in case.observations):
            raise ValueError("training feedback names an unregistered path")
    def facet_value(facet, case):
        if type(facet) in (RatioFeature, TileFeature):
            return facet.value(case.facts)
        operand, axis = facet
        return (case.facts.shapes[operand][axis] if operand < len(case.facts.shapes)
                and axis < len(case.facts.shapes[operand]) else None)
    values = {f: sorted({v for c in training if (v := facet_value(f, c)) is not None}) for f in (*axes, *features)}
    def facet_of(predicate):
        return predicate.feature if type(predicate) is FeaturePredicate else (predicate.operand, predicate.axis)
    def bound(facet, edge, lower=False):
        if type(facet) in (RatioFeature, TileFeature):
            value = (edge.numerator, edge.denominator)
            return FeaturePredicate(facet, **({"at_least": value} if lower else {"at_most": value}))
        return ShapePredicate(*facet, **({"at_least": edge} if lower else {"at_most": edge}))
    def minimum(facet):
        return 1 if type(facet) is tuple or type(facet) is TileFeature and facet.kind == "count" else 0
    atoms = []
    for facet, observed in values.items():
        if not observed:
            continue  # Absent dimensions or missing/zero resources remain UNKNOWN.
        indices = sorted({round(i*(len(observed)-1)/(max_thresholds-1)) for i in range(max_thresholds)})
        for index in indices:
            edge = observed[index]
            if edge > minimum(facet):
                atoms.append(bound(facet, edge, lower=True))
            atoms.append(bound(facet, edge))
    guards = [(p,) for p in atoms]
    guards += [(a, b) for a, b in combinations(atoms, 2) if facet_of(a) != facet_of(b)]
    scored, seen = [], set()
    steps = 0

    def visit(guard):
        nonlocal steps
        key = _identity([asdict(p) for p in guard])
        if key in seen or steps >= max_search_steps:
            return
        seen.add(key)
        steps += 1
        matched = [c for c in training if _match(guard, c.facts) is True]
        if len(matched) < min_support:
            return
        for path in alternatives:
            if path.name == incumbent:
                continue
            ratios, domains = [], set()
            for case in matched:
                pair = dict(case.observations)
                control, candidate = pair.get(incumbent), pair.get(path.name)
                if (control is None or candidate is None or not control.eligible or not candidate.eligible
                        or _match(path.predicates, case.facts) is not True):
                    break
                ratios.append(candidate.cycles/control.cycles)
                domains.add(PerformanceDomain.from_observation(candidate.identity,
                    implementation_sha256=case.implementation_sha256))
            if len(ratios) != len(matched) or len(domains) != 1 or max(ratios) > max_regression_ratio:
                continue
            score = math.exp(sum(math.log(r) for r in ratios)/len(ratios))
            if score < 1:
                scored.append((score, -len(matched), len(guard), key, path.name, guard, next(iter(domains))))

    for guard in guards:
        visit(guard)
    # Refine neighboring observed thresholds around the best coarse guards.
    # Heldout geometry and scores never enter this search or its ordering.
    for row in sorted(scored)[:8]:
        guard = row[5]
        for index, predicate in enumerate(guard):
            facet = facet_of(predicate)
            observed = values[facet]
            edge = predicate.at_most if predicate.at_most is not None else predicate.at_least
            if type(predicate) is FeaturePredicate:
                edge = Fraction(*edge)
            at = observed.index(edge)
            for neighbor in observed[max(0, at-1):at+2]:
                if predicate.at_most is None and neighbor <= minimum(facet):
                    continue
                refined = bound(facet, neighbor, lower=predicate.at_most is None)
                visit(guard[:index] + (refined,) + guard[index+1:])
    proposals = []
    for score, support, _count, key, candidate, guard, domain in sorted(scored)[:max_proposals]:
        group = ShapeScheduleClass("proposed_" + key[:16], guard, incumbent, design)
        pair = (paths[incumbent], paths[candidate])
        identity = _identity((key, [(v.name, v.passes, [asdict(p) for p in v.predicates]) for v in pair],
                              design.manifest()["sha256"], [_feedback_identity(c) for c in training],
                              max_regression_ratio))
        report = dict(schema="merlin.shape_policy_proposal.v1", proposal_sha256=identity,
            status="training_proposal_not_selected", candidate=candidate, incumbent=incumbent,
            guards=[asdict(p) for p in guard], matched_training_cases=-support,
            training_geometric_mean_cycle_ratio=score, domain=asdict(domain),
            max_regression_ratio=max_regression_ratio,
            objective="equal weight per matched training geometry; paired complete-cycle geometric mean ratio",
            search_strategy="observed exact dimension/feature quantiles, 2D conjunctions, then neighboring training thresholds",
            search_steps=steps, search_limit=max_search_steps, search_truncated=steps >= max_search_steps,
            validation_design=design.manifest(), training_only=True, performance_qualified=False)
        proposals.append(ShapePolicyProposal(group, pair, candidate, domain,
            tuple((c.geometry.identity, _feedback_identity(c)) for c in training), report,
            max_regression_ratio=max_regression_ratio, geometry_facts=tuple(c.facts for c in training)))
    return tuple(proposals)
