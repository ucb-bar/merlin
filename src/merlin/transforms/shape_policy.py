"""Empirical shape/resource guards over real registered pipeline alternatives.

This optional selector never establishes legality, executes a simulator or
reuses a training kernel for another source. It retains incumbent pass paths
until exact paired feedback covers the declared training, heldout and cutoff
boundaries. Ordinary current-IR pass checks still govern every selected path.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .evaluated_exploration import CompletedObservation, ExplorationIdentity, _pin, exploration_context_identity
from .pass_exploration import ExplorationGeometry, ValidationPartition, _identity
from .pipeline_policy import (
    GraphPredicate,
    PipelinePolicy,
    PipelineVariant,
    PolicyFacts,
    PropertyPredicate,
    ResourcePredicate,
    ShapePredicate,
    execute_policy,
)

_PREDICATES = (ShapePredicate, PropertyPredicate, ResourcePredicate, GraphPredicate)


def _match(predicates, facts):
    result = tuple(p.match(facts) for p in predicates)
    return None if any(v is None for v in result) else all(result)


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
    """
    geometry: ExplorationGeometry
    facts: PolicyFacts
    observations: tuple[tuple[str, CompletedObservation], ...]
    implementation_sha256: str

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
    return tuple(required)


def _value(predicate, facts):
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
                           eligible=o.eligible, cycles=o.cycles) for n, o in c.observations],
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
                outside = edge-1 if side == "lower" else edge+1
                others = group.predicates[:index] + group.predicates[index+1:]
                values = {_value(predicate, c.facts) for c in panel if _match(others, c.facts) is True}
                if edge not in values or outside not in values:
                    boundary_missing.append((index, side, edge, outside))
            eligible = (not missing and not boundary_missing
                and any(c.geometry.identity in training for c in matched)
                and any(c.geometry.identity in held for c in matched))
            for candidate in alternatives:
                ratios, training_ratios, reasons, domains = [], [], [], set()
                for case in panel:
                    if (_match(group.predicates, case.facts) is None
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
            legality="original variant predicates and current-IR registered pass checks remain required",
            qualification="provisional empirical panel support, not universal or physical qualification"))
