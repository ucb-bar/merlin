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

from merlin.compiler.transforms.exploration.branch_policy import (
    POLICY_PREDICATE_TYPES,
    DimensionRef,
    FeaturePredicate,
    GraphPredicate,
    PipelinePolicy,
    PipelineVariant,
    PolicyFacts,
    RatioFeature,
    ResourcePredicate,
    ShapePredicate,
    TileFeature,
    execute_policy,
)
from merlin.compiler.transforms.exploration.cycle_feedback import (
    CompletedObservation,
    ExplorationIdentity,
    _pin,
    exploration_context_identity,
)
from merlin.compiler.transforms.exploration.pass_orders import (
    ExplorationGeometry,
    ObservedMetrics,
    ValidationPartition,
    _identity,
)
from merlin.compiler.transforms.exploration.shape_assessment import (
    PerformanceDomain,
    ShapeCaseFeedback,
    ShapeScheduleClass,
    _boundaries,
    _feedback_identity,
    _geometry_match,
    _integer_boundary,
    _match,
    _request_matches,
    assess_shape_policy,
)

# ---- shape_policy ------------------------------------------------------------------------------


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
        if (
            type(self.shape_class) is not ShapeScheduleClass
            or self.shape_class.design is None
            or any(type(p) not in (ShapePredicate, FeaturePredicate) for p in self.shape_class.predicates)
            or type(self.alternatives) is not tuple
            or len(self.alternatives) != 2
            or any(type(v) is not PipelineVariant for v in self.alternatives)
            or {v.name for v in self.alternatives} != {self.shape_class.incumbent, self.candidate}
            or type(self.domain) is not PerformanceDomain
            or type(self.training_feedback) is not tuple
            or any(
                type(row) is not tuple or len(row) != 2 or not all(_pin(v) for v in row)
                for row in self.training_feedback
            )
            or type(self.report) is not dict
            or not _pin(self.report.get("proposal_sha256"))
        ):
            raise ValueError("proposal requires its frozen class, typed paths, domain and training identities")
        roster = {g.identity for g in (*self.shape_class.design.training, *self.shape_class.design.held_out)}
        if (
            type(self.geometry_facts) is not tuple
            or any(type(f) is not PolicyFacts for f in self.geometry_facts)
            or len({_identity(f.shapes) for f in self.geometry_facts}) != len(self.geometry_facts)
            or any(_identity(f.shapes) not in roster for f in self.geometry_facts)
        ):
            raise ValueError("proposal facts must bind distinct original frozen geometries")
        if (
            type(self.max_regression_ratio) not in (int, float)
            or not math.isfinite(self.max_regression_ratio)
            or self.max_regression_ratio < 1
        ):
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
            needed = (
                (group.incumbent, self.candidate)
                if _geometry_match(group.predicates, geometry, facts.get(geometry.identity)) is not False
                else (group.incumbent,)
            )
            missing = []
            for name in needed:
                observed = None if case is None else dict(case.observations).get(name)
                if (
                    case is None
                    or observed is None
                    or not observed.eligible
                    or _match(paths[name].predicates, case.facts) is not True
                    or PerformanceDomain.from_observation(
                        observed.identity, implementation_sha256=case.implementation_sha256
                    )
                    != self.domain
                ):
                    missing.append(name)
            if missing:
                pending.append(
                    dict(
                        geometry_sha256=geometry.identity,
                        family=geometry.family,
                        operand_shapes=geometry.operand_shapes,
                        alternatives=missing,
                        partition="training" if geometry in design.training else "held_out",
                    )
                )
        panel = (*design.training, *design.held_out)
        for index, side, edge in _boundaries(group.predicates):
            predicate = group.predicates[index]
            others = group.predicates[:index] + group.predicates[index + 1 :]
            integer = _integer_boundary(predicate, edge)
            targets = (
                ((edge, "equal"), (edge - 1 if side == "lower" else edge + 1, "equal"))
                if integer
                else ((edge, "equal"), (edge, "below" if side == "lower" else "above"))
            )
            for value, comparison in targets:
                request = dict(
                    predicate_index=index,
                    side=side,
                    value=int(value) if integer else dict(numerator=value.numerator, denominator=value.denominator),
                    other_guards=[asdict(p) for p in others],
                    reason="materialize a legal source with this boundary value, then freeze its geometry",
                )
                if type(predicate) is ShapePredicate:
                    request.update(operand=predicate.operand, axis=predicate.axis)
                else:
                    request.update(
                        feature=asdict(predicate.feature),
                        comparison=comparison,
                        source_facts_required="original selected resource facts for tile/resource features",
                    )
                if not any(_request_matches(request, group.predicates, g, facts.get(g.identity)) for g in panel):
                    boundaries.append(request)
        return dict(
            schema="merlin.shape_policy_requests.v1",
            proposal_sha256=self.report["proposal_sha256"],
            pending_evaluations=pending,
            missing_boundary_sources=boundaries,
            no_execution_performed=True,
            selected=False,
        )

    def freeze_boundaries(self, geometries, *, facts=()):
        """Extend the heldout roster before timing, without constructing source.

        Tile/resource feature boundaries also require the new source's typed
        ``PolicyFacts``. A geometry alone cannot establish its selected resource
        extent. Retain those facts for requests and recheck them at assessment.
        """
        if type(geometries) is not tuple or any(type(g) is not ExplorationGeometry for g in geometries):
            raise ValueError("boundary freeze takes source geometries, never observations or scores")
        if (
            type(facts) is not tuple
            or any(type(f) is not PolicyFacts for f in facts)
            or len({_identity(f.shapes) for f in facts}) != len(facts)
            or any(f.shapes not in {g.operand_shapes for g in geometries} for f in facts)
        ):
            raise ValueError("boundary facts must describe original new source geometries")
        by_shape = {_identity(f.shapes): f for f in facts}
        required = self.requests()["missing_boundary_sources"]
        existing = {g.identity for g in (*self.shape_class.design.training, *self.shape_class.design.held_out)}
        if len({g.identity for g in geometries}) != len(geometries) or any(g.identity in existing for g in geometries):
            raise ValueError("new boundary geometries must be distinct from the frozen panel")
        for geometry in geometries:
            if not any(
                _request_matches(r, self.shape_class.predicates, geometry, by_shape.get(geometry.identity))
                for r in required
            ):
                raise ValueError(
                    "boundary geometry needs original source/resource facts satisfying a fixed proposal request"
                )
        old = self.shape_class.design
        design = ValidationPartition(old.training, (*old.held_out, *geometries), old.withheld_families)
        return replace(
            self, shape_class=replace(self.shape_class, design=design), geometry_facts=(*self.geometry_facts, *facts)
        )

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
        paired = tuple(
            replace(c, observations=tuple(r for r in c.observations if r[0] in names))
            for c in cases
            if any(r[0] in names for r in c.observations)
        )
        return assess_shape_policy(
            (self.shape_class,),
            self.alternatives,
            paired,
            fallback=self.shape_class.incumbent,
            max_regression_ratio=self.max_regression_ratio,
        )


def propose_shape_policies(
    training,
    alternatives,
    *,
    design,
    incumbent,
    axes=(),
    features=(),
    max_proposals=4,
    max_thresholds=4,
    max_search_steps=2048,
    min_support=2,
    max_regression_ratio=1.0,
):
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
    from merlin.compiler.transforms.exploration.shape_assessment import (
        PerformanceDomain,
        ShapeCaseFeedback,
        ShapeScheduleClass,
        _feedback_identity,
        _match,
    )

    if (
        type(training) is not tuple
        or not 1 <= len(training) <= 256
        or any(type(c) is not ShapeCaseFeedback for c in training)
        or len({c.geometry.identity for c in training}) != len(training)
        or type(design) is not ValidationPartition
        or type(alternatives) is not tuple
        or not 2 <= len(alternatives) <= 16
        or any(type(v) is not PipelineVariant for v in alternatives)
        or len({v.name for v in alternatives}) != len(alternatives)
        or type(axes) is not tuple
        or type(features) is not tuple
        or not 1 <= len(axes) + len(features) <= 4
        or any(type(f) not in (RatioFeature, TileFeature) for f in features)
        or len(set(features)) != len(features)
        or any(type(a) is not tuple or len(a) != 2 or any(type(i) is not int or i < 0 for i in a) for a in axes)
        or len(set(axes)) != len(axes)
        or any(
            type(v) is not int or not lo <= v <= hi
            for v, lo, hi in (
                (max_proposals, 1, 16),
                (max_thresholds, 2, 8),
                (max_search_steps, 1, 4096),
                (min_support, 2, 256),
            )
        )
        or type(max_regression_ratio) not in (int, float)
        or not math.isfinite(max_regression_ratio)
        or max_regression_ratio < 1
    ):
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
        return (
            case.facts.shapes[operand][axis]
            if operand < len(case.facts.shapes) and axis < len(case.facts.shapes[operand])
            else None
        )

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
        indices = sorted({round(i * (len(observed) - 1) / (max_thresholds - 1)) for i in range(max_thresholds)})
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
                if (
                    control is None
                    or candidate is None
                    or not control.eligible
                    or not candidate.eligible
                    or _match(path.predicates, case.facts) is not True
                ):
                    break
                ratios.append(candidate.cycles / control.cycles)
                domains.add(
                    PerformanceDomain.from_observation(
                        candidate.identity, implementation_sha256=case.implementation_sha256
                    )
                )
            if len(ratios) != len(matched) or len(domains) != 1 or max(ratios) > max_regression_ratio:
                continue
            score = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
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
            for neighbor in observed[max(0, at - 1) : at + 2]:
                if predicate.at_most is None and neighbor <= minimum(facet):
                    continue
                refined = bound(facet, neighbor, lower=predicate.at_most is None)
                visit(guard[:index] + (refined,) + guard[index + 1 :])
    proposals = []
    for score, support, _count, key, candidate, guard, domain in sorted(scored)[:max_proposals]:
        group = ShapeScheduleClass("proposed_" + key[:16], guard, incumbent, design)
        pair = (paths[incumbent], paths[candidate])
        identity = _identity(
            (
                key,
                [(v.name, v.passes, [asdict(p) for p in v.predicates]) for v in pair],
                design.manifest()["sha256"],
                [_feedback_identity(c) for c in training],
                max_regression_ratio,
            )
        )
        report = dict(
            schema="merlin.shape_policy_proposal.v1",
            proposal_sha256=identity,
            status="training_proposal_not_selected",
            candidate=candidate,
            incumbent=incumbent,
            guards=[asdict(p) for p in guard],
            matched_training_cases=-support,
            training_geometric_mean_cycle_ratio=score,
            domain=asdict(domain),
            max_regression_ratio=max_regression_ratio,
            objective="equal weight per matched training geometry; paired complete-cycle geometric mean ratio",
            search_strategy="observed exact dimension/feature quantiles, 2D conjunctions, then neighboring training thresholds",
            search_steps=steps,
            search_limit=max_search_steps,
            search_truncated=steps >= max_search_steps,
            validation_design=design.manifest(),
            training_only=True,
            performance_qualified=False,
        )
        proposals.append(
            ShapePolicyProposal(
                group,
                pair,
                candidate,
                domain,
                tuple((c.geometry.identity, _feedback_identity(c)) for c in training),
                report,
                max_regression_ratio=max_regression_ratio,
                geometry_facts=tuple(c.facts for c in training),
            )
        )
    return tuple(proposals)
