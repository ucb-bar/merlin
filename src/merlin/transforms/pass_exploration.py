"""Bounded pass-order and option exploration over the ordinary pass manager.

Callers own transformations, legality facts and emission. This module generates
legal registered alternatives, executes real passes, and deduplicates identical
emitted programs. Counts and host profiles are not target performance scores.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from itertools import product
from types import MappingProxyType

from .pipeline_policy import PipelineVariant, RegisteredPass, evaluate_variants
from .pipeline_state import validate_pass_order


@dataclass(frozen=True)
class ExplorationGeometry:
    """Source geometry for validation grouping, without workload identifiers.

    Numeric formats, epilogues and aliases do not make an otherwise identical
    geometry a new holdout. The family separately reserves connected graphs.
    This metadata is neither program IR nor a legality/capability declaration.
    """
    family: str
    operand_shapes: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        if (type(self.family) is not str or not self.family or len(self.family) > 128
                or type(self.operand_shapes) is not tuple or not 1 <= len(self.operand_shapes) <= 64
                or any(type(shape) is not tuple or not 1 <= len(shape) <= 8
                       or any(type(d) is not int or not 0 < d < 2**63 for d in shape)
                       for shape in self.operand_shapes)):
            raise ValueError('exploration geometry needs a family and bounded complete static operand shapes')

    @property
    def identity(self):
        return _identity(self.operand_shapes)


@dataclass(frozen=True)
class ValidationPartition:
    """Freeze disjoint source geometries and optionally whole withheld families.

    Group aliases before constructing this record. It describes a prospective
    split for this experiment, not a claim of campaign-global unseen workloads.
    """
    training: tuple[ExplorationGeometry, ...]
    held_out: tuple[ExplorationGeometry, ...]
    withheld_families: tuple[str, ...] = ()

    def __post_init__(self):
        for values in (self.training, self.held_out):
            if (type(values) is not tuple or not 1 <= len(values) <= 256
                    or any(type(value) is not ExplorationGeometry for value in values)
                    or len({value.identity for value in values}) != len(values)):
                raise ValueError('validation partitions need distinct grouped source geometries')
        if {value.identity for value in self.training} & {value.identity for value in self.held_out}:
            raise ValueError('training and holdout overlap in geometry; aliases are not generalization')
        if (type(self.withheld_families) is not tuple
                or any(type(name) is not str or not name for name in self.withheld_families)
                or len(set(self.withheld_families)) != len(self.withheld_families)
                or set(self.withheld_families) & {value.family for value in self.training}
                or set(self.withheld_families) - {value.family for value in self.held_out}):
            raise ValueError('whole withheld families must occur only in the holdout')

    def manifest(self):
        def rows(values):
            return [dict(geometry_sha256=value.identity, family=value.family,
                         operand_shapes=value.operand_shapes) for value in values]

        record = dict(schema='merlin.pass_exploration.validation_partition.v1',
                      training=rows(self.training), held_out=rows(self.held_out),
                      withheld_families=self.withheld_families)
        return {**record, 'sha256': _identity(record)}


@dataclass(frozen=True)
class ObservedMetrics:
    """Explicit same-domain observations; unknown values never become zero.

    Static observations may screen emitted work. They are not device latency,
    measured utilization or performance qualification. Timing domains must be
    supplied by their actual provider and scope, not inferred from shape.
    """
    basis: str
    domain: str
    values: tuple[tuple[str, int | float | None], ...]

    def __post_init__(self):
        if (self.basis not in ('static_emitted_ir', 'modeled_roi', 'measured_roi')
                or type(self.domain) is not str or not self.domain or len(self.domain) > 4096
                or type(self.values) is not tuple or not 1 <= len(self.values) <= 64
                or any(type(row) is not tuple or len(row) != 2 for row in self.values)
                or any(type(name) is not str or not name for name, _ in self.values)
                or len({name for name, _ in self.values}) != len(self.values)
                or any(type(name) is not str or not name or value is not None
                       and (type(value) not in (int, float) or value < 0
                            or type(value) is float and not math.isfinite(value))
                       for name, value in self.values)):
            raise ValueError('observed metrics require one explicit domain and finite nonnegative values or unknown')


def rank_observed_metrics(observations, *, objectives):
    """Lexicographic screening under caller-selected objectives, with exclusions.

    The return value records unknown/missing objectives separately. It grants
    neither legality nor speedup; byte-identical controls can use the existing
    emitted-program dedup before this optional comparison.
    """
    if (type(observations) is not tuple or not 1 <= len(observations) <= 256
            or type(objectives) is not tuple or not 1 <= len(objectives) <= 64
            or any(type(name) is not str or not name for name in objectives)
            or len(set(objectives)) != len(objectives)
            or any(type(row) is not tuple or len(row) != 2 or type(row[0]) is not str or not row[0]
                   or type(row[1]) is not ObservedMetrics for row in observations)
            or len({name for name, _ in observations}) != len(observations)):
        raise ValueError('screening requires bounded unique alternatives and explicit metric objectives')
    if len({(metrics.basis, metrics.domain) for _, metrics in observations}) != 1:
        raise ValueError('different observation providers/scopes/domains cannot share a ranking')
    ranked, unknown = [], []
    for name, metrics in observations:
        values = dict(metrics.values)
        missing = tuple(key for key in objectives if values.get(key) is None)
        if missing:
            unknown.append((name, missing))
        else:
            ranked.append((name, tuple(values[key] for key in objectives)))
    return dict(basis=observations[0][1].basis, domain=observations[0][1].domain,
                objectives=objectives, ranked=tuple(sorted(ranked, key=lambda row: (row[1], row[0]))),
                unranked_unknown=tuple(unknown), performance_qualification=False)


def rank_performance_variants(observations):
    """Order same-domain cycle observations, without a static-count fallback.

    The caller retains numerical checks and source/model/window custody. Modeled
    latency is useful for exploration without becoming a hardware measurement.
    Utilization, traffic and instruction counts explain a decision; they cannot
    replace missing cycles or override a slower complete-call observation.
    """
    if (type(observations) is not tuple or not observations
            or any(type(row) is not tuple or len(row) != 2
                   or type(row[1]) is not ObservedMetrics for row in observations)):
        raise ValueError('performance ranking requires explicit cycle observations')
    if any(metrics.basis not in ('modeled_roi', 'measured_roi') for _, metrics in observations):
        raise ValueError('static instruction or command counts cannot rank performance')
    result = rank_observed_metrics(observations, objectives=('cycles',))
    result.update(
        decision_metric='complete_same_roi_cycles',
        provisional=observations[0][1].basis == 'modeled_roi',
        diagnostic_metrics=('utilization', 'roofline_gap', 'resource_activity',
                            'traffic', 'commands', 'instructions'),
        missing_cycles_policy='unranked; no instruction, command or utilization fallback',
    )
    return result


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class PassFamily:
    """One editable pass factory with explicit finite option domains.

    The factory returns the real RegisteredPass. Its prerequisites, stage and
    invalidations remain authoritative. Predicates use existing semantic facts,
    never a capsule/model name. Options do not confer legality or capability.
    """
    name: str
    bind: Callable[[Mapping], RegisteredPass]
    knobs: tuple[tuple[str, tuple[bool | int | str, ...]], ...] = ()
    predicates: tuple | Callable[[Mapping], tuple] = ()
    optional: bool = False

    def __post_init__(self):
        if (not isinstance(self.name, str) or not self.name or len(self.name) > 128 or not callable(self.bind)
                or type(self.knobs) is not tuple or len(self.knobs) > 8
                or type(self.optional) is not bool
                or not (type(self.predicates) is tuple or callable(self.predicates))):
            raise ValueError('pass families require a callable, bounded knobs and explicit predicates')
        names = set()
        for name, values in self.knobs:
            if (not isinstance(name, str) or not name or name in names or type(values) is not tuple
                    or not 1 <= len(values) <= 16
                    or any(type(v) not in (bool, int, str) or isinstance(v, str) and len(v) > 128 for v in values)
                    or len({_identity(v) for v in values}) != len(values)):
                raise ValueError('knob domains require unique bounded scalar values')
            names.add(name)


@dataclass(frozen=True)
class ExplorationSpace:
    variants: tuple[PipelineVariant, ...]
    registry: Mapping[str, RegisteredPass]
    bindings: Mapping[str, dict]
    search_steps: int
    truncated: bool


def generate_candidates(families: tuple[PassFamily, ...], *, output: str,
                        before: tuple[tuple[str, str], ...] = (),
                        max_candidates: int = 8, max_search_steps: int = 4096) -> ExplorationSpace:
    """Enumerate option bindings and legal topological pass orders, bounded.

    Existing stage, analysis-dependency and invalidation validation prunes each
    prefix before a pass runs. Explicit order edges add caller-owned semantic
    constraints. Every completed path must produce its requested final artifact.
    No factory is called per permutation; each option binding is registered once.
    """
    if (type(families) is not tuple or not 1 <= len(families) <= 16
            or any(type(f) is not PassFamily for f in families)
            or len({f.name for f in families}) != len(families)
            or type(max_candidates) is not int or not 1 <= max_candidates <= 16
            or type(max_search_steps) is not int or not 1 <= max_search_steps <= 65536
            or not isinstance(output, str) or not output):
        raise ValueError('exploration requires bounded unique families and an explicit final output')
    names = {f.name for f in families}
    if (type(before) is not tuple or len(before) > 128 or any(
            type(edge) is not tuple or len(edge) != 2 or edge[0] not in names
            or edge[1] not in names or edge[0] == edge[1] for edge in before)):
        raise ValueError('order edges must name distinct registered families')
    registry, bindings, domains, guards = {}, {}, [], {}
    steps = 0
    exhausted = False
    for family in families:
        domain = [None] if family.optional else []
        for values in product(*(v for _, v in family.knobs)):
            steps += 1
            if steps > max_search_steps or len(registry) >= 256:
                raise ValueError('pass option registration exceeds its finite work budget')
            options = dict(zip((k for k, _ in family.knobs), values, strict=True))
            bound = family.bind(MappingProxyType(options))
            if type(bound) is not RegisteredPass:
                raise ValueError('pass factory did not return its actual registered implementation')
            key = family.name + '_' + _identity(options)[:16]
            entry = RegisteredPass(replace(bound.spec, name=key), bound.action)
            registry[key] = entry
            bindings[key] = dict(family=family.name, registered_pass=bound.spec.name, knobs=options)
            guards[key] = (family.predicates(MappingProxyType(options))
                           if callable(family.predicates) else family.predicates)
            if type(guards[key]) is not tuple:
                raise ValueError('bound applicability predicates must be an explicit tuple')
            domain.append((key, family))
        domains.append(tuple(domain))
    variants = []

    def orders(selected, prefix=()):
        nonlocal steps, exhausted
        if steps >= max_search_steps:
            exhausted = True
            return
        steps += 1
        if not selected:
            if prefix and output in registry[prefix[-1]].spec.produces:
                yield prefix
            return
        stage = min(registry[key].spec.stage for key, _ in selected)
        remaining = {family.name for _, family in selected}
        for position, (key, family) in enumerate(selected):
            if registry[key].spec.stage != stage or any(
                    right == family.name and left in remaining for left, right in before):
                continue
            path = (*prefix, key)
            try:
                validate_pass_order(tuple(registry[name].spec for name in path))
            except ValueError:
                continue
            yield from orders(selected[:position] + selected[position + 1:], path)

    for selection in product(*domains):
        chosen = tuple(v for v in selection if v is not None)
        for path in orders(chosen):
            predicates = tuple(p for key, _ in chosen for p in guards[key])
            variant = PipelineVariant('candidate_' + _identity(path)[:16], path, predicates)
            variants.append(variant)
            if len(variants) >= max_candidates:
                return ExplorationSpace(tuple(variants), MappingProxyType(registry),
                                        MappingProxyType(bindings), steps, True)
        if exhausted:
            break
    if not variants:
        raise ValueError('no legal pipeline produces the requested artifact within the search budget')
    return ExplorationSpace(tuple(variants), MappingProxyType(registry), MappingProxyType(bindings), steps, exhausted)


@dataclass(frozen=True)
class EmittedProgram:
    """Exact emitted bytes at an explicit boundary, not a predicted schedule."""
    data: bytes
    format: str
    schedule_sha256: str

    def __post_init__(self):
        if (type(self.data) is not bytes or not self.data or not isinstance(self.format, str)
                or not self.format or len(self.format) > 64
                or type(self.schedule_sha256) is not str or len(self.schedule_sha256) != 64
                or any(c not in '0123456789abcdef' for c in self.schedule_sha256)):
            raise ValueError('emission requires actual nonempty bytes, format and transformed-schedule identity')


@dataclass(frozen=True)
class ExploredCandidate:
    evaluation: object
    bindings: tuple[dict, ...]
    emission: EmittedProgram | None
    duplicate_of: str | None
    feedback: Mapping | None

    def summary(self):
        return dict(self.evaluation.summary(), bindings=self.bindings,
                    program_sha256=hashlib.sha256(self.emission.data).hexdigest() if self.emission else None,
                    program_format=self.emission.format if self.emission else None,
                    transformed_schedule_sha256=self.emission.schedule_sha256 if self.emission else None,
                    duplicate_of=self.duplicate_of,
                    target_feedback=dict(self.feedback) if self.feedback else None)


def explore_candidates(module, space: ExplorationSpace, facts: Callable, *, output: str,
                       emission: Callable, feedback: Callable | None = None, **session_options):
    """Yield real compilation results immediately; submit unique programs once.

    The emission callback must serialize the artifact actually produced by the
    selected passes and retain its real schedule identity. Feedback is optional
    caller-observed data, joined to the exact program hash; there is no default
    score, simulator, ranking, approval or qualification. Duplicate emissions
    retain their distinct pass profiles but do not repeat the feedback callback.
    """
    if type(space) is not ExplorationSpace or not callable(emission) or feedback is not None and not callable(feedback):
        raise ValueError('explicit generated space and actual-emission adapter required')
    seen = {}
    for result in evaluate_variants(module, space.variants, space.registry, facts,
                                    output=output, **session_options):
        emitted, duplicate, observed = None, None, None
        variant = next(v for v in space.variants if v.name == result.variant)
        if result.status == 'compiled':
            emitted = emission(result.result)
            if type(emitted) is not EmittedProgram:
                raise ValueError('emission adapter did not return actual program bytes')
            digest = hashlib.sha256(emitted.data).hexdigest()
            key = (digest, emitted.format, _identity(result.result.receipt['after']['context']))
            duplicate = seen.get(key)
            if duplicate is None:
                seen[key] = result.variant
                if feedback is not None:
                    observed = feedback(emitted, result.result.receipt)
                    if not isinstance(observed, Mapping) or observed.get('program_sha256') != digest:
                        raise ValueError('feedback does not match the actual emitted program')
        yield ExploredCandidate(result, tuple(space.bindings[name] for name in variant.passes),
                                emitted, duplicate, observed)
