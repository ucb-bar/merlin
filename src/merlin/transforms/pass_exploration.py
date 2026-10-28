"""Bounded pass-order and option exploration over the ordinary pass manager.

Callers own transformations, legality facts and emission. This module generates
legal registered alternatives, executes real passes, and deduplicates identical
emitted programs. Counts and host profiles are not target performance scores.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from itertools import product
from threading import Lock
from types import MappingProxyType

from .pipeline_policy import PipelineVariant, RegisteredPass, evaluate_variants
from .pipeline_state import validate_pass_order


class ExplorationRefused(ValueError):
    """Explicit expected construction/emission refusal; not an infrastructure error.

    Adapters may raise this for a known unsupported realization. Other adapter
    exceptions propagate. The bounded reason records the caller's actual refusal,
    never a diagnosis inferred from an exception's wording.
    """

    def __init__(self, reason, *, layer, evidence=()):
        if (layer not in ('construction', 'adapter', 'emission')
                or type(reason) is not str or not reason or type(evidence) is not tuple):
            raise ValueError('exploration refusal needs an explicit owner and reason')
        super().__init__(reason)
        if any(type(row) is not dict or set(row) != {'passes', 'reason', 'owner'}
               or type(row['passes']) is not tuple or len(row['passes']) > 16
               or any(type(name) is not str or len(name) > 256 for name in row['passes'])
               or type(row['reason']) is not str or len(row['reason']) > 512
               or type(row['owner']) is not str or len(row['owner']) > 128 for row in evidence[:16]):
            raise ValueError('exploration refusal evidence requires bounded pass-order records')
        self.layer, self.evidence = layer, tuple(dict(row) for row in evidence[:16])

    def summary(self):
        return dict(status='refused', layer=self.layer, reason=str(self)[:512], evidence=self.evidence)


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
class MappingProposalKey:
    """Caller-selected mapping domain, shared across independent source instances.

    semantic_sha256 covers operation attributes (e.g. stride/padding), never a
    model/case name. Include every relevant input/output shape, layout and dtype;
    numerical_sha256 binds the exact applicable numerical policy. These pins
    select cached proposals only: no cached legality, proof or timing follows.
    """
    semantic_sha256: str
    shapes: tuple[tuple[int, ...], ...]
    layouts: tuple[str, ...]
    dtypes: tuple[str, ...]
    epilogue: tuple[str, ...]
    hardware_sha256: str
    library_sha256: str
    numerical_sha256: str

    def __post_init__(self):
        for value in (self.semantic_sha256, self.hardware_sha256, self.library_sha256, self.numerical_sha256):
            if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('mapping proposal domain requires exact semantic/hardware/library/numerical pins')
        if (type(self.shapes) is not tuple or not 1 <= len(self.shapes) <= 64
                or any(type(shape) is not tuple or not 1 <= len(shape) <= 8
                       or any(type(d) is not int or not 0 < d < 2**63 for d in shape) for shape in self.shapes)
                or any(type(values) is not tuple or len(values) != len(self.shapes)
                       or any(type(value) is not str or not 1 <= len(value) <= 128 for value in values)
                       for values in (self.layouts, self.dtypes))
                or type(self.epilogue) is not tuple or len(self.epilogue) > 32
                or any(type(value) is not str or not 1 <= len(value) <= 128 for value in self.epilogue)):
            raise ValueError('mapping proposal domain needs complete bounded shapes/layouts/dtypes/epilogue')

    @property
    def identity(self):
        self.__post_init__()
        return _identity(asdict(self))


class MappingProposalCache:
    """Explicit bounded memory for option paths, owned by one compilation run.

    No globals, disk, IR, pass callbacks, target proofs, emitted programs or
    measurements are retained. Every use rebinds current factories and checks
    their current order/dependency/output contracts; actual passes still execute.
    Independent compiler runs get independent instances. The small record lock
    permits a caller to share proposals across its concurrent shape tasks.
    """
    def __init__(self, *, max_entries=32):
        if type(max_entries) is not int or not 1 <= max_entries <= 256:
            raise ValueError('mapping proposal cache requires a bounded entry count')
        self.max_entries = max_entries
        self._records, self._lock = OrderedDict(), Lock()
        self.hits = self.misses = self.invalidations = 0

    def _get(self, key):
        with self._lock:
            value = self._records.get(key)
            if value is None:
                self.misses += 1
            else:
                self._records.move_to_end(key)
            return value

    def _hit(self):
        with self._lock:
            self.hits += 1

    def _discard(self, key):
        with self._lock:
            self._records.pop(key, None)
            self.invalidations += 1
            self.misses += 1

    def _put(self, key, paths, truncated):
        # Private storage accepts only generated option keys, never caller data.
        if (type(paths) is not tuple or not 1 <= len(paths) <= 16
                or any(type(path) is not tuple or not 1 <= len(path) <= 16
                       or any(type(name) is not str or not 1 <= len(name) <= 145 for name in path)
                       for path in paths) or type(truncated) is not bool):
            raise ValueError('mapping proposal cache only retains bounded option paths')
        with self._lock:
            self._records[key] = (paths, truncated)
            self._records.move_to_end(key)
            while len(self._records) > self.max_entries:
                self._records.popitem(last=False)

    def __len__(self):
        with self._lock:
            return len(self._records)


@dataclass(frozen=True)
class ExplorationSpace:
    variants: tuple[PipelineVariant, ...]
    registry: Mapping[str, RegisteredPass]
    bindings: Mapping[str, dict]
    search_steps: int
    truncated: bool
    refusals: tuple[dict, ...] = ()
    proposal_cache_hit: bool = False


def generate_candidates(families: tuple[PassFamily, ...], *, output: str,
                        before: tuple[tuple[str, str], ...] = (),
                        max_candidates: int = 8, max_search_steps: int = 4096,
                        proposal_cache: MappingProposalCache | None = None,
                        proposal_key: MappingProposalKey | None = None) -> ExplorationSpace:
    """Enumerate option bindings and legal topological pass orders, bounded.

    Existing stage, analysis-dependency and invalidation validation prunes each
    prefix before a pass runs. Explicit order edges add caller-owned semantic
    constraints. Every completed path must produce its requested final artifact.
    No factory is called per permutation; each option binding is registered once.
    Optional proposal reuse avoids repeated topological search, never current
    factory/predicate binding or legality checks. The current domain/limits are
    part of the cache key. Every resulting pass still runs on the current IR.
    Existing evaluate_variants(prefix=...) remains the common-preparation path.
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
    if ((proposal_cache is None) != (proposal_key is None)
            or proposal_cache is not None and (type(proposal_cache) is not MappingProposalCache
                                               or type(proposal_key) is not MappingProposalKey)):
        raise ValueError('mapping proposal reuse requires an explicit cache and complete domain key')
    cache_key = None if proposal_cache is None else _identity((proposal_key.identity,
        [(family.name, family.knobs, family.optional) for family in families],
        output, before, max_candidates, max_search_steps))
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
    if proposal_cache is not None:
        cached = proposal_cache._get(cache_key)
        if cached is not None:
            paths, truncated = cached
            try:
                for path in paths:
                    validate_pass_order(tuple(registry[key].spec for key in path))
                    if output not in registry[path[-1]].spec.produces:
                        raise ValueError('current proposal no longer produces the requested output')
            except (ValueError, KeyError):
                proposal_cache._discard(cache_key)
            else:
                # Preserve original family-order predicate conjunction while
                # obtaining every predicate from the CURRENT factory binding.
                family_order = {family.name: index for index, family in enumerate(families)}
                variants = tuple(PipelineVariant('candidate_' + _identity(path)[:16], path,
                    tuple(predicate for key in sorted(path,
                        key=lambda key: family_order[bindings[key]['family']]) for predicate in guards[key]))
                    for path in paths)
                proposal_cache._hit()
                return ExplorationSpace(variants, MappingProxyType(registry), MappingProxyType(bindings),
                    steps, truncated, proposal_cache_hit=True)

    def finish(variants, truncated, refusals):
        if proposal_cache is not None:
            proposal_cache._put(cache_key, tuple(value.passes for value in variants), truncated)
        return ExplorationSpace(tuple(variants), MappingProxyType(registry), MappingProxyType(bindings),
                                steps, truncated, tuple(refusals))

    variants, refusals = [], []

    def refused(path, reason):
        if len(refusals) < 16:
            refusals.append(dict(passes=path, reason=reason[:512], owner='pass_order_or_prerequisite'))

    def orders(selected, prefix=()):
        nonlocal steps, exhausted
        if steps >= max_search_steps:
            exhausted = True
            return
        steps += 1
        if not selected:
            if prefix and output in registry[prefix[-1]].spec.produces:
                yield prefix
            else:
                refused(prefix, 'selected path does not produce requested output: ' + output)
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
            except ValueError as exc:
                refused(path, str(exc))
                continue
            yield from orders(selected[:position] + selected[position + 1:], path)

    for selection in product(*domains):
        chosen = tuple(v for v in selection if v is not None)
        for path in orders(chosen):
            predicates = tuple(p for key, _ in chosen for p in guards[key])
            variant = PipelineVariant('candidate_' + _identity(path)[:16], path, predicates)
            variants.append(variant)
            if len(variants) >= max_candidates:
                return finish(variants, True, refusals)
        if exhausted:
            break
    if not variants:
        raise ExplorationRefused('no legal pipeline produces the requested artifact within the search budget',
                                 layer='construction', evidence=tuple(refusals))
    return finish(variants, exhausted, refusals)


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
    emission_issue: Mapping | None = None

    def summary(self):
        return dict(self.evaluation.summary(), bindings=self.bindings,
                    program_sha256=hashlib.sha256(self.emission.data).hexdigest() if self.emission else None,
                    program_format=self.emission.format if self.emission else None,
                    transformed_schedule_sha256=self.emission.schedule_sha256 if self.emission else None,
                    duplicate_of=self.duplicate_of,
                    target_feedback=dict(self.feedback) if self.feedback else None,
                    outcome=self.outcome())

    def outcome(self):
        """Separate observed IR effect, emission identity and missing evidence."""
        result = self.evaluation.result
        transformation = dict(source_ir='unavailable', branch_ir='unavailable')
        if result is not None:
            receipt = result.receipt
            after = receipt['after']['source_sha256']
            transformation = dict(
                source_ir='unchanged' if receipt['source']['source_sha256'] == after else 'changed',
                branch_ir='unchanged' if receipt['before']['source_sha256'] == after else 'changed',
                scope='IR fingerprints only; analyses or emitted artifacts may still differ',
            )
        status = self.evaluation.status
        if status == 'unsupported':
            code, layer, action = 'legality_refused', 'applicability', 'inspect_false_predicates_and_legal_options'
        elif status == 'unknown':
            code, layer, action = 'legality_unknown', 'applicability', 'supply_missing_current_ir_or_resource_facts'
        elif status == 'failed':
            code, layer, action = 'pass_failed', 'unknown', 'inspect_recorded_pass_failure_and_owner'
        elif self.emission_issue is not None:
            code, layer, action = 'emission_refused', self.emission_issue['layer'], 'inspect_explicit_emission_refusal'
        elif self.emission is None:
            code, layer, action = 'emission_missing', 'emission', 'produce_actual_artifact_from_completed_pipeline'
        elif self.duplicate_of is not None:
            code, layer = 'duplicate_artifact', 'emission'
            action = 'reuse_same_artifact_evaluation_try_distinct_emission'
        else:
            code, layer, action = 'emitted', 'evaluation', 'evaluate_exact_artifact_numerics_and_complete_roi'
        failed = tuple(row['name'] for row in self.evaluation.host_pass_profile if row.get('status') == 'failed')
        return dict(status=code, layer=layer, next_action=action, transformation=transformation,
                    reason=self.evaluation.error,
                    failed_passes=failed, emission_issue=dict(self.emission_issue) if self.emission_issue else None,
                    applicability=self.evaluation.applicability, bindings=self.bindings,
                    duplicate_of=self.duplicate_of)


def explore_candidates(module, space: ExplorationSpace, facts: Callable, *, output: str,
                       emission: Callable, feedback: Callable | None = None, **session_options):
    """Yield real compilation results immediately; submit unique programs once.

    The emission callback must serialize the artifact actually produced by the
    selected passes and retain its real schedule identity. Feedback is optional
    caller-observed data, joined to the exact program hash; there is no default
    score, simulator, ranking, approval or qualification. None explicitly reports
    a missing artifact; ExplorationRefused reports a known adapter refusal. Other
    adapter exceptions and malformed return values still propagate. Duplicate emissions
    retain their distinct pass profiles but do not repeat the feedback callback.
    """
    if type(space) is not ExplorationSpace or not callable(emission) or feedback is not None and not callable(feedback):
        raise ValueError('explicit generated space and actual-emission adapter required')
    seen = {}
    for result in evaluate_variants(module, space.variants, space.registry, facts,
                                    output=output, **session_options):
        emitted, duplicate, observed, issue = None, None, None, None
        variant = next(v for v in space.variants if v.name == result.variant)
        if result.status == 'compiled':
            try:
                emitted = emission(result.result)
            except ExplorationRefused as exc:
                issue = exc.summary()
            if emitted is not None and type(emitted) is not EmittedProgram:
                raise ValueError('emission adapter did not return actual program bytes')
        if emitted is not None:
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
                                emitted, duplicate, observed, issue)
