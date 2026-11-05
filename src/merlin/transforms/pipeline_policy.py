"""Bounded semantic branch selection over the existing pass session.

Policies are compiler choices, not proofs or hardware qualification. Facts must
come from current IR and selected contracts. They contain no workload identities,
benchmark scores or capsule names. Registered actions retain their original host
verification obligations; target lowering stays in the target adapter.

A compact counted-loop compiler can register its existing compact-IR actions.
This selector does not expand loops or route them through finite schedule IR.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Literal

from .pipeline_state import AnalysisStamp, PassSpec, PipelineSession, validate_pass_order


@dataclass(frozen=True)
class DimensionRef:
    operand: int
    axis: int

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in (self.operand, self.axis)):
            raise ValueError('dimension reference requires nonnegative operand and axis')

    def value(self, facts):
        if self.operand >= len(facts.shapes) or self.axis >= len(facts.shapes[self.operand]):
            return None
        return facts.shapes[self.operand][self.axis]


@dataclass(frozen=True)
class ResourceRef:
    resource: str

    def __post_init__(self):
        if type(self.resource) is not str or not self.resource:
            raise ValueError('resource reference requires a selected resource name')

    def value(self, facts):
        if not facts.stamp.context.resources:
            return None
        return dict(facts.resources).get(self.resource)


@dataclass(frozen=True)
class RatioFeature:
    """Dimension divided by another dimension or a selected resource extent.

    The caller must use commensurate dimension/resource units. Integer resource
    facts do not automatically convert elements into bytes or another unit.
    """
    numerator: DimensionRef
    denominator: DimensionRef | ResourceRef

    def __post_init__(self):
        if (type(self.numerator) is not DimensionRef
                or type(self.denominator) not in (DimensionRef, ResourceRef)):
            raise ValueError('ratio feature requires dimension and dimension/resource references')

    def value(self, facts):
        numerator = self.numerator.value(facts)
        denominator = self.denominator.value(facts)
        if numerator is None or denominator is None or denominator == 0:
            return None
        return Fraction(numerator, denominator)


@dataclass(frozen=True)
class TileFeature:
    """Ceiling tile count, remainder elements or remainder/tile extent.

    Tile extent is a selected positive resource in the same units as the
    dimension. Computing these features does not allocate any device storage.
    """
    dimension: DimensionRef
    tile: ResourceRef
    kind: Literal['count', 'tail', 'tail_fraction'] = 'count'

    def __post_init__(self):
        if (type(self.dimension) is not DimensionRef or type(self.tile) is not ResourceRef
                or self.kind not in ('count', 'tail', 'tail_fraction')):
            raise ValueError('tile feature requires dimension, selected tile resource and known kind')

    def value(self, facts):
        dimension = self.dimension.value(facts)
        tile = self.tile.value(facts)
        if dimension is None or tile is None or tile == 0:
            return None
        if self.kind == 'count':
            return Fraction((dimension + tile - 1) // tile)
        tail = dimension % tile
        return Fraction(tail, tile if self.kind == 'tail_fraction' else 1)


def _bound(value):
    if (type(value) is not tuple or len(value) != 2
            or any(type(v) is not int for v in value) or value[0] < 0 or value[1] <= 0):
        raise ValueError('feature bounds require nonnegative numerator and positive denominator')
    fraction = Fraction(*value)
    return fraction.numerator, fraction.denominator


@dataclass(frozen=True)
class FeaturePredicate:
    """Inclusive exact rational bounds, with no floating point cutoff rounding.

    This is performance applicability only. A matching feature never grants a
    pass's independent legality or changes its verification requirements.
    """
    feature: RatioFeature | TileFeature
    at_least: tuple[int, int] = (0, 1)
    at_most: tuple[int, int] | None = None

    def __post_init__(self):
        if type(self.feature) not in (RatioFeature, TileFeature):
            raise ValueError('feature predicate requires an exact shape/resource feature')
        lower = _bound(self.at_least)
        upper = None if self.at_most is None else _bound(self.at_most)
        if upper is not None and Fraction(*upper) < Fraction(*lower):
            raise ValueError('feature upper bound must not precede lower bound')
        object.__setattr__(self, 'at_least', lower)
        object.__setattr__(self, 'at_most', upper)

    def value(self, facts):
        return self.feature.value(facts)

    def value_record(self, facts):
        value = self.value(facts)
        return None if value is None else dict(numerator=value.numerator, denominator=value.denominator)

    def match(self, facts):
        value = self.value(facts)
        return None if value is None else (value >= Fraction(*self.at_least)
            and (self.at_most is None or value <= Fraction(*self.at_most)))


@dataclass(frozen=True)
class PolicyFacts:
    stamp: AnalysisStamp
    shapes: tuple[tuple[int | None, ...], ...] = ()
    element_types: tuple[str | None, ...] = ()
    layouts: tuple[str | None, ...] = ()
    operations: frozenset[str] | None = None
    numerical_semantics: frozenset[str] | None = None
    capabilities: tuple[tuple[str, bool | None], ...] = ()
    resources: tuple[tuple[str, int | None], ...] = ()
    graph: tuple[tuple[str, int | None], ...] = ()

    def __post_init__(self):
        if type(self.stamp) is not AnalysisStamp:
            raise ValueError('policy facts require a current IR/context stamp')
        if (type(self.shapes) is not tuple or any(type(s) is not tuple or
                any(d is not None and (type(d) is not int or d < 0) for d in s) for s in self.shapes)):
            raise ValueError('shapes require immutable nonnegative dimensions or unknown')
        for values in (self.element_types, self.layouts):
            if type(values) is not tuple or any(v is not None and (type(v) is not str or not v) for v in values):
                raise ValueError('operand properties require strings or unknown')
        for values in (self.operations, self.numerical_semantics):
            if values is not None and (type(values) is not frozenset
                                       or any(type(v) is not str or not v for v in values)):
                raise ValueError('semantic facts require immutable names or unknown')
        for values, scalar in ((self.capabilities, bool), (self.resources, int), (self.graph, int)):
            if (type(values) is not tuple or any(type(v) is not tuple or len(v) != 2 for v in values)
                    or len({v[0] for v in values}) != len(values)
                    or any(type(k) is not str or not k or v is not None and
                           (type(v) is not scalar or scalar is int and v < 0) for k, v in values)):
                raise ValueError('contract facts require unique typed names and values')
        if any(name not in GRAPH_METRICS for name, _ in self.graph):
            raise ValueError('unsupported graph property; workload identities are not policy facts')


GRAPH_METRICS = frozenset({'max_fanout', 'live_in_count', 'live_out_count',
                           'peak_live_values', 'region_operation_count'})


@dataclass(frozen=True)
class GraphPredicate:
    metric: Literal['max_fanout', 'live_in_count', 'live_out_count', 'peak_live_values', 'region_operation_count']
    at_least: int = 0
    at_most: int | None = None

    def __post_init__(self):
        if (self.metric not in GRAPH_METRICS or type(self.at_least) is not int or self.at_least < 0
                or self.at_most is not None and (type(self.at_most) is not int or self.at_most < self.at_least)):
            raise ValueError('invalid graph predicate')

    def match(self, facts):
        value = dict(facts.graph).get(self.metric)
        return None if value is None else value >= self.at_least and (self.at_most is None or value <= self.at_most)


@dataclass(frozen=True)
class ShapePredicate:
    operand: int
    axis: int
    at_least: int = 0
    at_most: int | None = None
    multiple_of: int = 1

    def __post_init__(self):
        if (any(type(v) is not int or v < 0 for v in (self.operand, self.axis, self.at_least))
                or type(self.multiple_of) is not int or self.multiple_of < 1
                or self.at_most is not None and (type(self.at_most) is not int or self.at_most < self.at_least)):
            raise ValueError('invalid shape predicate')

    def match(self, facts):
        if self.operand >= len(facts.shapes) or self.axis >= len(facts.shapes[self.operand]):
            return None
        size = facts.shapes[self.operand][self.axis]
        return None if size is None else (size >= self.at_least and
            (self.at_most is None or size <= self.at_most) and size % self.multiple_of == 0)


@dataclass(frozen=True)
class PropertyPredicate:
    facet: Literal['element_type', 'layout', 'operation', 'numerical_semantics', 'capability']
    value: str
    operand: int = 0

    def __post_init__(self):
        if (self.facet not in {'element_type', 'layout', 'operation', 'numerical_semantics', 'capability'}
                or type(self.value) is not str or not self.value
                or type(self.operand) is not int or self.operand < 0):
            raise ValueError('unsupported semantic predicate')

    def match(self, facts):
        if self.facet in {'element_type', 'layout'}:
            values = facts.element_types if self.facet == 'element_type' else facts.layouts
            value = values[self.operand] if self.operand < len(values) else None
            return None if value is None else value == self.value
        if self.facet == 'capability':
            return dict(facts.capabilities).get(self.value) if facts.stamp.context.hardware else None
        if self.facet == 'numerical_semantics' and not facts.stamp.context.numerical_policy:
            return None
        values = facts.operations if self.facet == 'operation' else facts.numerical_semantics
        return None if values is None else self.value in values


@dataclass(frozen=True)
class ResourcePredicate:
    resource: str
    at_least: int

    def __post_init__(self):
        if type(self.resource) is not str or not self.resource or type(self.at_least) is not int or self.at_least < 0:
            raise ValueError('resource predicate requires a named nonnegative capacity')

    def match(self, facts):
        value = dict(facts.resources).get(self.resource) if facts.stamp.context.resources else None
        return None if value is None else value >= self.at_least


POLICY_PREDICATE_TYPES = (ShapePredicate, PropertyPredicate, ResourcePredicate, GraphPredicate, FeaturePredicate)


@dataclass(frozen=True)
class RegisteredPass:
    spec: PassSpec
    action: Callable[[PipelineSession], Mapping[str, object]]

    def __post_init__(self):
        if type(self.spec) is not PassSpec or not callable(self.action):
            raise ValueError('registry entries require an existing pass spec and callable action')


@dataclass(frozen=True)
class PipelineVariant:
    name: str
    passes: tuple[str, ...]
    predicates: tuple[
        ShapePredicate | PropertyPredicate | ResourcePredicate | GraphPredicate | FeaturePredicate, ...
    ] = ()

    def __post_init__(self):
        if (type(self.name) is not str or not self.name or type(self.passes) is not tuple
                or not 1 <= len(self.passes) <= 64 or any(type(v) is not str or not v for v in self.passes)
                or type(self.predicates) is not tuple or len(self.predicates) > 32
                or any(type(p) not in POLICY_PREDICATE_TYPES for p in self.predicates)):
            raise ValueError('variant requires a bounded ordered pass path and typed predicates')


@dataclass(frozen=True)
class PipelinePolicy:
    variants: tuple[PipelineVariant, ...]
    fallback: str

    def __post_init__(self):
        if (type(self.variants) is not tuple or not 1 <= len(self.variants) <= 16
                or any(type(v) is not PipelineVariant for v in self.variants)
                or len({v.name for v in self.variants}) != len(self.variants)):
            raise ValueError('policy requires at most sixteen uniquely named variants')
        fallback = next((v for v in self.variants if v.name == self.fallback), None)
        if fallback is None or fallback.predicates:
            raise ValueError('policy requires an explicit unconditional fallback')
        if any(not v.predicates for v in self.variants if v is not fallback):
            raise ValueError('non-fallback variants require applicability predicates')


@dataclass(frozen=True)
class PolicyResult:
    artifact: object
    module: object
    receipt: dict


def registry_metadata(registry: Mapping[str, RegisteredPass]):
    """Expose only declared pass contracts; no invented knobs or applicability."""
    if not isinstance(registry, Mapping) or len(registry) > 256:
        raise ValueError('pass registry exceeds bounded policy scope')
    for name, entry in registry.items():
        if type(entry) is not RegisteredPass or name != entry.spec.name:
            raise ValueError('registry key differs from declared pass identity')
    return tuple(dict(asdict(entry.spec), input_ir_kind=entry.spec.ir_level,
                      current_ir_kind_after_pass=entry.spec.ir_level,
                      output_analyses=entry.spec.produces) for entry in registry.values())


def execute_policy(session: PipelineSession, policy: PipelinePolicy,
                   registry: Mapping[str, RegisteredPass], facts: PolicyFacts, *, output: str):
    """Run one selected path through the native manager and consume its output.

    Use a fresh private session for each bounded candidate variant. There is no
    retry after a partially mutating failure, and no fallback to stale artifacts.
    Register verified common transforms, schedule compilation or compact target
    lowering adapters as appropriate; this function supplies none of their proof
    or execution authority. The final pass must produce the requested output on
    the current IR, so selected transformations are on the emission path.
    """
    if type(session) is not PipelineSession or type(policy) is not PipelinePolicy or type(facts) is not PolicyFacts:
        raise ValueError('explicit pipeline session, policy and typed facts required')
    if session.records or facts.stamp != session.stamp():
        raise ValueError('policy requires fresh current-IR facts and an unused session')
    registry = dict(registry)
    metadata = registry_metadata(registry)
    if not isinstance(output, str) or not output:
        raise ValueError('policy requires an explicit emitted output analysis')
    for variant in policy.variants:
        if any(name not in registry for name in variant.passes):
            raise ValueError(f'unknown registered pass in variant {variant.name}')
        specs = tuple(registry[name].spec for name in variant.passes)
        validate_pass_order(specs)
        if any(spec.ir_level != session.ir_level for spec in specs) or output not in specs[-1].produces:
            raise ValueError('variant must consume its selected IR level and produce its final output')
    selected = None
    decisions = []
    for variant in policy.variants:
        if variant.name == policy.fallback:
            continue
        matches = tuple(predicate.match(facts) for predicate in variant.predicates)
        status = 'unsupported' if False in matches else 'unknown' if None in matches else 'applicable'
        decisions.append(dict(variant=variant.name, status=status, predicates=[
            dict(predicate=asdict(p), match=m, **(
                dict(feature_value=p.value_record(facts)) if type(p) is FeaturePredicate else {}))
            for p, m in zip(variant.predicates, matches)]))
        if status == 'applicable':
            selected = variant
            break
    fallback = selected is None
    selected = selected or next(v for v in policy.variants if v.name == policy.fallback)
    before = session.stamp()
    for name in selected.passes:
        entry = registry[name]
        session.run(entry.spec, lambda entry=entry: entry.action(session))
    artifact = session.analysis(output)
    return PolicyResult(artifact, session.module, dict(
        schema='merlin.compiler_pipeline_policy.v1', selected_variant=selected.name,
        fallback_selected=fallback, decisions=decisions, policy=asdict(policy),
        before=asdict(before), after=asdict(session.stamp()), registry=metadata,
        passes=tuple(asdict(record) for record in session.records), output=output,
        host_pass_profile=tuple(session.host_pass_profile),
        facts=dict(shapes=facts.shapes, element_types=facts.element_types, layouts=facts.layouts,
                   operations=sorted(facts.operations) if facts.operations is not None else None,
                   numerical_semantics=(sorted(facts.numerical_semantics)
                                        if facts.numerical_semantics is not None else None),
                   capabilities=facts.capabilities, resources=facts.resources, graph=facts.graph),
        qualification='pass execution and current-IR custody only; no semantic or hardware proof',
    ))


@dataclass(frozen=True)
class VariantEvaluation:
    """One completed machine-search alternative; compilation is not qualification."""

    variant: str
    status: Literal['compiled', 'unsupported', 'unknown', 'failed']
    result: PolicyResult | None
    applicability: tuple[bool | None, ...]
    error: str | None = None
    host_pass_profile: tuple[dict, ...] = ()

    def summary(self):
        """Bounded feedback; the caller retains the full IR/receipt separately."""
        return dict(
            variant=self.variant, status=self.status, applicability=self.applicability,
            error=self.error,
            source_sha256=self.result.receipt['source']['source_sha256'] if self.result else None,
            prepared_ir_sha256=self.result.receipt['before']['source_sha256'] if self.result else None,
            output_ir_sha256=self.result.receipt['after']['source_sha256'] if self.result else None,
            prepared_passes=tuple(row['name'] for row in self.result.receipt['prefix_passes']) if self.result else (),
            executed_passes=tuple(row['name'] for row in self.result.receipt['passes']) if self.result else (),
            host_pass_profile=self.host_pass_profile,
            qualification='compilation only; independent correctness and timing required',
        )


def evaluate_variants(module, variants: tuple[PipelineVariant, ...],
                      registry: Mapping[str, RegisteredPass], facts: Callable, *, output: str,
                      context=None, ir_level='structured', clone=None,
                      fingerprint=None, verify_ir=None, prefix: tuple[str, ...] = ()):
    """Yield every applicable explicit variant on a fresh private IR.

    This is deterministic compiler search, with no LLM, simulator or implicit
    performance model. Variants may select different registered pass orders or
    separately registered parameterizations. Each yielded result is immediately
    available to the caller's rolling execution queue. Unsupported/unknown guards
    never become permissions; failed alternatives discard their partial IR and
    cannot contaminate later alternatives. No automatic fallback or ranking is
    inferred from a compilation result.

    ``facts(session)`` derives the existing typed facts for the current clone.
    Ordinary xDSL modules use their native clone/verification. Alternate counted
    IR must supply clone, fingerprint and verify_ir adapters without flattening
    loops. Host transform verification and independent execution obligations are
    unchanged. The source must remain immutable while this iterator is consumed.

    An explicit ``prefix`` executes shared preparation once before branching.
    Its transformed IR is cloned, while all prefix analyses are discarded: each
    branch must derive its own current facts and any pass prerequisites. This is
    invocation-local preparation, not a persistent cache or reusable proof. Only
    preparation legal for every alternative belongs in the unconditional prefix.
    """
    if (type(variants) is not tuple or not 1 <= len(variants) <= 16
            or any(type(v) is not PipelineVariant for v in variants)
            or len({v.name for v in variants}) != len(variants)):
        raise ValueError('search requires one to sixteen uniquely named variants')
    if not callable(facts) or not isinstance(output, str) or not output:
        raise ValueError('search requires a facts factory and explicit output analysis')
    if clone is not None and not callable(clone):
        raise ValueError('search clone adapter must be callable')
    registry = dict(registry)
    metadata = registry_metadata(registry)
    if (type(prefix) is not tuple or len(prefix) > 64
            or any(type(name) is not str or name not in registry for name in prefix)):
        raise ValueError('search prefix requires at most sixty-four registered passes')
    prefix_specs = tuple(registry[name].spec for name in prefix)
    validate_pass_order(prefix_specs)
    if any(spec.ir_level != ir_level for spec in prefix_specs):
        raise ValueError('prefix must consume the selected IR level')
    # Validate the entire search space before any action, including late paths.
    for variant in variants:
        if any(name not in registry for name in variant.passes):
            raise ValueError(f'unknown registered pass in variant {variant.name}')
        specs = tuple(registry[name].spec for name in variant.passes)
        validate_pass_order(specs)
        if any(spec.ir_level != ir_level for spec in specs) or output not in specs[-1].produces:
            raise ValueError('variant must consume its selected IR level and produce its final output')
        if prefix_specs and specs[0].stage < prefix_specs[-1].stage:
            raise ValueError('variant pass order precedes shared preparation')
    source_session = PipelineSession(module, context=context, ir_level=ir_level,
                                     fingerprint=fingerprint, verify_ir=verify_ir)
    source_stamp = source_session.stamp()
    retained_clones = []  # At most one prefix plus sixteen private alternatives.
    prepared_session = source_session
    if prefix:
        private = clone(module) if clone is not None else module.clone()
        if private is module:
            raise ValueError('search prefix requires a distinct private clone')
        retained_clones.append(private)
        prepared_session = PipelineSession(private, context=source_stamp.context, ir_level=ir_level,
                                           fingerprint=fingerprint, verify_ir=verify_ir)
        if prepared_session.stamp() != source_stamp:
            raise ValueError('search prefix clone differs from the selected source or contracts')
        for name in prefix:
            entry = registry[name]
            prepared_session.run(entry.spec, lambda entry=entry: entry.action(prepared_session))
    prepared_stamp = prepared_session.stamp()
    prefix_records = tuple(asdict(record) for record in prepared_session.records)
    prefix_profile = tuple(prepared_session.host_pass_profile)
    # Retain only prepared IR and records. Drop produced analysis objects even
    # from the preparation session; they may contain references into its IR.
    prepared_session = PipelineSession(prepared_session.module, context=prepared_stamp.context,
                                       ir_level=ir_level, fingerprint=fingerprint, verify_ir=verify_ir)
    for variant in variants:
        if source_session.stamp() != source_stamp:
            raise ValueError('search source changed between alternatives')
        if prepared_session.stamp() != prepared_stamp:
            raise ValueError('search prepared IR or contracts changed between alternatives')
        private = clone(prepared_session.module) if clone is not None else prepared_session.module.clone()
        if (private is module or private is prepared_session.module
                or any(private is previous for previous in retained_clones)):
            raise ValueError('search requires a distinct private clone for every alternative')
        retained_clones.append(private)
        session = PipelineSession(private, context=source_stamp.context, ir_level=ir_level,
                                  fingerprint=fingerprint, verify_ir=verify_ir)
        if session.stamp() != prepared_stamp:
            raise ValueError('search clone differs from the selected source or contracts')
        applicability = ()
        try:
            current = facts(session)
            if (type(current) is not PolicyFacts or current.stamp != session.stamp()
                    or session.stamp() != prepared_stamp):
                raise ValueError('search facts require unchanged current-IR source and contracts')
            applicability = tuple(predicate.match(current) for predicate in variant.predicates)
            status = 'unsupported' if False in applicability else 'unknown' if None in applicability else 'compiled'
            if status != 'compiled':
                evaluation = VariantEvaluation(variant.name, status, None, applicability)
            else:
                before = session.stamp()
                for name in variant.passes:
                    entry = registry[name]
                    session.run(entry.spec, lambda entry=entry: entry.action(session))
                result = PolicyResult(session.analysis(output), session.module, dict(
                    schema='merlin.compiler_pipeline_variant.v1', variant=asdict(variant),
                    source=asdict(source_stamp), prefix_passes=prefix_records,
                    prefix_host_pass_profile=prefix_profile,
                    before=asdict(before), after=asdict(session.stamp()), registry=metadata,
                    passes=tuple(asdict(record) for record in session.records), output=output,
                    host_pass_profile=tuple(session.host_pass_profile),
                    applicability=applicability,
                    qualification='pass execution and current-IR custody only; no semantic or hardware proof',
                ))
                evaluation = VariantEvaluation(variant.name, status, result, applicability,
                                               host_pass_profile=tuple(session.host_pass_profile))
        except Exception as exc:
            evaluation = VariantEvaluation(variant.name, 'failed', None, applicability,
                                           f'{type(exc).__name__}: {str(exc)[:512]}',
                                           host_pass_profile=tuple(session.host_pass_profile))
        if source_session.stamp() != source_stamp:
            raise ValueError('search source changed during an alternative')
        if prepared_session.stamp() != prepared_stamp:
            raise ValueError('search prepared IR or contracts changed during an alternative')
        yield evaluation
