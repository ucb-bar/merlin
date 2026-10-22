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
from typing import Literal

from .pipeline_state import AnalysisStamp, PassSpec, PipelineSession, validate_pass_order


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
            if values is not None and (type(values) is not frozenset or any(type(v) is not str or not v for v in values)):
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
    predicates: tuple[ShapePredicate | PropertyPredicate | ResourcePredicate | GraphPredicate, ...] = ()

    def __post_init__(self):
        if (type(self.name) is not str or not self.name or type(self.passes) is not tuple
                or not 1 <= len(self.passes) <= 64 or any(type(v) is not str or not v for v in self.passes)
                or type(self.predicates) is not tuple or len(self.predicates) > 32
                or any(type(p) not in (ShapePredicate, PropertyPredicate, ResourcePredicate, GraphPredicate) for p in self.predicates)):
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
            dict(predicate=asdict(p), match=m) for p, m in zip(variant.predicates, matches)]))
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
        facts=dict(shapes=facts.shapes, element_types=facts.element_types, layouts=facts.layouts,
                   operations=sorted(facts.operations) if facts.operations is not None else None,
                   numerical_semantics=sorted(facts.numerical_semantics) if facts.numerical_semantics is not None else None,
                   capabilities=facts.capabilities, resources=facts.resources, graph=facts.graph),
        qualification='pass execution and current-IR custody only; no semantic or hardware proof',
    ))
