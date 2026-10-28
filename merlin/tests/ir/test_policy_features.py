from dataclasses import replace
from fractions import Fraction

import pytest

from merlin.transforms.certificates import ir_text, parse_ir
from merlin.transforms.pipeline import PipelineOptions, apply_common_pipeline
from merlin.transforms.pipeline_policy import (
    DimensionRef,
    FeaturePredicate,
    PipelinePolicy,
    PipelineVariant,
    PolicyFacts,
    PropertyPredicate,
    RatioFeature,
    RegisteredPass,
    ResourceRef,
    TileFeature,
    execute_policy,
)
from merlin.transforms.pipeline_state import PassSpec, PipelineContext, PipelineSession


def session():
    module = parse_ir('''builtin.module { func.func @f() -> i32 {
      %a = arith.constant 3 : i32
      %b = arith.constant 4 : i32
      %sum = arith.addi %a, %b : i32
      func.return %sum : i32
    }}''')
    return PipelineSession(module, context=PipelineContext(hardware='h', resources='r'))


def facts(shapes=((10, 3),), resources=(('tile_extent', 4),)):
    return PolicyFacts(session().stamp(), shapes=shapes, resources=resources)


def test_exact_ratios_and_canonical_bounds_without_float_rounding():
    aspect = FeaturePredicate(RatioFeature(DimensionRef(0, 0), DimensionRef(0, 1)),
                              at_least=(20, 6), at_most=(10, 3))
    assert aspect.at_least == aspect.at_most == (10, 3)
    assert aspect.match(facts()) is True
    assert aspect.value_record(facts()) == dict(numerator=10, denominator=3)
    relative = FeaturePredicate(RatioFeature(DimensionRef(0, 0), ResourceRef('tile_extent')),
                                at_least=(5, 2), at_most=(5, 2))
    assert relative.match(facts()) is True
    huge = 2**80
    cutoff = FeaturePredicate(RatioFeature(DimensionRef(0, 0), DimensionRef(0, 1)),
                              at_most=(1, 1))
    assert cutoff.match(facts(((huge + 1, huge),))) is False
    assert cutoff.match(facts(((huge, huge),))) is True
    assert cutoff.match(facts(((0, 3),))) is True
    assert cutoff.match(facts(((0, 0),))) is None


@pytest.mark.parametrize('kind,expected', [('count', 3), ('tail', 2), ('tail_fraction', Fraction(1, 2))])
def test_selected_tile_features_are_compact_and_zero_dimension_is_known(kind, expected):
    feature = TileFeature(DimensionRef(0, 0), ResourceRef('tile_extent'), kind)
    assert feature.value(facts()) == expected
    assert feature.value(facts(((0,),))) == 0
    assert TileFeature(DimensionRef(0, 0), ResourceRef('tile_extent')).value(
        facts(((10**30,),), (('tile_extent', 7),))) == (10**30 + 6) // 7


@pytest.mark.parametrize('resource', [(), (('tile_extent', None),), (('tile_extent', 0),)])
def test_missing_unknown_or_zero_resource_stays_unknown(resource):
    current = facts(resources=resource)
    for feature in (RatioFeature(DimensionRef(0, 0), ResourceRef('tile_extent')),
                    TileFeature(DimensionRef(0, 0), ResourceRef('tile_extent'))):
        predicate = FeaturePredicate(feature)
        assert predicate.match(current) is None
        assert predicate.value_record(current) is None


def test_unselected_resources_and_dynamic_or_absent_dimensions_stay_unknown():
    feature = RatioFeature(DimensionRef(0, 0), ResourceRef('tile_extent'))
    current = facts()
    unselected = replace(current, stamp=replace(current.stamp, context=PipelineContext()))
    assert feature.value(unselected) is None
    assert feature.value(facts(((None,),))) is None
    assert feature.value(facts(())) is None
    assert RatioFeature(DimensionRef(0, 0), DimensionRef(1, 0)).value(current) is None
    assert RatioFeature(DimensionRef(0, 0), DimensionRef(0, 9)).value(current) is None


@pytest.mark.parametrize('bad', [(0, 0), (-1, 1), (True, 1), (1, 1.0), [1, 2]])
def test_rational_bounds_require_exact_nonnegative_integer_pairs(bad):
    with pytest.raises(ValueError, match='feature bounds'):
        FeaturePredicate(RatioFeature(DimensionRef(0, 0), DimensionRef(0, 1)), at_least=bad)


def test_fixed_feature_types_refuse_arbitrary_expressions_and_inverted_bounds():
    with pytest.raises(ValueError, match='dimension reference'):
        DimensionRef(True, 0)
    with pytest.raises(ValueError, match='known kind'):
        TileFeature(DimensionRef(0, 0), ResourceRef('tile_extent'), 'benchmark_name')
    with pytest.raises(ValueError, match='feature predicate'):
        FeaturePredicate(lambda _: 1)
    with pytest.raises(ValueError, match='upper bound'):
        FeaturePredicate(RatioFeature(DimensionRef(0, 0), DimensionRef(0, 1)), (3, 2), (1, 1))


@pytest.mark.parametrize('resource,capability,selected,status', [
    (4, True, 'relative', 'applicable'),
    (8, True, 'fallback', 'unsupported'),
    (0, True, 'fallback', 'unknown'),
    (None, True, 'fallback', 'unknown'),
    (4, False, 'fallback', 'unsupported'),
    (4, None, 'fallback', 'unknown'),
])
@pytest.mark.parametrize('kind,threshold,values', [
    ('ratio', (2, 1), {4: (5, 2), 8: (5, 4)}),
    ('count', (3, 1), {4: (3, 1), 8: (2, 1)}),
    ('tail_fraction', (1, 2), {4: (1, 2), 8: (1, 4)}),
])
def test_features_select_actual_registered_passes_without_granting_legality(
    resource, capability, selected, status, kind, threshold, values,
):
    current = session()
    seen = []

    def normalize(s):
        seen.append('normalize')
        transformed = apply_common_pipeline(s.module, PipelineOptions(64, reuse_buffers=False), plan_only=True)
        s.module = transformed.module
        return {}

    registry = {
        'normalize': RegisteredPass(PassSpec('normalize', 0), normalize),
        'emit': RegisteredPass(PassSpec('emit', 1, produces=('emitted_ir',), mutates_ir=False),
                               lambda s: {'emitted_ir': ir_text(s.module)}),
    }
    feature = (RatioFeature(DimensionRef(0, 0), ResourceRef('tile_extent')) if kind == 'ratio'
               else TileFeature(DimensionRef(0, 0), ResourceRef('tile_extent'), kind))
    predicate = FeaturePredicate(feature, at_least=threshold)
    policy = PipelinePolicy((
        PipelineVariant('relative', ('normalize', 'emit'), (predicate, PropertyPredicate('capability', 'legal'))),
        PipelineVariant('fallback', ('emit',)),
    ), 'fallback')
    current_facts = PolicyFacts(current.stamp(), shapes=((10,),), resources=(('tile_extent', resource),),
                                capabilities=(('legal', capability),))
    result = execute_policy(current, policy, registry, current_facts, output='emitted_ir')
    assert result.receipt['selected_variant'] == selected
    assert result.receipt['decisions'][0]['status'] == status
    assert seen == (['normalize'] if selected == 'relative' else [])
    assert ('arith.addi' not in result.artifact) == (selected == 'relative')
    assert result.artifact == ir_text(result.module)
    expected = values.get(resource)
    assert result.receipt['decisions'][0]['predicates'][0]['feature_value'] == (
        dict(numerator=expected[0], denominator=expected[1]) if expected else None)


def test_resource_feature_selection_refuses_stale_context_before_actions():
    current = session()
    stale = PolicyFacts(current.stamp(), shapes=((10,),), resources=(('tile_extent', 4),))
    current.context = PipelineContext(hardware='h', resources='new_resource_identity')
    registry = {'emit': RegisteredPass(PassSpec('emit', 0, produces=('out',), mutates_ir=False),
                                      lambda _: pytest.fail('stale policy must not execute'))}
    policy = PipelinePolicy((PipelineVariant('fallback', ('emit',)),), 'fallback')
    with pytest.raises(ValueError, match='current-IR'):
        execute_policy(current, policy, registry, stale, output='out')
