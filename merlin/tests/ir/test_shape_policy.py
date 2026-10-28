"""Synthetic cycle fixtures check policy custody; actual passes prove emission."""
from dataclasses import replace

import pytest
from test_convolution_exploration import issue_law, record, retention_law
from test_convolution_exploration import source as convolution_source
from test_pipeline_policy import registry, source

from merlin.compile.scheduling.convolution_exploration import (
    ConvolutionExplorationSource,
    convolution_pass_space,
    plan_convolution_schedule,
)
from merlin.transforms.evaluated_exploration import (
    CompletedObservation,
    ExplorationIdentity,
    exploration_context_identity,
)
from merlin.transforms.pass_exploration import EmittedProgram, ExplorationGeometry, ObservedMetrics, ValidationPartition
from merlin.transforms.pipeline_policy import PipelineVariant, PolicyFacts, ResourcePredicate, ShapePredicate
from merlin.transforms.pipeline_state import PipelineContext, PipelineSession
from merlin.transforms.shape_policy import (
    PerformanceDomain,
    ShapeCaseFeedback,
    ShapeScheduleClass,
    assess_shape_policy,
)

CONTEXT = PipelineContext(hardware='facts', resources='resources', library='selected')
PATHS = (PipelineVariant('original', ('emit',)), PipelineVariant('fold', ('normalize', 'emit'),
         (ResourcePredicate('private_bytes', 64),)))


def case(m, cycles=80, *, resources=128, unknown=False):
    session = PipelineSession(source(), context=CONTEXT)
    facts = PolicyFacts(session.stamp(), shapes=((m,),), resources=(('private_bytes', resources),))
    geometry = ExplorationGeometry('integer', facts.shapes)
    identity = ExplorationIdentity(*('1'*64 for _ in range(9)), exploration_context_identity(CONTEXT),
                                   'xdsl', 'modeled_roi')
    identity = replace(identity, source_sha256=session.stamp().source_sha256)
    control = CompletedObservation(identity, '2'*64, '3'*64, 'pass', 'complete', True, 100)
    changed = replace(control, program_sha256='4'*64, cycles=None if unknown else cycles,
                      timing_status='unknown' if unknown else 'complete')
    return ShapeCaseFeedback(geometry, facts, (('original', control), ('fold', changed)), '5'*64)


def policy(cases=None):
    cases = cases or (case(3), case(4), case(5))
    design = ValidationPartition((cases[0].geometry, cases[1].geometry), (cases[2].geometry,))
    group = ShapeScheduleClass('rectangle', (ShapePredicate(0, 0, at_least=4),), 'original', design)
    return assess_shape_policy((group,), PATHS, cases, fallback='original')


def execute(assessed, m=8, resources=128, *, domain=True):
    session = PipelineSession(source(), context=CONTEXT)
    facts = PolicyFacts(session.stamp(), shapes=((m,),), resources=(('private_bytes', resources),))
    selected_domain = PerformanceDomain.from_observation(case(4).observations[0][1].identity,
                                                        implementation_sha256='5'*64)
    return assessed.execute(session, registry(), facts, output='emitted_ir', domain=selected_domain if domain else None)


def test_cutoff_and_heldout_support_select_actual_current_ir_path():
    assessed = policy()
    assert assessed.selected[0][1] == 'fold'
    result = execute(assessed)
    assert result.receipt['selected_variant'] == 'fold'
    assert 'arith.addi' not in result.artifact
    assert 'arith.addi' in execute(assessed, m=3).artifact  # outside guard
    assert 'arith.addi' in execute(assessed, domain=False).artifact  # no domain transfer
    assert 'arith.addi' in execute(assessed, resources=32).artifact  # legality, despite performance match
    assert 'arith.addi' in execute(assessed, resources=None).artifact


@pytest.mark.parametrize('cases', [
    (case(3), case(4), case(5, cycles=101)),  # independent regression
    (case(3), case(4), case(5, unknown=True)),
    (case(2), case(4), case(5)),  # missing adjacent outside cutoff
])
def test_regression_unknown_or_missing_boundary_retains_class_incumbent(cases):
    assessed = policy(cases)
    assert assessed.selected[0][1:] == ('original', None)
    assert execute(assessed).receipt['selected_variant'] == 'original'
    # An exact-source winner remains useful even without a general guard.
    assert assessed.report['exact_source_winners'][1]['winner'] == 'fold'


def test_per_class_incumbent_and_frozen_panel_custody():
    low = (case(1), case(2), case(3))
    high = (case(4), case(5), case(6, cycles=110))
    groups = (
        ShapeScheduleClass('short', (ShapePredicate(0, 0, at_most=2),), 'original',
                           ValidationPartition((low[0].geometry, low[2].geometry), (low[1].geometry,))),
        ShapeScheduleClass('long', (ShapePredicate(0, 0, at_least=4),), 'original',
                           ValidationPartition((high[0].geometry, low[2].geometry),
                                               (high[1].geometry, high[2].geometry))),
    )
    assessed = assess_shape_policy(groups, PATHS, (*low, *high), fallback='original')
    assert [row[1] for row in assessed.selected] == ['fold', 'original']
    assert execute(assessed, m=2).receipt['selected_variant'] == 'fold'
    assert execute(assessed, m=5).receipt['selected_variant'] == 'original'
    mutated = replace(high[0].observations[1][1], identity=replace(high[0].observations[1][1].identity,
                      inputs_sha256='9'*64))
    with pytest.raises(ValueError, match='paired shape feedback'):
        replace(high[0], observations=(high[0].observations[0], ('fold', mutated)))
    with pytest.raises(ValueError, match='performance guards'):
        ShapeScheduleClass('alignment_is_legality', (ShapePredicate(0, 0, multiple_of=4),), 'original')
    with pytest.raises(ValueError, match='performance domain'):
        PerformanceDomain('bad', *('1'*64 for _ in range(3)), 'modeled_roi', '5'*64)


def test_registered_convolution_policy_emits_current_geometry_not_training_bytes():
    # Real registered counted-convolution actions and original effect emitter,
    # including actual loads/compute/drains. Cycles here are synthetic routing
    # fixtures, not timing claims or simulator output.
    original = convolution_source()
    shapes = []
    for ci in (3, 4, 5):
        plan = replace(original.plan, window=replace(original.plan.window, input_shape=(1, 3, 5, ci)))
        shapes.append(ConvolutionExplorationSource(plan, original.column_intervals, 'original-signed-window'))

    def space(current):
        return convolution_pass_space(current, verify_source=lambda s: s.__post_init__(),
            realize=lambda schedule: EmittedProgram(record(schedule), 'neutral-effects', schedule.identity),
            retention_law=retention_law(), issue_law=issue_law(), live_tile_limits=(1, 2),
            issue_modes=(False,), max_candidates=2)

    def session(current):
        return PipelineSession(current, context=CONTEXT, ir_level='counted_convolution_source',
                               fingerprint=lambda s: s.identity, verify_ir=lambda s: s.__post_init__())

    first = space(shapes[0])
    paths = tuple(first.variants)
    assert len(paths) == 2 and not paths[0].predicates
    rows = []
    for current in shapes:
        s = session(current)
        facts = PolicyFacts(s.stamp(), shapes=(current.plan.window.input_shape,))
        identity = replace(case(4).observations[0][1].identity, source_sha256=current.identity)
        observations = tuple((path.name, CompletedObservation(identity, str(i+2)*64, '3'*64,
            'pass', 'complete', True, 100 if i == 0 else 80)) for i, path in enumerate(paths))
        rows.append(ShapeCaseFeedback(ExplorationGeometry('spatial', facts.shapes), facts, observations, '5'*64))
    group = ShapeScheduleClass('channels', (ShapePredicate(0, 3, at_least=4),), paths[0].name,
                               ValidationPartition((rows[0].geometry, rows[1].geometry), (rows[2].geometry,)))
    assessed = assess_shape_policy((group,), paths, tuple(rows), fallback=paths[0].name)
    current = shapes[2]
    s = session(current)
    emitted = assessed.execute(s, space(current).registry, PolicyFacts(s.stamp(),
        shapes=(current.plan.window.input_shape,)), output='convolution_program',
        domain=PerformanceDomain.from_observation(rows[-1].observations[0][1].identity,
                                                  implementation_sha256='5'*64))
    assert emitted.receipt['selected_variant'] == paths[1].name
    assert b'load_input' in emitted.artifact.data and b'compute' in emitted.artifact.data
    assert emitted.artifact.data != record(plan_convolution_schedule(shapes[0]))


def test_source_and_implementation_mutations_cannot_reuse_prior_class_feedback():
    original = case(4)
    other_source = replace(original.observations[0][1], identity=replace(
        original.observations[0][1].identity, source_sha256='8'*64))
    with pytest.raises(ValueError, match='source/context stamp'):
        replace(original, observations=(('original', other_source),))
    assessed = policy()
    session = PipelineSession(source(), context=CONTEXT)
    facts = PolicyFacts(session.stamp(), shapes=((8,),), resources=(('private_bytes', 128),))
    old_domain = PerformanceDomain.from_observation(original.observations[0][1].identity,
                                                   implementation_sha256='5'*64)
    result = assessed.execute(session, registry(), facts, output='emitted_ir',
                              domain=replace(old_domain, implementation_sha256='6'*64))
    assert result.receipt['selected_variant'] == 'original'
    cases = (case(3), case(4), replace(case(5), implementation_sha256='6'*64))
    assert policy(cases).selected[0][1] == 'original'
    # A numerical-pass label for an unadmitted target path is not performance support.
    assert policy((case(3), case(4), case(5, resources=None))).selected[0][1] == 'original'


def diagnostic_case(m, *, cycles=80, unknown=False):
    current = case(m, cycles=cycles, unknown=unknown)
    metrics = tuple((name, ObservedMetrics(observed.identity.basis, observed.identity.digest(), (
        ('cycles', observed.cycles),
        ('device_requested_read_bytes', 1024 if name == 'original' else 256),
        ('modeled_compute_ceiling_attainment', 0.25 if name == 'original' else 0.75),
        ('host_instruction_count', 100 if name == 'original' else 25),
        ('host_preparation_cycles', None),
        ('physical_engine_occupancy', None),
    ))) for name, observed in current.observations)
    return replace(current, diagnostics=metrics)


def test_compact_same_window_diagnostics_preserve_unknown_and_actual_pass_execution():
    # Synthetic timing/diagnostic fixtures, never physical measurements.
    assessed = policy(tuple(diagnostic_case(m) for m in (3, 4, 5)))
    assert assessed.selected[0][1] == 'fold'
    assert execute(assessed).receipt['selected_variant'] == 'fold'
    assert assessed.report['decision_metric'] == 'complete_same_roi_cycles'
    observation = assessed.report['exact_source_winners'][1]['observations'][1]
    metrics = observation['diagnostics']
    assert metrics['values']['device_requested_read_bytes'] == 256
    assert metrics['values']['modeled_compute_ceiling_attainment'] == 0.75
    assert metrics['values']['host_instruction_count'] == 25
    assert metrics['values']['host_preparation_cycles'] is None
    assert set(metrics['unknown']) == {'host_preparation_cycles', 'physical_engine_occupancy'}
    assert metrics['use'].startswith('explanation_only')


@pytest.mark.parametrize('unknown', (False, True))
def test_better_diagnostic_costs_cannot_override_slow_or_unknown_cycles(unknown):
    rows = tuple(diagnostic_case(m, cycles=110, unknown=unknown) for m in (3, 4, 5))
    assessed = policy(rows)
    assert assessed.selected[0][1:] == ('original', None)
    assert execute(assessed).receipt['selected_variant'] == 'original'
    for item in assessed.report['exact_source_winners']:
        assert item['winner'] == 'original'
        changed = item['observations'][1]
        assert changed['diagnostics']['values']['device_requested_read_bytes'] == 256
        assert changed['cycles'] == (None if unknown else 110)
        assert changed['eligible'] is (not unknown)


@pytest.mark.parametrize('change', (
    {'domain': '9' * 64},  # other source/input/scope/provider
    {'basis': 'measured_roi'},  # modeled booking data is not a hardware measurement
    {'values': (('cycles', 1),)},  # cannot replace the original cycle observation
))
def test_shape_diagnostic_identity_cannot_change_the_performance_domain(change):
    current = diagnostic_case(4)
    name, metrics = current.diagnostics[1]
    with pytest.raises(ValueError, match='diagnostic'):
        replace(current, diagnostics=((name, replace(metrics, **change)),))


def test_shape_diagnostic_rows_require_existing_distinct_observed_alternatives():
    current = diagnostic_case(4)
    with pytest.raises(ValueError, match='distinct observed alternatives'):
        replace(current, diagnostics=(current.diagnostics[0], current.diagnostics[0]))
    with pytest.raises(ValueError, match='distinct observed alternatives'):
        replace(current, diagnostics=(('unobserved', current.diagnostics[0][1]),))
