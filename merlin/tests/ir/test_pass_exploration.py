import hashlib

import pytest

from merlin.transforms.certificates import ir_digest, ir_text, parse_ir
from merlin.transforms.pass_exploration import (
    EmittedProgram,
    ExplorationRefused,
    PassFamily,
    explore_candidates,
    generate_candidates,
)
from merlin.transforms.pipeline import PipelineOptions, apply_common_pipeline
from merlin.transforms.pipeline_policy import PolicyFacts, RegisteredPass, ResourcePredicate, ShapePredicate
from merlin.transforms.pipeline_state import PassSpec, PipelineContext


def source():
    return parse_ir('''builtin.module {func.func @f() -> i32 {
      %a = arith.constant 3 : i32
      %b = arith.constant 4 : i32
      %v = arith.addi %a, %b : i32
      func.return %v : i32
    }}''')


def analyze(name):
    return PassFamily(name, lambda knobs: RegisteredPass(
        PassSpec(name, 0, produces=(name,), mutates_ir=False),
        lambda session: {name: len(tuple(session.module.walk()))}))


def families(predicates=()):
    def bind(knobs):
        def transform(session):
            result = apply_common_pipeline(session.module,
                PipelineOptions(64, normalize_shapes=False, simplify_integers=knobs['simplify'], reuse_buffers=False),
                plan_only=True)
            session.module = result.module
            return {'program': ir_text(session.module)}
        return RegisteredPass(PassSpec('normalize', 1, requires=('shape', 'resources'), produces=('program',)),
                              transform)
    return (analyze('shape'), analyze('resources'),
            PassFamily('normalize', bind, (('simplify', (False, True)),), predicates))


def facts(session):
    return PolicyFacts(session.stamp(), shapes=((32, 64),), resources=(('scratch_bytes', 4096),))


def emission(result):
    assert result.artifact == ir_text(result.module)
    return EmittedProgram(result.artifact.encode(), 'xdsl', ir_digest(result.module))


def test_actual_transform_order_generation_and_identical_program_dedup():
    original = source()
    space = generate_candidates(families(), output='program')
    assert len(space.variants) == 4  # two valid analysis orders per real option
    calls = []
    def feedback(program, receipt):
        digest = hashlib.sha256(program.data).hexdigest()
        calls.append(digest)
        return {'program_sha256': digest, 'target_cycles': None, 'status': 'not_simulated'}
    results = list(explore_candidates(original, space, facts, output='program', emission=emission, feedback=feedback))
    assert all(r.evaluation.status == 'compiled' for r in results)
    assert len(calls) == 2
    assert sum(r.duplicate_of is not None for r in results) == 2
    assert any(b'arith.addi' in r.emission.data for r in results)
    assert any(b'arith.addi' not in r.emission.data for r in results)
    assert 'arith.addi' in ir_text(original)
    assert all(r.summary()['host_pass_profile'] for r in results)


def test_semantic_shape_and_resource_guards_skip_unknown_or_unsupported():
    for predicates, context, expected in [
        ((ShapePredicate(0, 0, at_least=64),), None, 'unsupported'),
        ((ResourcePredicate('scratch_bytes', 128),), None, 'unknown'),
        ((ResourcePredicate('scratch_bytes', 128),), PipelineContext(resources='selected-resource-identity'),
         'compiled'),
    ]:
        space = generate_candidates(families(predicates), output='program')
        results = list(explore_candidates(source(), space, facts, output='program', emission=emission, context=context))
        assert all(r.evaluation.status == expected for r in results)
        assert all((r.emission is not None) == (expected == 'compiled') for r in results)


def test_missing_prerequisites_cycle_and_finite_search_refuse_before_actions():
    with pytest.raises(ExplorationRefused, match='no legal pipeline') as caught:
        generate_candidates(families()[1:], output='program')
    refusal = caught.value.summary()
    assert refusal['layer'] == 'construction'
    assert any('prerequisites' in row['reason'] for row in refusal['evidence'])
    with pytest.raises(ValueError, match='no legal pipeline'):
        generate_candidates(families(), output='program', before=(('shape', 'resources'), ('resources', 'shape')))
    space = generate_candidates(families(), output='program', max_candidates=1)
    assert len(space.variants) == 1 and space.truncated


def test_explicit_emission_refusal_and_missing_artifact_do_not_hide_infrastructure():
    space = generate_candidates(families(), output='program', max_candidates=1)
    args = (source(), space, facts)
    missing = list(explore_candidates(*args, output='program', emission=lambda _: None))[0]
    assert missing.summary()['outcome']['status'] == 'emission_missing'
    assert missing.outcome()['transformation']['branch_ir'] == 'unchanged'
    def refused(_):
        raise ExplorationRefused('selected layout has no adapter realization', layer='adapter')
    rejected = list(explore_candidates(*args, output='program', emission=refused))[0]
    assert rejected.outcome()['emission_issue']['reason'] == 'selected layout has no adapter realization'
    assert rejected.outcome()['layer'] == 'adapter'
    def broken(_):
        raise OSError('real tool transport failed')
    with pytest.raises(OSError, match='transport failed'):
        list(explore_candidates(*args, output='program', emission=broken))


def test_failed_registered_pass_keeps_raw_reason_without_invented_owner():
    def broken(session):
        raise ValueError('declared pass could not realize this source')
    broken_family = PassFamily('broken', lambda _: RegisteredPass(
        PassSpec('broken', 0, produces=('program',)), broken))
    results = list(explore_candidates(source(), generate_candidates((broken_family,), output='program'),
                                     facts, output='program', emission=emission))
    outcome = results[0].summary()['outcome']
    assert outcome['status'] == 'pass_failed' and outcome['layer'] == 'unknown'
    assert 'could not realize' in outcome['reason']
    assert len(outcome['failed_passes']) == 1


def test_feedback_cannot_attach_to_another_program():
    space = generate_candidates(families(), output='program', max_candidates=1)
    with pytest.raises(ValueError, match='actual emitted program'):
        list(explore_candidates(source(), space, facts, output='program', emission=emission,
                                feedback=lambda program, receipt: {'program_sha256': '0' * 64}))


def test_knob_specific_shape_guards_generate_different_applicable_paths():
    def guards(knobs):
        return (ShapePredicate(0, 0, at_least=64 if knobs['simplify'] else 1),)
    space = generate_candidates(families(guards), output='program')
    results = list(explore_candidates(source(), space, facts, output='program', emission=emission))
    assert [r.evaluation.status for r in results].count('compiled') == 2
    assert [r.evaluation.status for r in results].count('unsupported') == 2
