"""Machine-search variants transform and consume real IR without sharing edits."""

from dataclasses import replace

import pytest
from xdsl.interpreter import Interpreter
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.func import FuncFunctions
from xdsl.interpreters.scf import ScfFunctions

from merlin.transforms.certificates import ir_digest, ir_text, parse_ir
from merlin.transforms.pipeline import PipelineOptions, apply_common_pipeline
from merlin.transforms.pipeline_policy import (
    PipelineVariant, PolicyFacts, PropertyPredicate, RegisteredPass, evaluate_variants,
)
from merlin.transforms.pipeline_state import PassSpec


def source(extent=13):
    return parse_ir(f'''builtin.module {{ func.func @kernel(%x: i32) -> i32 {{
      %lo = arith.constant 0 : index
      %hi = arith.constant {extent} : index
      %step = arith.constant 1 : index
      %zero = arith.constant 0 : i32
      %result = scf.for %i = %lo to %hi step %step iter_args(%acc = %x) -> (i32) {{
        %same = arith.addi %acc, %zero : i32
        %iv = arith.index_cast %i : index to i32
        %next = arith.addi %same, %iv : i32
        scf.yield %next : i32
      }}
      func.return %result : i32
    }} }}''')


def execute(text, value):
    interpreter = Interpreter(parse_ir(text))
    for functions in (ArithFunctions(), FuncFunctions(), ScfFunctions()):
        interpreter.register_implementations(functions)
    return interpreter.call_op('kernel', (value,))


def registry():
    def transform(options):
        def action(session):
            # Exercise the original shared broker in explicit unverified planning
            # mode. Interpreted checks below do not manufacture formal authority.
            result = apply_common_pipeline(session.module, options, plan_only=True)
            session.module = result.module
            return {}
        return action
    return {
        'clean': RegisteredPass(PassSpec('clean', 0),
            transform(PipelineOptions(64, reuse_buffers=False))),
        'unroll4': RegisteredPass(PassSpec('unroll4', 1),
            transform(PipelineOptions(64, reuse_buffers=False, unroll_factors=(4,)))),
        'emit': RegisteredPass(PassSpec('emit', 2, produces=('ir',), mutates_ir=False),
            lambda s: {'ir': ir_text(s.module)}),
    }


def facts(session):
    return PolicyFacts(session.stamp(), operations=frozenset(op.name for op in session.module.walk()))


def variants():
    return (
        PipelineVariant('baseline', ('emit',)),
        PipelineVariant('clean', ('clean', 'emit')),
        PipelineVariant('unrolled', ('clean', 'unroll4', 'emit'),
                        (PropertyPredicate('operation', 'scf.for'),)),
    )


@pytest.mark.parametrize('extent', [0, 3, 13, 32])
def test_real_shared_pass_variants_preserve_numerics_and_feed_consumed_ir(extent):
    original = source(extent)
    before = ir_text(original)
    rows = list(evaluate_variants(original, variants(), registry(), facts, output='ir'))
    assert [row.status for row in rows] == ['compiled'] * 3
    assert ir_text(original) == before
    assert rows[1].result.artifact != before
    assert rows[2].result.receipt['passes'][-1]['name'] == 'emit'
    for row in rows:
        assert row.result.artifact == ir_text(row.result.module)
        for value in (-33, 0, 19):
            assert execute(row.result.artifact, value) == execute(before, value)
        summary = row.summary()
        assert 'artifact' not in summary and 'registry' not in summary
        assert summary['source_sha256'] == ir_digest(original)
        assert summary['executed_passes']


def test_large_loop_stays_counted_and_feedback_is_incremental():
    touched = []
    passes = registry()
    emit = passes['emit']
    passes['emit'] = replace(emit, action=lambda s: touched.append(1) or emit.action(s))
    stream = evaluate_variants(source(10**12 + 1), variants(), passes, facts, output='ir')
    first = next(stream)
    assert len(touched) == 1
    assert first.result.artifact.count('"scf.for"') == 1
    rows = [first, *stream]
    assert len(touched) == 3
    assert all(len(row.result.artifact) < 10000 for row in rows)


def test_guards_skip_unknown_and_unsupported_without_executing_passes():
    paths = (
        PipelineVariant('unknown', ('emit',), (PropertyPredicate('capability', 'async_copy'),)),
        PipelineVariant('unsupported', ('emit',), (PropertyPredicate('operation', 'missing.op'),)),
        variants()[0],
    )
    rows = list(evaluate_variants(source(), paths, registry(), facts, output='ir'))
    assert [row.status for row in rows] == ['unknown', 'unsupported', 'compiled']
    assert rows[0].result is rows[1].result is None


def test_failure_discards_partial_ir_and_does_not_poison_next_alternative():
    passes = registry()
    def fail(session):
        session.module = source(2)
        raise ValueError('broken emission')
    passes['bad'] = RegisteredPass(PassSpec('bad', 0), fail)
    paths = (PipelineVariant('broken', ('bad', 'emit')), variants()[0])
    rows = list(evaluate_variants(source(13), paths, passes, facts, output='ir'))
    assert rows[0].status == 'failed' and rows[0].result is None
    assert 'broken emission' in rows[0].error
    assert execute(rows[1].result.artifact, 5) == (83,)


def test_invalid_late_path_refuses_before_any_actions():
    touched = []
    passes = registry()
    passes['emit'] = replace(passes['emit'], action=lambda s: touched.append(1) or {'ir': ''})
    with pytest.raises(ValueError, match='unknown registered'):
        list(evaluate_variants(source(), (variants()[0], PipelineVariant('bad', ('absent',))),
                               passes, facts, output='ir'))
    assert touched == []


def test_source_mutation_and_bad_clone_do_not_gain_results():
    original = source()
    with pytest.raises(ValueError, match='distinct private clone'):
        list(evaluate_variants(original, variants(), registry(), facts, output='ir', clone=lambda m: m))
    with pytest.raises(ValueError, match='clone differs'):
        list(evaluate_variants(original, variants(), registry(), facts, output='ir', clone=lambda m: source(1)))
    stream = evaluate_variants(original, variants(), registry(), facts, output='ir')
    next(stream)
    from xdsl.dialects.builtin import StringAttr
    original.attributes['test.change'] = StringAttr('changed')
    with pytest.raises(ValueError, match='source changed'):
        next(stream)


def test_shared_transform_prefix_runs_once_and_branches_rederive_analyses():
    source_ir = source(13)
    before = ir_text(source_ir)
    passes = registry()
    prefix_calls, analyses = [], []
    cleanup = passes['clean']
    def prepare(session):
        prefix_calls.append(session.module)
        cleanup.action(session)
        return {'prefix_analysis': [session.module]}
    passes['prepare'] = RegisteredPass(PassSpec('prepare', 0, produces=('prefix_analysis',)), prepare)
    def analyze(session):
        # A fresh per-branch prerequisite owns references into that branch only.
        value = [session.module]
        analyses.append(value)
        return {'live': value}
    passes['analyze'] = RegisteredPass(PassSpec('analyze', 1, produces=('live',), mutates_ir=False), analyze)
    passes['emit'] = replace(passes['emit'], spec=replace(passes['emit'].spec, requires=('live',)))
    def prepared_facts(session):
        with pytest.raises(ValueError, match='missing or stale'):
            session.analysis('prefix_analysis')
        return facts(session)
    paths = (PipelineVariant('prepared_baseline', ('analyze', 'emit')),
             PipelineVariant('unrolled', ('unroll4', 'analyze', 'emit')))
    rows = list(evaluate_variants(source_ir, paths, passes, prepared_facts,
                                 output='ir', prefix=('prepare',)))
    assert len(prefix_calls) == 1
    assert len(analyses) == 2 and analyses[0] is not analyses[1]
    assert analyses[0][0] is not analyses[1][0]
    assert ir_text(source_ir) == before
    for row in rows:
        assert row.status == 'compiled'
        summary = row.summary()
        assert summary['prepared_passes'] == ('prepare',)
        assert summary['prepared_ir_sha256'] != summary['source_sha256']
        assert summary['source_sha256'] == ir_digest(source_ir)
        for value in (-12, 0, 19):
            assert execute(row.result.artifact, value) == execute(before, value)


def test_prefix_analysis_cannot_satisfy_a_branch_prerequisite():
    seen = []
    passes = registry()
    passes['prepare'] = RegisteredPass(PassSpec('prepare', 0, produces=('live',), mutates_ir=False),
                                       lambda s: seen.append(1) or {'live': [s.module]})
    passes['emit'] = replace(passes['emit'], spec=replace(passes['emit'].spec, requires=('live',)))
    with pytest.raises(ValueError, match='prerequisites'):
        list(evaluate_variants(source(), (variants()[0],), passes, facts,
                               output='ir', prefix=('prepare',)))
    assert seen == []


def test_failed_prefix_aborts_before_any_branch():
    passes = registry()
    def fail(session):
        session.module = source(2)
        raise ValueError('shared preparation failed')
    passes['prepare'] = RegisteredPass(PassSpec('prepare', 0), fail)
    with pytest.raises(ValueError, match='shared preparation failed'):
        list(evaluate_variants(source(), variants(), passes, facts, output='ir', prefix=('prepare',)))


@pytest.mark.parametrize('changed', ['prepared', 'source', 'contracts'])
def test_prefix_and_input_contract_changes_cannot_be_reused(changed):
    from xdsl.dialects.builtin import StringAttr
    from merlin.transforms.pipeline_state import PipelineContext
    original = source()
    passes, prepared = registry(), []
    def remember(session):
        prepared.append(session.module)
        return {}
    passes['prepare'] = RegisteredPass(PassSpec('prepare', 0, mutates_ir=False), remember)
    context = PipelineContext(hardware='selected-hardware')
    stream = evaluate_variants(original, variants(), passes, facts, output='ir',
                               prefix=('prepare',), context=context)
    assert next(stream).status == 'compiled'
    if changed == 'contracts':
        object.__setattr__(context, 'hardware', 'changed-hardware')
    else:
        (prepared[0] if changed == 'prepared' else original).attributes['test.change'] = StringAttr('changed')
    with pytest.raises(ValueError, match='changed'):
        next(stream)


def test_prefix_stage_order_is_checked_before_actions():
    passes = registry()
    passes['late'] = RegisteredPass(PassSpec('late', 3, mutates_ir=False), lambda s: {})
    with pytest.raises(ValueError, match='precedes shared preparation'):
        list(evaluate_variants(source(), variants(), passes, facts, output='ir', prefix=('late',)))
