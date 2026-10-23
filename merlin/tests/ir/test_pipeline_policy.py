from dataclasses import replace

import pytest

from merlin.transforms.certificates import ir_text, parse_ir
from merlin.transforms.pipeline import PipelineOptions, apply_common_pipeline
from merlin.transforms.pipeline_policy import (
    GraphPredicate, PipelinePolicy, PipelineVariant, PolicyFacts, PropertyPredicate, RegisteredPass,
    ResourcePredicate, ShapePredicate, execute_policy, registry_metadata,
)
from merlin.transforms.pipeline_state import PassSpec, PipelineContext, PipelineSession


def source():
    return parse_ir('''builtin.module { func.func @f() -> i32 {
      %a = arith.constant 3 : i32
      %b = arith.constant 4 : i32
      %sum = arith.addi %a, %b : i32
      func.return %sum : i32
    }}''')


def registry():
    def normalize(session):
        # Real shared transformation broker, explicitly unverified planning here.
        result = apply_common_pipeline(session.module,
            PipelineOptions(64, reuse_buffers=False), plan_only=True)
        session.module = result.module
        return {}
    return {
        'normalize': RegisteredPass(PassSpec('normalize', 0), normalize),
        'emit': RegisteredPass(PassSpec('emit', 1, produces=('emitted_ir',), mutates_ir=False),
                               lambda session: {'emitted_ir': ir_text(session.module)}),
    }


def policy():
    return PipelinePolicy((
        PipelineVariant('integer', ('normalize', 'emit'), (
            PropertyPredicate('element_type', 'i32'), ShapePredicate(0, 0, multiple_of=4))),
        PipelineVariant('fallback', ('emit',)),
    ), 'fallback')


def test_real_shared_transform_changes_consumed_emitted_ir_and_keeps_source_private():
    original = source()
    session = PipelineSession(original.clone())
    facts = PolicyFacts(session.stamp(), shapes=((8,),), element_types=('i32',))
    result = execute_policy(session, policy(), registry(), facts, output='emitted_ir')
    assert 'arith.addi' not in result.artifact
    assert 'arith.addi' in ir_text(original)
    assert result.artifact == ir_text(result.module)
    assert [v['name'] for v in result.receipt['passes']] == ['normalize', 'emit']
    assert result.receipt['before']['source_sha256'] != result.receipt['after']['source_sha256']
    assert result.receipt['selected_variant'] == 'integer'
    assert [v['name'] for v in result.receipt['host_pass_profile']] == ['normalize', 'emit']


@pytest.mark.parametrize('fails', [False, True])
def test_pass_profile_separates_action_from_framework_and_preserves_failure(monkeypatch, fails):
    import merlin.transforms.pipeline_state as state

    ticks = iter((0, 2, 5, 12))
    monkeypatch.setattr(state, 'perf_counter_ns', lambda: next(ticks))
    session = PipelineSession(source())
    before = session.stamp()
    invocations = []

    def action():
        invocations.append(True)
        if fails:
            raise RuntimeError('failed transformation')
        return {}

    if fails:
        with pytest.raises(RuntimeError, match='failed transformation'):
            session.run(PassSpec('inspect', 0, mutates_ir=False), action)
        with pytest.raises(ValueError, match='session failed'):
            session.stamp()
    else:
        session.run(PassSpec('inspect', 0, mutates_ir=False), action)
        assert session.stamp() == before
    assert invocations == [True]
    assert session.host_pass_profile == [dict(
        name='inspect', status='failed' if fails else 'completed', action_invoked=True,
        action_wall_ns=3, framework_wall_ns=9, total_wall_ns=12, scope='host_pass_invocation')]


@pytest.mark.parametrize('shape,status', [((None,), 'unknown'), ((7,), 'unsupported')])
def test_unknown_and_tail_select_explicit_fallback(shape, status):
    session = PipelineSession(source())
    facts = PolicyFacts(session.stamp(), shapes=(shape,), element_types=('i32',))
    result = execute_policy(session, policy(), registry(), facts, output='emitted_ir')
    assert result.receipt['fallback_selected']
    assert result.receipt['decisions'][0]['status'] == status
    assert 'arith.addi' in result.artifact


def test_contract_predicates_require_selected_identity_and_stale_facts_refuse():
    session = PipelineSession(source())
    facts = PolicyFacts(session.stamp(), capabilities=(('async_dma', True),),
                        resources=(('scratch_bytes', 4096),),
                        numerical_semantics=frozenset({'wrapping_integer'}))
    cap = PropertyPredicate('capability', 'async_dma')
    resource = ResourcePredicate('scratch_bytes', 1024)
    numeric = PropertyPredicate('numerical_semantics', 'wrapping_integer')
    assert cap.match(facts) is resource.match(facts) is numeric.match(facts) is None
    session.context = PipelineContext(hardware='h', resources='r', numerical_policy='n')
    with pytest.raises(ValueError, match='current-IR'):
        execute_policy(session, policy(), registry(), facts, output='emitted_ir')
    facts = replace(facts, stamp=session.stamp())
    assert cap.match(facts) is resource.match(facts) is numeric.match(facts) is True
    with pytest.raises(ValueError, match='unsupported semantic'):
        PropertyPredicate('capsule_name', 'private')


def test_dependency_and_invalidation_refuse_before_actions():
    session = PipelineSession(source())
    seen = []
    passes = registry()
    passes['analyze'] = RegisteredPass(PassSpec('analyze', 0, produces=('live',), mutates_ir=False),
                                      lambda session: seen.append(True) or {'live': 1})
    passes['normalize'] = replace(passes['normalize'], spec=PassSpec('normalize', 0))
    passes['emit'] = replace(passes['emit'], spec=PassSpec('emit', 1, requires=('live',),
                                                         produces=('emitted_ir',), mutates_ir=False))
    invalid = PipelinePolicy((PipelineVariant('fallback', ('analyze', 'normalize', 'emit')),), 'fallback')
    with pytest.raises(ValueError, match='prerequisites'):
        execute_policy(session, invalid, passes, PolicyFacts(session.stamp()), output='emitted_ir')
    assert not seen and not session.records


def test_failed_transform_does_not_retry_fallback_or_emit():
    session = PipelineSession(source())
    passes = registry()
    def broken(session):
        raise ValueError('cannot preserve selected numerics')
    passes['normalize'] = replace(passes['normalize'], action=broken)
    with pytest.raises(ValueError, match='preserve selected numerics'):
        execute_policy(session, policy(), passes,
                       PolicyFacts(session.stamp(), shapes=((4,),), element_types=('i32',)),
                       output='emitted_ir')
    with pytest.raises(ValueError, match='session failed'):
        session.stamp()


def test_counted_loop_route_remains_compact_and_registry_reports_real_contract():
    module = parse_ir('''builtin.module {
      func.func @loop() {
        %lo = arith.constant 0 : index
        %hi = arith.constant 1000000000 : index
        %step = arith.constant 1 : index
        scf.for %i = %lo to %hi step %step { scf.yield }
        func.return
      }
    }''')
    session = PipelineSession(module, ir_level='compact')
    spec = PassSpec('compact_emit', 0, produces=('emitted_ir',), mutates_ir=False, ir_level='compact')
    passes = {'compact_emit': RegisteredPass(spec, lambda s: {'emitted_ir': ir_text(s.module)})}
    chosen = PipelinePolicy((PipelineVariant('compact', ('compact_emit',)),), 'compact')
    result = execute_policy(session, chosen, passes, PolicyFacts(session.stamp()), output='emitted_ir')
    assert result.artifact.count('scf.for') == 1
    assert '1000000000' in result.artifact
    assert len(result.artifact) < 1000
    assert registry_metadata(passes)[0]['ir_level'] == 'compact'
    assert registry_metadata(passes)[0]['produces'] == ('emitted_ir',)


def test_bounded_alternatives_and_final_output_required():
    with pytest.raises(ValueError, match='sixteen'):
        PipelinePolicy(tuple(PipelineVariant(str(i), ('emit',)) for i in range(17)), '0')
    session = PipelineSession(source())
    with pytest.raises(ValueError, match='final output'):
        execute_policy(session, policy(), registry(), PolicyFacts(session.stamp()), output='old_artifact')


def test_same_policy_preserves_live_io_on_distinct_graphs():
    from xdsl.interpreter import Interpreter
    from xdsl.interpreters.arith import ArithFunctions
    from xdsl.interpreters.func import FuncFunctions
    from xdsl.dialects import func
    def run(module, arguments):
        interpreter = Interpreter(module)
        interpreter.register_implementations(ArithFunctions())
        interpreter.register_implementations(FuncFunctions())
        return interpreter.call_op('f', arguments)
    fanout = parse_ir('''builtin.module { func.func @f(%x: i32) -> (i32, i32) {
      %zero = arith.constant 0 : i32
      %sum = arith.addi %x, %zero : i32
      %product = arith.muli %sum, %x : i32
      func.return %sum, %product : i32, i32
    }}''')
    for module, inputs in ((source(), ()), (fanout, (9,))):
        session = PipelineSession(module.clone())
        function = next(op for op in module.walk() if isinstance(op, func.FuncOp))
        inputs_before, outputs_before = function.function_type.inputs, function.function_type.outputs
        facts = PolicyFacts(session.stamp(), shapes=((4,),), element_types=('i32',),
                            operations=frozenset(op.name for op in module.walk()),
                            graph=(('live_in_count', len(inputs_before)), ('live_out_count', len(outputs_before))))
        result = execute_policy(session, policy(), registry(), facts, output='emitted_ir')
        emitted = parse_ir(result.artifact)
        after = next(op for op in emitted.walk() if isinstance(op, func.FuncOp))
        assert (after.function_type.inputs, after.function_type.outputs) == (inputs_before, outputs_before)
        assert run(emitted, inputs) == run(module, inputs)
        assert GraphPredicate('live_out_count', at_least=1).match(facts) is True
        assert GraphPredicate('max_fanout', at_least=2).match(facts) is None
    with pytest.raises(ValueError, match='workload identities'):
        PolicyFacts(session.stamp(), graph=(('model_name', 1),))
