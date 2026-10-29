"""Bounded public exploration consumes actual native products, not just receipts."""
import copy
import json
from dataclasses import replace

import numpy as np
import pytest
from xdsl.dialects import func
from xdsl.dialects.builtin import StringAttr

from merlin.compile import fusion_workflow as F
from merlin.transforms.certificates import ir_digest
from test_multioutput_fusion import source_program
from test_physical_fusion import fixture as physical_fixture
from test_shared_input_schedule import fixture as shared_fixture, execute

OPTIONS = dict(capability_contract={}, numerical_policy_sha256='a' * 64)


@pytest.mark.parametrize('width,shape', [(8, (3, 5)), (16, (3, 5)), (32, (7, 2))])
def test_fusion_materialization_counts_payload_and_retains_external_observers(width, shape):
    module, operations = source_program(width=width, shape=shape)
    before = ir_digest(module)
    report = F.discover(module, **OPTIONS)
    full = next(row for row in report['candidates'] if row['route'] == 'multioutput')
    payload = shape[0] * shape[1] * width // 8
    assert full['materialization']['potential_saved_logical_bytes'] == payload * 3
    frontier = report['materialization_frontier']
    assert frontier['selected_candidate_indices'] == [full['candidate_index']]
    assert frontier['potential_saved_logical_bytes'] == payload * 3
    assert not frontier['rewrite_authority'] and not frontier['performance_ranking']
    assert ir_digest(module) == before
    # Returning the original producer makes its materialization externally
    # observable. It must not be treated as an eliminated private tensor.
    ret = next(op for op in module.walk() if isinstance(op, func.ReturnOp))
    ret.operands = [*ret.operands, operations[0].results[0]]
    ret.parent_op().update_function_type()
    observed = F.discover(module, **OPTIONS)
    assert all(row['materialization']['potential_saved_logical_bytes'] == 0 for row in observed['candidates'])
    assert observed['materialization_frontier']['selected_candidate_indices'] == []


def test_materialization_conflicts_compare_combinations_and_bound_large_search():
    def row(index, operations, benefit):
        return dict(candidate_index=index, operation_ordinals=operations, status='unknown',
                    materialization=dict(potential_saved_logical_bytes=benefit))
    report = dict(schema='merlin.fusion_workflow.v1', binding_sha256='a' * 64,
                  candidates=[row(0, [1, 2], 100), row(1, [1], 60), row(2, [2], 60), row(3, [3], 20)])
    selected = F.plan_materialization_frontier(report)
    assert selected['selected_candidate_indices'] == [1, 2, 3]
    assert selected['potential_saved_logical_bytes'] == 140
    bounded = F.plan_materialization_frontier(report, exact_cluster_limit=1)
    assert bounded['selected_candidate_indices'] == [0, 3]
    assert 'greedy' in bounded['conflict_cluster_methods']
    unknown = row(4, [4], None)
    report['candidates'].append(unknown)
    extended = F.plan_materialization_frontier(report)
    assert extended['selected_candidate_indices'] == [1, 2, 3]
    assert extended['excluded_from_byte_selection'][-1]['candidate_index'] == 4
    assert extended['evaluation_order'][-1] == 4  # Retained for other benefits/native analysis.


def test_materialization_selected_fanout_reaches_actual_emission(tmp_path):
    from test_multioutput_fusion import model, execute_actual_bytes

    module, operations, selection = model(tmp_path, shape=(3, 5))
    report = F.discover(module, **OPTIONS)
    indices = report['materialization_frontier']['selected_candidate_indices']
    row = next(row for row in report['candidates'] if row['candidate_index'] in indices)
    emitted = F.realize_candidate(module, row, **selection, **OPTIONS)
    assert emitted.receipt['status'] == 'applied'
    # Replay the actual emitted bytes with an independent scalar byte machine,
    # including saturation/overflow and output guards. This is a neutral target
    # fixture, not Gemmini hardware qualification.
    execute_actual_bytes(emitted.artifact, 16, False, tmp_path)


def test_pagination_is_lossless_bound_and_nonfatal():
    module, _ = source_program()
    before = ir_digest(module)
    full = F.discover(module, **OPTIONS)
    page = F.discover(module, max_candidates=1, **OPTIONS)
    assert page['status'] == 'budget_exhausted'
    first_cursor = page['next_cursor']
    rows = list(page['candidates'])
    while page['next_cursor'] is not None:
        page = F.discover(module, cursor=json.loads(json.dumps(page['next_cursor'])),
                          max_candidates=1, **OPTIONS)
        rows.extend(page['candidates'])
    assert rows == full['candidates']
    assert page['status'] == 'complete' and ir_digest(module) == before
    for changes in ({'capability_contract': {'changed': True}},
                    {'numerical_policy_sha256': 'b' * 64}, {'max_region_operations': 3}):
        with pytest.raises(ValueError, match='cursor differs'):
            F.discover(module, cursor=first_cursor, **dict(OPTIONS, **changes))
    module.attributes['changed'] = StringAttr('source')
    with pytest.raises(ValueError, match='cursor differs'):
        F.discover(module, cursor=first_cursor, **OPTIONS)


def test_incremental_frontier_retains_fanout_external_outputs():
    module, operations = source_program()
    before = ir_digest(module)
    rows = F.discover(module, max_region_operations=3, **OPTIONS)['candidates']
    ordinal = next(i for i, op in enumerate(module.walk()) if op is operations[0])
    growing = [row for row in rows if row['detail'].get('growth') == 'forward_frontier'
               and row['operation_ordinals'][0] == ordinal]
    assert [len(row['operation_ordinals']) for row in growing] == [2, 3]
    # The partial producer-consumer frontier retains the producer consumed by
    # the other branch, and the selected consumer result returned by the graph.
    assert len(growing[0]['live_outputs']) == 2
    bounded = F.discover(module, max_region_operations=2, **OPTIONS)
    assert any(row['detail'].get('growth_status') == 'region_budget_exhausted'
               for row in bounded['candidates'])
    unsupported = F.realize_candidate(module, growing[-1], max_region_operations=3, **OPTIONS)
    assert unsupported.receipt['status'] == 'unsupported'
    assert unsupported.artifact is module and ir_digest(module) == before
    ret = next(op for op in module.walk() if isinstance(op, func.ReturnOp))
    ret.operands = [*ret.operands, operations[0].results[0]]
    ret.parent_op().update_function_type()
    declined = next(row for row in F.discover(module, **OPTIONS)['candidates'] if row['route'] == 'multioutput')
    result = F.realize_candidate(module, declined, **OPTIONS)
    assert result.receipt['status'] == 'declined' and result.unfused is module
    assert result.receipt['reason']


def test_shared_prefix_native_tail_arithmetic_and_later_epilogue():
    module, group, tile, storage, resources = shared_fixture(epilogue=True)
    before = ir_digest(module)
    rows = [row for row in F.discover(module, **OPTIONS)['candidates'] if row['route'] == 'shared_input']
    assert [tuple(row['detail']['widths']) for row in rows] == [(7, 3), (7, 3, 9)]
    for row in rows:
        # Rows may cross an offline JSON boundary; realization rederives source.
        result = F.realize_candidate(module, json.loads(json.dumps(row)), primitive_tile=tile,
                                     storage=storage, resources=resources, **OPTIONS)
        assert result.receipt['status'] == 'applied'
        selected = result.artifact.group
        rng = np.random.default_rng(len(selected.widths))
        lhs = rng.integers(-128, 128, (selected.m, selected.k), dtype=np.int64)
        weights = [rng.integers(-128, 128, (selected.k, n), dtype=np.int64) for n in selected.widths]
        outputs = execute(result.artifact.operations, (module, selected, tile, storage, resources), lhs, weights)
        for actual, weight in zip(outputs, weights, strict=True):
            np.testing.assert_array_equal(actual, (lhs @ weight).astype(np.int32).astype(np.int64))
    assert ir_digest(module) == before  # Unselected sibling and epilogue remain.
    tampered = copy.deepcopy(rows[0])
    tampered['detail']['widths'] = (7, 4)
    with pytest.raises(ValueError, match='current bound discovery row'):
        F.realize_candidate(module, tampered, **OPTIONS)


def test_resource_refusal_and_budget_exhaustion_do_not_poison_fallback():
    module, group, tile, storage, resources = shared_fixture()
    row = next(row for row in F.discover(module, **OPTIONS)['candidates'] if row['route'] == 'shared_input')
    def realize(source, **limits):
        return F.realize_candidate(source, row, primitive_tile=tile, storage=storage,
                                   resources=replace(resources, **limits), **OPTIONS)
    emitted = []
    def emit(product, session):
        emitted.append(product.receipt['status'])
        return product.artifact
    results = list(F.evaluate_fusions(module, (
        F.FusionAlternative('budget', lambda source: realize(source, candidate_budget=1)),
        F.FusionAlternative('capacity', lambda source: realize(source, operand_bytes=1)),
        F.FusionAlternative('tails', lambda source: realize(source, partial_output_tiles=False)),
        F.FusionAlternative('legal', realize),
    ), emit))
    assert [r.status for r in results] == ['compiled', 'budget_exhausted', 'no-match', 'no-match', 'compiled']
    assert emitted == ['unfused', 'applied']
    assert results[0].evaluation.result.artifact is results[0].realization.unfused
    assert results[-1].evaluation.result.artifact is results[-1].realization.artifact
    # Explicit choices must not pay or fail the unselected full product search.
    selected = F.realize_candidate(module, row, primitive_tile=tile, storage=storage,
        resources=replace(resources, candidate_budget=1), selected_block=(1, 1, 1), **OPTIONS)
    assert selected.receipt['status'] == 'applied'


def test_alternatives_feed_normal_emission_actual_transformed_calls():
    from merlin.compile.scheduling.schedule_pipeline import compile_schedule, ScheduleBudgets
    kernel, objects, banks, adapter, capability, encoding, limits, options = physical_fixture()
    seen = []
    def realize(source):
        return F.realize_adjacent(source, objects, banks, adapter, (capability,), **options)
    def emit(product, session):
        # This is the ordinary allocation, physical effect, encoding and decoded
        # conformance pipeline, not a receipt-only fake emitter.
        seen.append((session.module, product.artifact))
        return compile_schedule(product.artifact, product.objects, banks, adapter, encoding, limits,
                                budgets=ScheduleBudgets(max_fusions=0), **options)
    results = list(F.evaluate_fusions(kernel, (F.FusionAlternative('adjacent', realize),), emit,
        clone=copy.deepcopy, fingerprint=lambda k: k.digest(), verify_ir=lambda k: None,
        ir_level='finite_schedule', unfused_objects=objects))
    assert [r.status for r in results] == ['compiled', 'compiled'], [r.reason for r in results]
    programs = [r.evaluation.result.artifact.program for r in results]
    decoded = [tuple(call for segment in p.segments for call in encoding.decode_calls(segment)) for p in programs]
    assert [tuple(call.instr for call in calls) for calls in decoded] == [('producer', 'consumer'), ('fused',)]
    assert [p.report['instruction_count'] for p in programs] == [2, 1]
    assert seen[0][0] is not kernel and seen[1][0] is not seen[0][0]
    assert seen[1][0] is not seen[1][1] and len(kernel.body) == 2


def test_immutable_source_and_round_limit():
    module, _ = source_program()
    alternatives = tuple(F.FusionAlternative(str(i), lambda source: None) for i in range(16))
    with pytest.raises(ValueError, match='fifteen'):
        list(F.evaluate_fusions(module, alternatives, lambda product, session: product.artifact))
    def corrupt(source):
        source.attributes['changed'] = StringAttr('illegal')
        return F.Realization(source, source, {'status': 'applied'})
    results = list(F.evaluate_fusions(module, (F.FusionAlternative('corrupt', corrupt),),
                                     lambda product, session: product.artifact))
    assert [r.status for r in results] == ['compiled', 'failed']
    assert 'unchanged private unfused source' in results[-1].reason
    assert 'changed' not in module.attributes


def test_preparation_requests_bind_individual_ssa_values_even_without_reuse():
    from merlin.llvmlower.tensor_preparation_identity import TensorPreparationRequest
    module, operations = source_program()
    op = operations[0]
    a = TensorPreparationRequest(op.operands[0], op, 'b' * 64)
    b = TensorPreparationRequest(op.operands[1], op, 'b' * 64)
    page = F.discover(module, max_candidates=1, preparation_requests=(a,), **OPTIONS)
    with pytest.raises(ValueError, match='cursor differs'):
        F.discover(module, cursor=page['next_cursor'], preparation_requests=(b,), **OPTIONS)


def test_paged_fanout_candidate_emits_native_images_and_all_outputs(tmp_path):
    from test_multioutput_fusion import model, execute_actual_bytes
    module, operations, selection = model(tmp_path)
    row = next(row for row in F.discover(module, **OPTIONS)['candidates'] if row['route'] == 'multioutput')
    def realize(source):
        return F.realize_candidate(source, row, **OPTIONS, **selection)
    def emit(product, session):
        return product.artifact.images() if product.receipt['status'] == 'applied' else str(product.artifact)
    results = list(F.evaluate_fusions(module, (F.FusionAlternative('fanout', realize),), emit))
    assert [r.status for r in results] == ['compiled', 'compiled'], [r.reason for r in results]
    compiled = results[-1].realization.artifact
    assert results[-1].evaluation.result.artifact == compiled.images()
    execute_actual_bytes(compiled, 16, False, tmp_path)


def test_incorrect_numeric_realization_cannot_reach_emitter():
    kernel, objects, banks, adapter, capability, encoding, limits, options = physical_fixture()
    def changed_semantics(call):
        effects = adapter.project(call)
        if call.instr != 'fused':
            return effects
        changed = effects.elementwise.function_ir.replace(b'32752', b'32751')
        return replace(effects, elementwise=replace(effects.elementwise, function_ir=changed))
    def realize(source):
        return F.realize_adjacent(source, objects, banks, replace(adapter, project=changed_semantics),
                                  (capability,), **options)
    emitted = []
    def emit(product, session):
        emitted.append(product.receipt['status'])
        return encoding.encode(product.artifact)
    results = list(F.evaluate_fusions(kernel, (F.FusionAlternative('wrong_bias', realize),), emit,
        clone=copy.deepcopy, fingerprint=lambda k: k.digest(), verify_ir=lambda k: None,
        ir_level='finite_schedule', unfused_objects=objects))
    assert results[0].status == 'compiled' and results[-1].status in {'failed', 'no-match'}
    assert emitted == ['unfused']


def test_contract_change_between_variants_is_refused():
    from merlin.transforms.pipeline_state import PipelineContext
    module, _ = source_program()
    context = PipelineContext(hardware='public-hardware', numerical_policy='exact-numerics')
    iterator = F.evaluate_fusions(module, (F.FusionAlternative('unused', lambda source: None),),
                                 lambda product, session: product.artifact, context=context)
    assert next(iterator).status == 'compiled'
    object.__setattr__(context, 'hardware', 'changed')
    with pytest.raises(ValueError, match='source changed between alternatives'):
        next(iterator)
