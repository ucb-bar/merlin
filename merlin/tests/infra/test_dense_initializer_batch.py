"""Disjoint ACC initialization batches preserve original data and drain before consumers."""
from dataclasses import replace

import pytest
from test_compact_dense_bias import ADAPTER
from test_dense_retained_reduction import inputs

from merlin.compile.scheduling.compact_dense import DenseBiasSource
from merlin.compile.scheduling.dense_panel_pipeline import (
    schedule_dense_panel_pipeline,
    verify_dense_retained_reduction,
)
from merlin.compile.scheduling.dense_panel_readout import batch_dense_panel_initializers
from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan
from merlin.sched.ir import Const, Loop, evaluate, instances


def checked(shape=(9, 14, 25)):
    p, r, s, b, a = inputs(shape=shape)
    p = replace(p, bias=True)
    a = replace(a, hardware_identity=ADAPTER.configuration_identity,
                numerical_policy_identity=ADAPTER.numerical_policy_identity)
    bias = DenseBiasSource(p.n, 'i32')
    plan = make_dense_schedule_plan(p, r, s, ADAPTER, source_identity='a'*64,
        dtypes=('i8', 'i8', 'i32'), static_budget=128, block_tiles=(b.m, b.k, b.n),
        bias_source=bias, pipeline_kind='retained_full_k', pipeline_prefetch=False)
    result = schedule_dense_panel_pipeline(p, r, s, ADAPTER, a, name='biased',
        dtypes=('i8', 'i8', 'i32'), schedule_plan=plan, source_identity='a'*64,
        external_buffers_disjoint=True, bias_source=bias)
    return p, r, s, b, a, bias, result


def selected(a, count=4):
    return replace(a, max_issued_initializations=count,
                   initialization_configuration='ordered_snapshot_before_disjoint_accumulator_issue')


@pytest.mark.parametrize('shape', [(9, 14, 25), (5, 6, 13)])
def test_checked_initializer_groups_retain_every_original_call_and_source_boundary(shape):
    p, r, s, b, a, bias, original = checked(shape)
    result = batch_dense_panel_initializers(original, p, b, r, s, ADAPTER, selected(a))

    def calls(kernel):
        return [(c.instr, tuple((k, evaluate(v, env)) for k, v in c.args))
                for c, env in instances(kernel) if c.instr != 'merlin_dense_panel_drain']

    assert calls(original.kernel) == calls(result.kernel)
    assert result.realization['initializer_batching']['removed_dynamic_drains'] > 0
    assert max(g['initializations_per_epoch'] for g in result.realization['initializer_batching']['groups']) == 4
    assert result.bias_source == bias
    result.revalidate()
    result.materialize_scf().verify()
    with pytest.raises(ValueError, match='initializer exceeds'):
        verify_dense_retained_reduction(result.kernel, p, b, r, s, ADAPTER, a,
                                       static_budget=128, bias_source=bias)
    with pytest.raises(ValueError, match='premises'):
        batch_dense_panel_initializers(original, p, b, r, s, ADAPTER,
                                       replace(selected(a), max_issued_loads=a.max_issued_loads+1))


def test_initializer_effect_admission_and_actual_aliases_refuse():
    p, r, s, b, a, bias, original = checked()
    with pytest.raises(ValueError, match='premises'):
        replace(a, max_issued_initializations=4)
    result = batch_dense_panel_initializers(original, p, b, r, s, ADAPTER, selected(a))

    def alias(body):
        return tuple(replace(x, body=alias(x.body)) if type(x) is Loop else
            replace(x, args=tuple((name, Const(s.accumulator.base) if name == 'dst' else value)
                                  for name, value in x.args)) if x.instr == 'bias_init' else x for x in body)

    broken = replace(result.kernel, body=alias(result.kernel.body))
    with pytest.raises(ValueError, match='overlapping in-flight destinations'):
        verify_dense_retained_reduction(broken, p, b, r, s, ADAPTER, selected(a),
                                       static_budget=128, bias_source=bias)
    with pytest.raises(ValueError, match='source|checked'):
        replace(result, admission=selected(a, 5)).revalidate()
