"""Readout batches retain source cover and finite-queue numerical execution."""

from dataclasses import replace

import pytest
from test_compact_dense import ADAPTER
from test_dense_panel_pipeline import setup

from merlin.compile.scheduling.dense_panel_pipeline import schedule_dense_panel_pipeline, verify_dense_panel_pipeline
from merlin.compile.scheduling.dense_panel_readout import batch_dense_panel_readouts
from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan
from merlin.sched.ir import evaluate, instances


def checked():
    p, r, s, b, a = setup()
    plan = make_dense_schedule_plan(p, r, s, ADAPTER, source_identity="a" * 64,
                                   dtypes=("i8", "i8", "i32"), static_budget=128,
                                   block_tiles=(b.m, b.k, b.n), pipeline_prefetch=True)
    result = schedule_dense_panel_pipeline(p, r, s, ADAPTER, a, name="readout",
        dtypes=("i8", "i8", "i32"), schedule_plan=plan, source_identity="a" * 64,
        external_buffers_disjoint=True)
    return p, r, s, b, a, result


def test_checked_batch_preserves_actual_data_stream_and_requires_store_admission():
    p, r, s, b, a, original = checked()
    a2 = replace(a, max_issued_readouts=4, pending_readouts=1,
                 readout_configuration="ordered_snapshot_before_store_issue")
    result = batch_dense_panel_readouts(original, p, b, r, s, ADAPTER, a2)

    def calls(kernel):
        return [(call.instr, tuple((key, evaluate(value, env)) for key, value in call.args))
                for call, env in instances(kernel) if call.instr != "merlin_dense_panel_drain"]

    assert calls(result.kernel) == calls(original.kernel)
    assert result.realization["readout_batching"]["removed_dynamic_drains"] > 0
    assert max(e["readout_commands"] for e in result.realization["epochs"]) == 4
    assert result.realization["source"]["output_tiles"] == p.tiles[0] * p.tiles[2]
    result.materialize_scf().verify()
    with pytest.raises(ValueError, match="readout exceeds"):
        verify_dense_panel_pipeline(result.kernel, p, b, r, s, ADAPTER, a, static_budget=128)
    with pytest.raises(ValueError, match="premises"):
        batch_dense_panel_readouts(original, p, b, r, s, ADAPTER,
                                   replace(a2, max_issued_loads=a2.max_issued_loads + 1))
    with pytest.raises(ValueError, match="premises"):
        replace(a, max_issued_readouts=4)


@pytest.mark.parametrize("biased,narrow", [(False, False), (True, True)])
def test_batched_readouts_with_tail_bias_and_reversed_finite_queue_completion(biased, narrow):
    from test_dense_panel_pipeline import test_reversed_completion_with_finite_queue_backpressure_matches_every_output

    test_reversed_completion_with_finite_queue_backpressure_matches_every_output(
        True, biased, narrow, "prefetch_rhs_between_columns", batch_readouts=True)


@pytest.mark.parametrize('retained', [False, True])
@pytest.mark.parametrize('uniform', [False, True])
def test_uniform_original_tiled_coefficients_batch_without_changing_source_calls(retained, uniform):
    from merlin.compile.scheduling.block_schedule import Store
    from merlin.compile.scheduling.dense_contracts import DenseTiledIntegerReadout, TiledIntegerStore
    from merlin.sched.ir import evaluate, instances

    if retained:
        from test_dense_retained_reduction import inputs
        p, r, s, b, a = inputs()
    else:
        p, r, s, b, a = setup()
    coefficients = [0.5]*p.n
    if not uniform:
        coefficients[-1] = 0.25
    readout = DenseTiledIntegerReadout.from_column_scales(coefficients, tile_columns=p.tile_n)

    def encode(op):
        if type(op) is TiledIntegerStore:
            base = ADAPTER.encode(Store(op.dram_row, op.dram_col, op.rows, op.cols, op.accumulator_row))
            return replace(base, instr=readout.instruction_name, args=(*base.args, ('readout_tile', op.readout_tile)))
        return ADAPTER.encode(op)

    def project(call):
        if call.instr == readout.instruction_name:
            op = ADAPTER.project(replace(call, instr='store',
                args=tuple((k, v) for k, v in call.args if k != 'readout_tile')))
            return TiledIntegerStore(op.dram_row, op.dram_col, op.rows, op.cols, op.accumulator_row,
                                     readout.identity, call.arg('readout_tile'))
        return ADAPTER.project(call)

    adapter = replace(ADAPTER, encode=encode, project=project)
    plan = make_dense_schedule_plan(p, r, s, adapter, source_identity='a'*64,
        dtypes=('i8', 'i8', 'i32'), static_budget=128, block_tiles=(b.m, b.k, b.n),
        output_readout=readout, pipeline_kind='retained_full_k' if retained else 'two_window',
        pipeline_prefetch=not retained)
    original = schedule_dense_panel_pipeline(p, r, s, adapter, a, name='source_uniform',
        dtypes=('i8', 'i8', 'i32'), schedule_plan=plan, source_identity='a'*64,
        external_buffers_disjoint=True, output_readout=readout,
        readout_qualification_identity='independent-original-i32-f32-rne-saturated-readout')
    selected = replace(a, max_issued_readouts=b.m*b.n, pending_readouts=1,
                       readout_configuration='ordered_snapshot_before_store_issue')
    if not uniform:
        with pytest.raises(ValueError, match='independent snapshot mapping'):
            batch_dense_panel_readouts(original, p, b, r, s, adapter, selected)
        return
    result = batch_dense_panel_readouts(original, p, b, r, s, adapter, selected)

    def data_calls(kernel):
        return [(call.instr, tuple((name, evaluate(value, env)) for name, value in call.args))
                for call, env in instances(kernel) if call.instr != 'merlin_dense_panel_drain']

    assert data_calls(result.kernel) == data_calls(original.kernel)
    assert result.realization['readout_batching']['removed_dynamic_drains'] > 0
    assert result.realization['readout_batching']['uniform_original_scale_bits'] == readout.column_scale_bits[0]
    assert result.output_readout.identity == readout.identity
    assert result.realization['source']['output_tiles'] == p.tiles[0]*p.tiles[2]
    result.revalidate()
