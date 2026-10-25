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
