"""Executable split/fission proposals expose RHS stripes without crossing drains."""

from collections import Counter

import pytest
from test_compact_dense import ADAPTER
from test_dense_panel_pipeline import setup

from merlin.compile.scheduling.dense_panel_pipeline import (
    DRAIN,
    build_dense_panel_pipeline,
    dense_panel_rhs_stripes,
    schedule_dense_panel_pipeline,
)
from merlin.compile.scheduling.loop_schedule import propose_loop_transforms
from merlin.compile.scheduling.schedule_ir import (
    dense_schedule_text,
    make_dense_schedule_plan,
    parse_dense_schedule_plan,
)
from merlin.sched.ir import evaluate, instances


def data(kernel):
    return [
        (call.instr, tuple((key, evaluate(value, env)) for key, value in call.args)) for call, env in instances(kernel)
    ]


def test_actual_typed_stripe_decisions_preserve_source_and_complete_physical_pairs():
    p, r, storage, block, admission = setup()
    original = build_dense_panel_pipeline(
        p,
        block,
        r,
        storage,
        ADAPTER,
        name="pipeline",
        dtypes=("i8", "i8", "i32"),
        issue_order="prefetch_rhs_between_columns",
    )
    requests = dense_panel_rhs_stripes(original, ADAPTER, factor=2)
    proposed = propose_loop_transforms(original, requests)
    assert data(proposed) != data(original)
    assert Counter(data(proposed)) == Counter(data(original))
    assert sum(call.instr == DRAIN for call, _ in instances(proposed)) == sum(
        call.instr == DRAIN for call, _ in instances(original)
    )
    plan = make_dense_schedule_plan(
        p,
        r,
        storage,
        ADAPTER,
        source_identity="d" * 64,
        dtypes=("i8", "i8", "i32"),
        static_budget=128,
        block_tiles=(block.m, block.k, block.n),
        pipeline_prefetch=True,
        pipeline_issue_order="prefetch_rhs_between_columns",
        loop_transforms=requests,
    )
    plan = parse_dense_schedule_plan(dense_schedule_text(plan))
    result = schedule_dense_panel_pipeline(
        p,
        r,
        storage,
        ADAPTER,
        admission,
        name="pipeline",
        dtypes=("i8", "i8", "i32"),
        schedule_plan=plan,
        source_identity="d" * 64,
        external_buffers_disjoint=True,
    )
    assert result.kernel == proposed
    assert result.realization["schedule_ir"]["loop_transforms"]
    result.materialize_scf().verify()


def test_stripes_refuse_serial_completion_boundaries_and_unsupported_factor():
    p, r, s, block, _ = setup()
    for prefetch, order in ((False, "prefetch_rhs_between_columns"), (True, "compute_then_prefetch")):
        kernel = build_dense_panel_pipeline(
            p, block, r, s, ADAPTER, name="pipeline", dtypes=("i8", "i8", "i32"), prefetch=prefetch, issue_order=order
        )
        with pytest.raises(ValueError, match="mixed current-compute"):
            dense_panel_rhs_stripes(kernel, ADAPTER, factor=2)


@pytest.mark.parametrize("biased,narrow", [(False, False), (True, False), (True, True), (False, "tiled")])
def test_stripes_preserve_every_output_under_reversed_finite_queue_completion(monkeypatch, biased, narrow):
    import test_dense_panel_pipeline as existing

    original = existing.build_dense_panel_pipeline

    def striped(*args, **kwargs):
        kernel = original(*args, **kwargs)
        return propose_loop_transforms(kernel, dense_panel_rhs_stripes(kernel, args[4], factor=2))

    monkeypatch.setattr(existing, "build_dense_panel_pipeline", striped)
    existing.test_reversed_completion_with_finite_queue_backpressure_matches_every_output(
        True, biased, narrow, "prefetch_rhs_between_columns"
    )
