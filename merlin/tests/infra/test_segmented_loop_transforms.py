"""Executable typed loop decisions compose with segmented source/hazard checks."""

from test_compact_dense import ADAPTER

from merlin.compile.scheduling.dense_panel_pipeline import DRAIN
from merlin.compile.scheduling.dense_segmented_pipeline import schedule_dense_segmented_pipeline
from merlin.compile.scheduling.schedule_ir import (
    dense_schedule_text,
    make_dense_schedule_plan,
    parse_dense_schedule_plan,
)
from merlin.sched.ir import evaluate, instances


def consume(p, r, s, b, admission, requests):
    plan = make_dense_schedule_plan(
        p,
        r,
        s,
        ADAPTER,
        source_identity="b" * 64,
        dtypes=("i8", "i8", "i32"),
        static_budget=128,
        name="segmented",
        block_tiles=(b.m, b.k, b.n),
        pipeline_kind="retained_lhs_segmented_k",
        pipeline_prefetch=True,
        pipeline_issue_order="prefetch_rhs_between_columns",
        loop_transforms=requests,
    )
    return schedule_dense_segmented_pipeline(
        p,
        r,
        s,
        ADAPTER,
        admission,
        name="segmented",
        dtypes=("i8", "i8", "i32"),
        schedule_plan=parse_dense_schedule_plan(dense_schedule_text(plan)),
        source_identity="b" * 64,
        external_buffers_disjoint=True,
    )


def epochs(kernel):
    result, pending = [], []
    for call, env in instances(kernel):
        if call.instr == DRAIN:
            result.append(pending)
            pending = []
        else:
            pending.append((call.instr, tuple((k, evaluate(v, env)) for k, v in call.args)))
    assert not pending
    return result
