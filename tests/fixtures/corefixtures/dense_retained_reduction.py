from dataclasses import replace

from corefixtures.compact_dense import ADAPTER, fixture
from merlin.compiler.schedule.compact_dense import emit_compact_dense
from merlin.compiler.schedule.dense_panel_pipeline import (
    DenseRetainedReductionAdmission,
    dense_retained_operand_hoists,
    schedule_dense_panel_pipeline,
)
from merlin.compiler.schedule.schedule_ir import (
    dense_schedule_text,
    make_dense_schedule_plan,
    parse_dense_schedule_plan,
)
from merlin.compiler.transforms.dense_planning import dense_blocks


def inputs(shape=(9, 14, 25), tile=(2, 3, 4)):
    p, r, s = fixture(shape=shape, tile=tile)
    s = replace(s, lhs=replace(s.lhs, capacity=128), rhs=replace(s.rhs, capacity=128))
    r = replace(
        r,
        operand_bytes=128 * (s.lhs.unit_bytes + s.rhs.unit_bytes),
        max_m_blocks=2,
        max_n_blocks=2,
        max_k_blocks=p.tiles[1],
    )
    b = next((b for b in dense_blocks(p, r) if (b.m, b.k, b.n) == (2, p.tiles[1], 2)))
    a = DenseRetainedReductionAdmission(
        ADAPTER.configuration_identity,
        ADAPTER.numerical_policy_identity,
        "independent-neutral-ordered-commit-interpreter-v1",
        b.m * b.k * b.n,
        (b.m + b.n) * b.k,
        2,
        2,
    )
    return (p, r, s, b, a)


def chosen(p, r, s, b, a, *, role="lhs", order=("m", "n"), hoist=True, compute_order=None, compute_unroll=None):
    raw = emit_compact_dense(p, b, s, ADAPTER, name="retained", dtypes=("i8", "i8", "i32"), outer_order=order)
    transforms = dense_retained_operand_hoists(raw, ADAPTER, role=role) if hoist else ()
    if compute_order is not None:
        from merlin.compiler.schedule.dense_panel_pipeline import dense_retained_compute_order
        from merlin.compiler.schedule.loop_schedule import propose_loop_transforms

        current = propose_loop_transforms(raw, transforms)
        transforms += dense_retained_compute_order(current, ADAPTER, order=compute_order)
    if compute_unroll is not None:
        from merlin.compiler.schedule.dense_panel_pipeline import dense_retained_compute_unroll
        from merlin.compiler.schedule.loop_schedule import propose_loop_transforms

        current = propose_loop_transforms(raw, transforms)
        axis, factor = compute_unroll
        transforms += dense_retained_compute_unroll(current, ADAPTER, axis=axis, factor=factor)
    plan = make_dense_schedule_plan(
        p,
        r,
        s,
        ADAPTER,
        source_identity="d" * 64,
        dtypes=("i8", "i8", "i32"),
        static_budget=128,
        block_tiles=(b.m, b.k, b.n),
        outer_order=order,
        loop_transforms=transforms,
        pipeline_kind="retained_full_k",
        pipeline_prefetch=False,
    )
    return schedule_dense_panel_pipeline(
        p,
        r,
        s,
        ADAPTER,
        a,
        name="retained",
        dtypes=("i8", "i8", "i32"),
        schedule_plan=parse_dense_schedule_plan(dense_schedule_text(plan)),
        source_identity="d" * 64,
        external_buffers_disjoint=True,
        static_budget=128,
    )
