"""``plan_arena`` — the static memory plan, and the four ways it used to FAIL OPEN.

There were no tests for this module at all: no overlap property, no dynamic-dim case, no alignment
case. That matters more than usual here, because every one of its failure modes produced a plan that
looked complete. A short arena does not raise at plan time; it corrupts the heap on the board,
arbitrarily far from the cause, and the numbers may still come out right on the run that happens to
fit. Nothing consumes ``plan_arena`` yet, so these are the tests that have to exist BEFORE it is wired.
"""
from __future__ import annotations

import pytest

from merlin.xdsl_dialects.lowering.arena_plan import (ARENA_ALIGN, ArenaPlanError, _align_up,
                                                      _buf_bytes, _elem_bytes, plan_arena)
from merlin.xdsl_dialects.lowering.dispatch_program import Buffer, DispatchProgram, Node

DYNAMIC = -9223372036854775808          # MLIR ShapedType::kDynamic


def _prog(buffers, nodes, results, args=()):
    return DispatchProgram(entry="forward", args=list(args),
                           buffers={b.id: b for b in buffers}, nodes=list(nodes),
                           results=list(results))


def _chain(n_intermediates: int, shape=(64, 64), dtype="f32"):
    """A straight-line program: arg -> t0 -> t1 -> ... -> result. Each t_i dies at the next node, so a
    correct planner reuses ONE slot for all of them and a broken one uses n."""
    bufs = [Buffer(id="a", shape=list(shape), dtype=dtype, kind="arg", arg_index=0)]
    nodes, prev = [], "a"
    for i in range(n_intermediates):
        bid = f"t{i}"
        bufs.append(Buffer(id=bid, shape=list(shape), dtype=dtype, kind="intermediate"))
        nodes.append(Node(kind="dispatch", op=f"op{i}", inputs=[prev], outputs=[bid]))
        prev = bid
    bufs.append(Buffer(id="r", shape=list(shape), dtype=dtype, kind="intermediate"))
    nodes.append(Node(kind="dispatch", op="last", inputs=[prev], outputs=["r"]))
    return _prog(bufs, nodes, ["r"], args=[0])


# ---- the fail-open cases ------------------------------------------------------------------

def test_a_dynamic_dimension_is_refused_not_sized_as_one_element():
    """The retired behavior mapped any non-positive extent to 1, so tensor<?x768xf32> planned as 768
    elements: plan reports success, arena is short by the real extent."""
    with pytest.raises(ArenaPlanError) as e:
        _buf_bytes([DYNAMIC, 768], "f32")
    assert "dynamic extent" in str(e.value)
    # and it must reach plan_arena, not be swallowed there
    prog = _prog([Buffer(id="t", shape=[DYNAMIC, 768], dtype="f32", kind="intermediate")],
                 [Node(kind="dispatch", op="op", inputs=[], outputs=["t"])], [])
    with pytest.raises(ArenaPlanError):
        plan_arena(prog)


def test_an_unknown_dtype_is_refused_not_sized_as_f32():
    """Defaulting to 4 bytes halves the arena for an f64/i64 buffer and doubles it for an i8 one."""
    with pytest.raises(ArenaPlanError):
        _elem_bytes("some_future_type")
    # ...while every dtype the models DO use is sized exactly
    assert (_elem_bytes("f64"), _elem_bytes("i64")) == (8, 8)
    assert (_elem_bytes("bf16"), _elem_bytes("f16"), _elem_bytes("i16")) == (2, 2, 2)
    assert (_elem_bytes("i8"), _elem_bytes("i1")) == (1, 1)
    # a longest-suffix match, so bf16 is never read as f16 nor ui32 as i32
    assert _elem_bytes("ui32") == 4 and _elem_bytes("si8") == 1


def test_a_wrapped_element_type_is_refused_rather_than_read_as_one_scalar():
    """`vector<4xf32>` holds FOUR elements; reading it as one f32 under-sizes the buffer 4x."""
    with pytest.raises(ArenaPlanError):
        _elem_bytes("vector<4xf32>")


def test_every_offset_and_the_arena_total_are_aligned_to_what_the_allocator_gives_today():
    """The per-op path this replaces hands out 64-byte-aligned blocks (merlin_malloc.c bumps with
    align 64). A byte-tight plan silently weakens that for every buffer, and an under-aligned base is
    exactly the bug that shows up as wrong numbers on silicon and nowhere in simulation."""
    # sizes that are deliberately NOT multiples of 64
    prog = _prog(
        [Buffer(id="a", shape=[3], dtype="i8", kind="arg", arg_index=0),
         Buffer(id="t0", shape=[7], dtype="i8", kind="intermediate"),
         Buffer(id="t1", shape=[13], dtype="f32", kind="intermediate"),
         Buffer(id="r", shape=[5], dtype="i8", kind="intermediate")],
        [Node(kind="dispatch", op="o0", inputs=["a"], outputs=["t0"]),
         Node(kind="dispatch", op="o1", inputs=["a", "t0"], outputs=["t1"]),
         Node(kind="dispatch", op="o2", inputs=["t0", "t1"], outputs=["r"])],
        ["r"], args=[0])
    plan = plan_arena(prog)
    assert plan.offsets, "nothing was planned"
    for bid, off in plan.offsets.items():
        assert off % ARENA_ALIGN == 0, (bid, off)
    assert plan.arena_bytes % ARENA_ALIGN == 0
    assert plan.stats["align"] == ARENA_ALIGN
    assert _align_up(1) == ARENA_ALIGN and _align_up(ARENA_ALIGN) == ARENA_ALIGN


# ---- the property the planner exists for -------------------------------------------------

def test_live_buffers_never_overlap():
    """The correctness property, checked directly: two buffers whose live ranges intersect must not
    share a byte. This is what a memory planner IS, and it had no test."""
    prog = _chain(6)
    plan = plan_arena(prog)
    first_def, last_use = {}, {}
    for i, node in enumerate(prog.nodes):
        for b in node.inputs:
            last_use[b] = i
        for b in node.outputs:
            first_def.setdefault(b, i)
    placed = [(b, plan.offsets[b], plan.sizes[b]) for b in plan.offsets]
    for i, (bi, oi, si) in enumerate(placed):
        for bj, oj, sj in placed[i + 1:]:
            overlaps_bytes = oi < oj + sj and oj < oi + si
            live_i = (first_def.get(bi, 0), last_use.get(bi, first_def.get(bi, 0)))
            live_j = (first_def.get(bj, 0), last_use.get(bj, first_def.get(bj, 0)))
            live_together = live_i[0] <= live_j[1] and live_j[0] <= live_i[1]
            assert not (overlaps_bytes and live_together), (bi, bj, live_i, live_j, oi, si, oj, sj)


def test_a_straight_line_chain_reuses_one_slot_instead_of_n():
    """The reason the pass exists: ~4391 per-op mallocs on smolvla. In a chain each intermediate dies
    at the next node, so the arena must be O(1) slots, not O(n)."""
    one = _align_up(64 * 64 * 4)
    for n in (2, 6, 12):
        plan = plan_arena(_chain(n))
        assert plan.arena_bytes <= 2 * one, (n, plan.arena_bytes, one)
        assert plan.stats["reuse_factor"] >= 1.0
        assert plan.stats["n_intermediate_buffers"] == n      # the result is excluded


def test_results_and_args_stay_out_of_the_arena():
    prog = _chain(3)
    plan = plan_arena(prog)
    assert "a" not in plan.offsets, "an arg is bound from the arg table, not the arena"
    assert "r" not in plan.offsets, "a result is bound to the caller's buffer, not the arena"
