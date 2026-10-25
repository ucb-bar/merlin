"""Executed user-level split/interchange compositions through normal consumers."""

from dataclasses import replace

import pytest
from test_compact_dense import ADAPTER, fixture, schedule
from test_loop_schedule import SOURCE, trace
from test_scf_commands import record

from merlin.compile.scheduling.compact_dense import emit_compact_dense
from merlin.compile.scheduling.loop_schedule import propose_loop_transforms, tile_loop_nest_transforms
from merlin.compile.scheduling.scf_commands import materialize_scf_commands
from merlin.compile.scheduling.schedule_ir import (
    dense_schedule_text,
    make_dense_schedule_plan,
    parse_dense_schedule_plan,
)
from merlin.sched.ir import Kernel, Ptr, TensorArg, Var, call, loop, select
from merlin.sched.primitives import NotApplicable
from merlin.transforms.dense_planning import dense_blocks


def pointwise(extents=(6, 12)):
    return Kernel(
        "pointwise",
        (TensorArg("x", extents, "i8", "read"),),
        (loop("i", extents[0], loop("j", extents[1],
            call("consume", source=Ptr("x", Var("i") * extents[1] + Var("j")),
                 row=Var("i"), col=Var("j"), last=select("j", extents[1] - 1, 1, 0)))),),
    )


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_compound_current_paths_preserve_every_point_and_pointer(order):
    original = pointwise()
    requests = tile_loop_nest_transforms(original, ("i",), factors=(2, 3), outer_order=order)
    actual = propose_loop_transforms(original, requests)
    assert {r.action for r in requests} == {"split", "reorder"}
    assert trace(original) != trace(actual)
    assert sorted(map(repr, trace(original))) == sorted(map(repr, trace(actual)))
    assert len(trace(actual)) == 72
    # Original byte-addressed pointer operands survive the composed index
    # substitutions. The separate SCF family intentionally accepts indices only.
    index_original = replace(original, args=(), body=(loop("i", 6, loop("j", 12,
        call("consume", offset=Var("i") * 12 + Var("j"), last=select("j", 11, 1, 0)))),))
    index_requests = tile_loop_nest_transforms(index_original, ("i",), factors=(2, 3), outer_order=order)
    scf = materialize_scf_commands(propose_loop_transforms(index_original, index_requests))
    calls = record(scf)
    assert sorted(args["offset"] for _, args in calls) == list(range(72))
    assert sum(args["last"] for _, args in calls) == 6


def test_two_tile_orders_are_real_distinct_emitter_inputs_not_just_analysis_orders():
    original = pointwise()
    alternatives = [propose_loop_transforms(original,
        tile_loop_nest_transforms(original, ("i",), factors=(2, 3), outer_order=order))
        for order in ((0, 1), (1, 0))]
    assert alternatives[0].digest() != alternatives[1].digest()
    assert trace(alternatives[0]) != trace(alternatives[1])
    assert sorted(map(repr, trace(alternatives[0]))) == sorted(map(repr, trace(alternatives[1])))


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_real_dense_dialect_final_verifier_and_scf_emission_consume_composed_requests(order):
    p, r, s = fixture(shape=(8, 6, 16))
    r = replace(r, max_m_blocks=1, max_k_blocks=1, max_n_blocks=1)
    block = next(b for b in dense_blocks(p, r) if (b.m, b.k, b.n) == (1, 1, 1))
    raw = emit_compact_dense(p, block, s, ADAPTER, name="dense", dtypes=("i8", "i8", "i32"))
    requests = tile_loop_nest_transforms(raw, ("outer_m",), factors=(2, 2), outer_order=order)
    def plan(transforms):
        return parse_dense_schedule_plan(dense_schedule_text(make_dense_schedule_plan(
            p, r, s, ADAPTER, source_identity=SOURCE, dtypes=("i8", "i8", "i32"), static_budget=80,
            block_tiles=(1, 1, 1), loop_transforms=transforms)))
    ordinary = schedule(p, r, s, schedule_plan=plan(()), source_identity=SOURCE)
    actual = schedule(p, r, s, schedule_plan=plan(requests), source_identity=SOURCE)
    actual.revalidate()  # unchanged arithmetic, local lifetime and output cover checker
    assert actual.realization["schedule_ir"]["loop_transforms"]
    assert "schedule.loop_transform" in actual.realization["schedule_ir"]["text"]
    assert actual.kernel.digest() != ordinary.kernel.digest()
    before, after = (record(materialize_scf_commands(x.kernel)) for x in (ordinary, actual))
    assert before != after
    assert sorted(map(repr, before)) == sorted(map(repr, after))
    assert actual.realization["executed_instruction_calls"] == ordinary.realization["executed_instruction_calls"]


def test_composition_is_compact_for_huge_rectangles_and_names_do_not_capture_scopes():
    original = pointwise((10**9, 10**6))
    requests = tile_loop_nest_transforms(original, ("i",), factors=(8, 16), outer_order=(1, 0))
    actual = propose_loop_transforms(original, requests)
    assert actual.text().count("consume(") == 1
    assert len(requests) < 10 and len(actual.text()) < 2000
    unrelated = replace(original, body=(*original.body, loop("i_tile_0", 1, call("other"))))
    requests = tile_loop_nest_transforms(unrelated, ("i",), factors=(8, 16))
    assert all("i_tile_0" not in r.names for r in requests)
    propose_loop_transforms(unrelated, requests)


@pytest.mark.parametrize("extents", [(7, 12), (6, 13), (0, 12), (1, 12)])
def test_tails_and_empty_or_too_small_extents_refuse_without_moving_or_dropping_work(extents):
    with pytest.raises((NotApplicable, ValueError), match="divisible|checked|tail"):
        tile_loop_nest_transforms(pointwise(extents), ("i",), factors=(2, 3))


def test_nonperfect_bounds_invalid_permutations_static_budget_refuse_and_closed_events_survive():
    original = pointwise()
    imperfect = replace(original, body=(replace(original.body[0], body=(call("setup"), *original.body[0].body)),))
    with pytest.raises(NotApplicable, match="perfect"):
        tile_loop_nest_transforms(imperfect, ("i",), factors=(2, 3))
    for order in ((0, 0), (True, 0), (0,), [0, 1]):
        with pytest.raises(ValueError, match="permutations"):
            tile_loop_nest_transforms(original, ("i",), factors=(2, 3), outer_order=order)
    with pytest.raises(NotApplicable, match="budget"):
        tile_loop_nest_transforms(original, ("i",), factors=(2, 3), max_static_operations=4)
    asynchronous = Kernel("async", (), (loop("i", 4, loop("j", 6,
        call("launch", produces="done"), call("wait", awaits="done"))),))
    requests = tile_loop_nest_transforms(asynchronous, ("i",), factors=(2, 3))
    actual = propose_loop_transforms(asynchronous, requests)
    assert trace(actual) == trace(asynchronous)
    # Closed real completion pairs are not erased or replaced by new ordering laws.
    materialize_scf_commands(actual).verify()
