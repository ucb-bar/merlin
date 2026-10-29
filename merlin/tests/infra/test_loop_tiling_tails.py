"""Exact ragged tiling composes existing typed decisions and real consumers."""

from collections import Counter
from dataclasses import replace

import pytest
from test_compact_dense import ADAPTER, fixture, schedule
from test_loop_schedule import SOURCE, trace
from test_scf_commands import record

from merlin.compile.scheduling.compact_dense import emit_compact_dense
from merlin.compile.scheduling.compact_verification import verify_compact_dense
from merlin.compile.scheduling.loop_schedule import propose_loop_transforms, tile_loop_nest_transforms
from merlin.compile.scheduling.scf_commands import materialize_scf_commands
from merlin.compile.scheduling.schedule_ir import (
    dense_schedule_text,
    make_dense_schedule_plan,
    parse_dense_schedule_plan,
)
from merlin.sched.ir import Kernel, Loop, Ptr, TensorArg, Var, call, loop, select
from merlin.sched.primitives import NotApplicable
from merlin.transforms.dense_planning import dense_blocks


def pointwise(m, n):
    return Kernel("ragged", (TensorArg("x", (m*n,), "i8", "read"),), (
        loop("i", m, loop("j", n, call("work", row=Var("i"), column=Var("j"),
            pointer=Ptr("x", Var("i")*n+Var("j")),
            last=select("j", n-1, 1, 0)))),))


@pytest.mark.parametrize("shape", [(8, 12), (5, 7), (1, 19), (17, 1), (101, 7)])
@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_all_original_points_predicates_and_pointer_offsets_survive(shape, order):
    original = pointwise(*shape)
    requests = tile_loop_nest_transforms(original, ("i",), factors=(4, 3),
                                         outer_order=order, tail_policy="cut")
    actual = propose_loop_transforms(original, requests)
    assert Counter(map(repr, trace(actual))) == Counter(map(repr, trace(original)))
    assert len(trace(actual)) == shape[0]*shape[1]
    assert len(requests) <= 32 and actual.text().count("work(") <= 4
    # No padding, dynamic mask, modulus, division or dynamic enumeration.
    assert not {"div", "mod", "mask"}.intersection(r.action for r in requests)


def test_third_axis_reduction_keeps_exact_source_k_order_for_every_original_output():
    original = Kernel("reduction", (), (loop("m", 5, loop("n", 8, loop("k", 6,
        call("mac", m=Var("m"), n=Var("n"), k=Var("k"))))),))
    requests = tile_loop_nest_transforms(original, ("m",), factors=(2, 2, 3),
                                         outer_order=(2, 0, 1), tail_policy="cut")
    actual = propose_loop_transforms(original, requests)
    def reductions(program):
        result = {}
        for operation in trace(program):
            a = dict(operation.args)
            result.setdefault((a["m"], a["n"]), []).append(a["k"])
        return result
    assert reductions(actual) == reductions(original)
    assert record(materialize_scf_commands(actual)) != record(materialize_scf_commands(original))


def test_huge_ragged_domains_stay_counted_without_instance_enumeration(monkeypatch):
    from merlin.sched import primitives

    monkeypatch.setattr(primitives, "_instance_keys", lambda *_: pytest.fail("no dynamic enumeration"))
    original = pointwise(10**9+1, 10**6+3)
    requests = tile_loop_nest_transforms(original, ("i",), factors=(8, 16), tail_policy="cut")
    actual = propose_loop_transforms(original, requests)
    assert len(actual.text()) < 8000 and actual.text().count("work(") == 4
    assert len(requests) < 20


@pytest.mark.parametrize("shape", [(10, 6, 20), (1, 6, 20)])
def test_dense_typed_decisions_and_unchanged_source_storage_checker_consume_tails(shape):
    p, resources, storage = fixture(shape=shape)
    resources = replace(resources, max_m_blocks=1, max_k_blocks=1, max_n_blocks=1)
    block = next(b for b in dense_blocks(p, resources) if (b.m, b.k, b.n) == (1, 1, 1))
    raw = emit_compact_dense(p, block, storage, ADAPTER, name="dense", dtypes=("i8", "i8", "i32"))
    requests = tile_loop_nest_transforms(raw, ("outer_m",), factors=(2, 3), tail_policy="cut")
    def plan(transforms):
        return parse_dense_schedule_plan(dense_schedule_text(make_dense_schedule_plan(
            p, resources, storage, ADAPTER, source_identity=SOURCE, dtypes=("i8", "i8", "i32"),
            static_budget=80, block_tiles=(1, 1, 1), loop_transforms=transforms)))
    before = schedule(p, resources, storage, schedule_plan=plan(()), source_identity=SOURCE)
    after = schedule(p, resources, storage, schedule_plan=plan(requests), source_identity=SOURCE)
    after.revalidate()
    assert "partition" in {r.action for r in requests}
    if shape[0] > 1:
        assert "fission" in {r.action for r in requests}
    assert after.realization["schedule_ir"]["loop_transforms"]
    a, b = [record(materialize_scf_commands(x.kernel)) for x in (before, after)]
    assert Counter(map(repr, a)) == Counter(map(repr, b))
    if shape[0] > 1:
        assert a != b
    assert after.realization["executed_instruction_calls"] == before.realization["executed_instruction_calls"]
    # A rectangle count is no storage/alias grant. The same final checker
    # refuses an actual mapping that aliases the original live operands.
    aliased = replace(storage, rhs=replace(storage.rhs, backing=storage.lhs.backing,
                                         unit_bytes=storage.lhs.unit_bytes, base=storage.lhs.base))
    with pytest.raises(ValueError):
        verify_compact_dense(after.kernel, p, resources, aliased, ADAPTER, static_budget=80)


def test_unsupported_shell_budget_event_escape_and_policy_refuse_without_mutating_source():
    original = pointwise(5, 7)
    before = original.digest()
    with pytest.raises(ValueError, match="tail policy"):
        tile_loop_nest_transforms(original, ("i",), factors=(2, 3), tail_policy="pad")
    with pytest.raises(NotApplicable, match="budget"):
        tile_loop_nest_transforms(original, ("i",), factors=(2, 3), tail_policy="cut", max_static_operations=8)
    imperfect = replace(original, body=(replace(original.body[0], body=(call("unknown"),*original.body[0].body)),))
    with pytest.raises(NotApplicable, match="perfect"):
        tile_loop_nest_transforms(imperfect, ("i",), factors=(2, 3), tail_policy="cut")
    escaped = Kernel("bad", (), (call("launch", produces="outside"),
        Loop("i",5,(Loop("j",7,(call("consume",awaits="outside"),)),))))
    requests = tile_loop_nest_transforms(escaped, ("i",), factors=(2, 3), tail_policy="cut")
    with pytest.raises(ValueError, match="event|scope|completion"):
        materialize_scf_commands(propose_loop_transforms(escaped, requests))
    assert original.digest() == before
