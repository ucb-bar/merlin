"""Replay of real Voyager compiler output into an ordered event trace (merlin.baselines.voyager_ir).

The fixtures are the pinned compiler's own output (see merlin/tests/data/voyager_ir/AGENT.md). The
assertions are structural facts of Voyager's schedule that a bridge must reproduce exactly -- which tile
is loaded into which slot at which step, when a load is skipped because its tile is already resident,
and how a split reduction is combined -- plus the two ways the replay must refuse instead of guessing.
"""
from __future__ import annotations

import copy
from collections import Counter

import pytest

from merlin.baselines.voyager_ir import (Copy, FusedCompute, SemaphoreViolation, UnsupportedConstruct,
                                         Wait, load_model, replay)
from merlin.common.paths import merlin_dir

FIXTURES = merlin_dir() / "tests" / "data" / "voyager_ir"


def _trace(name: str):
    return replay(load_model(FIXTURES / name / "model.json"))


def test_a_double_buffered_grid_alternates_slots_and_loads_the_resident_input_once() -> None:
    trace = _trace("lin64")
    loads = [c for c in trace.of(Copy) if c.is_load]
    stores = [c for c in trace.of(Copy) if c.is_store]

    by_operand = Counter(c.src.box.node for c in loads)
    assert by_operand == {"x_preprocess": 1, "fc_weight_t": 4, "fc_bias": 4}
    weight_slots = [c.dst.slot for c in loads if c.src.box.node == "fc_weight_t"]
    assert weight_slots == [0, 1, 0, 1]
    # Slot 1 of a banked allocation sits one bank stride above slot 0: that is the placement the
    # bridge has to honour, so it must come out of the replay as a concrete byte address.
    weight = [c for c in loads if c.src.box.node == "fc_weight_t"]
    assert weight[1].dst.slot_address - weight[0].dst.slot_address == weight[0].dst.box.bank_stride > 0

    computes = trace.of(FusedCompute)
    assert len(computes) == 4 and len(stores) == 4
    assert {c.anchor.target for c in computes} == {"quantized_ops::linear"}
    assert computes[0].tail == ("quantized_ops::dequantize", "aten::relu")
    assert [s.indices for s in stores] == [(0,), (1,), (2,), (3,)]


def test_split_k_commits_an_add_into_the_first_partial_and_reloads_only_changed_tiles() -> None:
    trace = _trace("split_k_256x512x256")
    computes = trace.of(FusedCompute)
    tails = Counter(c.tail for c in computes)
    assert tails == {("quantized_ops::dequantize",): 8,
                     ("quantized_ops::dequantize", "aten::add"): 8}
    # K is the innermost grid axis: every first partial is immediately followed by its add.
    assert [len(c.tail) for c in computes] == [1, 2] * 8

    stores = [c for c in trace.of(Copy) if c.is_store]
    assert len(stores) == 8                      # one per (M, N) output tile, after its last K split
    assert len(set(s.indices for s in stores)) == 8

    # The interstellar mapping travels with the compute, innermost loop first, per level.
    assert computes[0].tiling == ((("LOOP_OX", 32), ("LOOP_OC", 2)),
                                  (("LOOP_IC", 16), ("LOOP_OX", 2), ("LOOP_OC", 4)))


def test_every_wait_and_commit_is_matched_by_an_earlier_signal() -> None:
    for name in ("lin64", "split_k_256x512x256"):
        trace = _trace(name)
        assert trace.of(Wait), name
        assert all(count == 0 for count in trace.semaphores.values()), (name, trace.semaphores)


def test_an_unmodelled_scalar_operation_is_refused_not_skipped() -> None:
    model = load_model(FIXTURES / "lin64" / "model.json")
    broken = copy.deepcopy(model)
    loop = next(op for op in broken["ops"] if "loop" in op)
    scalar_op = next(op for op in loop["loop"]["for_loop"]["body"]["ops"]
                     if op.get("prim", {}).get("target") == "add")
    scalar_op["prim"]["target"] = "bitwise_xor"
    with pytest.raises(UnsupportedConstruct, match="bitwise_xor"):
        replay(broken)


def test_a_wait_with_no_prior_signal_is_a_violation() -> None:
    model = load_model(FIXTURES / "lin64" / "model.json")
    broken = copy.deepcopy(model)
    # Drop the input load that precedes the loop: the first commit then depends on a semaphore that
    # nothing signalled, which a real pipeline would deadlock on.
    broken["ops"] = [op for op in broken["ops"]
                     if op.get("prim", {}).get("target") != "voyager::async_copy"]
    with pytest.raises(SemaphoreViolation):
        replay(broken)
    assert replay(broken, check_semaphores=False).of(FusedCompute)
