"""Shared schedule verification must catch live-state faults before execution."""

from dataclasses import replace
from itertools import permutations

import numpy as np
import pytest

from merlin.compile.scheduling.block_schedule import (
    AXES,
    NEST,
    BlockScheduleError,
    Compute,
    Contraction,
    ConvContraction,
    Geometry,
    Knobs,
    Load,
    Preload,
    Store,
    check_residency,
    execute,
    schedule_contraction,
    schedule_convolution,
)


def _schedule():
    return schedule_contraction(
        Contraction(7, 10, 9), Geometry(4, 256, 128, 128, True),
        Knobs(lookahead_steps=0, load_grouping=NEST),
    )


def _mutate(schedule, kind, change, occurrence=0):
    ops = list(schedule.ops)
    index = [i for i, op in enumerate(ops) if isinstance(op, kind)][occurrence]
    ops[index] = change(ops[index])
    return replace(schedule, ops=tuple(ops))


@pytest.mark.parametrize("order", list(permutations(AXES)))
@pytest.mark.parametrize("edge", [3, 7])
def test_matmul_and_gathered_convolution_accept_all_loop_orders(order, edge):
    geometry = Geometry(edge, 4096, 2048, 1024, True)
    knobs = Knobs(loop_order=order)
    check_residency(schedule_contraction(Contraction(8, 11, 10), geometry, knobs))
    check_residency(schedule_convolution(ConvContraction(1, 4, 5, 5, 2, 2, 8), geometry, knobs))


@pytest.mark.parametrize("occurrence", [0, 1])
def test_store_must_read_the_completed_accumulator_for_its_output(occurrence):
    schedule = _schedule()
    stores = [op for op in schedule.ops if isinstance(op, Store)]
    wrong = stores[1 - occurrence].accumulator_row
    corrupted = _mutate(schedule, Store, lambda op: replace(op, accumulator_row=wrong), occurrence)
    with pytest.raises(BlockScheduleError, match="accumulator"):
        check_residency(corrupted)


@pytest.mark.parametrize("occurrence,accumulate", [(0, True), (6, False)])
def test_initialization_and_accumulation_boundaries_are_preserved(occurrence, accumulate):
    corrupted = _mutate(_schedule(), Preload, lambda op: replace(op, accumulate=accumulate), occurrence)
    with pytest.raises(BlockScheduleError, match="accumulat"):
        check_residency(corrupted)


def test_dropped_compute_cannot_be_hidden_by_valid_transfer_counts():
    schedule = _schedule()
    ops = list(schedule.ops)
    index = next(i for i, op in enumerate(ops) if isinstance(op, Compute))
    del ops[index]
    with pytest.raises(BlockScheduleError, match="preload|reduction"):
        check_residency(replace(schedule, ops=tuple(ops)))


@pytest.mark.parametrize("duplicate", [False, True])
def test_output_coverage_is_complete_and_unique(duplicate):
    schedule = _schedule()
    store = next(op for op in schedule.ops if isinstance(op, Store))
    ops = schedule.ops + (store,) if duplicate else tuple(op for op in schedule.ops if op is not store)
    with pytest.raises(BlockScheduleError, match="output|store"):
        check_residency(replace(schedule, ops=ops))


def test_weight_residency_cannot_be_invented_by_dropping_the_first_preload():
    corrupted = _mutate(_schedule(), Preload, lambda op: replace(op, weight_row=None, weight_block=None))
    with pytest.raises(BlockScheduleError, match="resident|weight"):
        check_residency(corrupted)


def test_fresh_weight_flag_must_agree_with_the_preload():
    corrupted = _mutate(_schedule(), Compute, lambda op: replace(op, fresh_weights=False))
    with pytest.raises(BlockScheduleError, match="weight"):
        check_residency(corrupted)


def test_incomplete_output_cannot_be_overwritten_by_another_output():
    corrupted = _mutate(_schedule(), Preload, lambda op: replace(op, accumulator_row=0), occurrence=1)
    with pytest.raises(BlockScheduleError, match="still live"):
        check_residency(corrupted)


def test_completed_stores_allow_accumulator_reuse_in_a_different_layout():
    base = _schedule()
    schedule = schedule_contraction(
        base.contraction, base.geometry, Knobs(loop_order=("m", "n", "k")),
    )
    ops = tuple(replace(op, accumulator_row=0) if isinstance(op, (Preload, Store)) else op
                for op in schedule.ops)
    reused = replace(schedule, ops=ops, geometry=replace(schedule.geometry, accumulator_rows=4))
    check_residency(reused)
    lhs = np.arange(70).reshape(7, 10) % 7 - 3
    rhs = np.arange(90).reshape(10, 9) % 5 - 2
    np.testing.assert_array_equal(execute(reused, lhs, rhs), lhs @ rhs)


def test_reduction_order_is_checked_even_when_all_operands_are_loaded():
    base = _schedule()
    schedule = schedule_contraction(base.contraction, base.geometry, Knobs(lookahead_steps=None))
    ops = list(schedule.ops)
    first = next(i for i, op in enumerate(ops) if isinstance(op, Compute) and op.input_block == (0, 0))
    second = next(i for i, op in enumerate(ops) if isinstance(op, Compute) and op.input_block == (0, 1))
    ops[first - 1:first + 1], ops[second - 1:second + 1] = ops[second - 1:second + 1], ops[first - 1:first + 1]
    with pytest.raises(BlockScheduleError, match="reduction"):
        check_residency(replace(schedule, ops=tuple(ops)))


def test_compute_cannot_read_columns_the_load_did_not_initialize():
    corrupted = _mutate(_schedule(), Load, lambda op: replace(op, cols=op.cols - 1))
    with pytest.raises(BlockScheduleError, match="width"):
        check_residency(corrupted)


@pytest.mark.parametrize("kind", [Load, Preload, Compute, Store])
@pytest.mark.parametrize("rows", [0, -1, 5])
def test_invalid_operation_extents_are_refused(kind, rows):
    corrupted = _mutate(_schedule(), kind, lambda op: replace(op, rows=rows))
    with pytest.raises(BlockScheduleError):
        check_residency(corrupted)


def test_shared_backing_and_obvious_output_alias_need_an_explicit_model():
    schedule = _schedule()
    with pytest.raises(BlockScheduleError, match="alias|backing"):
        check_residency(replace(schedule, geometry=replace(schedule.geometry, separate_accumulator_space=False)))
    with pytest.raises(BlockScheduleError, match="alias"):
        check_residency(replace(schedule, contraction=replace(schedule.contraction, out=schedule.contraction.lhs)))


def test_unknown_operations_have_no_implicit_effect_contract():
    with pytest.raises(BlockScheduleError, match="unknown|unsupported"):
        check_residency(replace(_schedule(), ops=_schedule().ops + (object(),)))


def test_gather_cannot_write_more_rows_than_its_declared_footprint():
    schedule = _schedule()
    first = next(op for op in schedule.ops if isinstance(op, Load) and op.role == "lhs")
    gather = tuple((first.dram_row + i, first.dram_col) for i in range(first.rows + 1))
    ops = tuple(replace(op, gather=gather) if op is first else
                replace(op, input_gather=gather) if isinstance(op, Compute) and op.input_block == first.block else op
                for op in schedule.ops)
    with pytest.raises(BlockScheduleError, match="gather"):
        check_residency(replace(schedule, ops=ops))
