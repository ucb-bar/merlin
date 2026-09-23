"""A gather splits into runs that one multi-row load each can move, without changing a byte.

The runs are what a target reads when it loads several DRAM rows at a configured stride: joined back
row by row they must reproduce the gather exactly, and they must be maximal -- two neighbouring runs that
one load could have moved would leave the cheaper stream on the table.
"""

from __future__ import annotations

import random

import pytest

from merlin.compile.scheduling import (
    LHS,
    ConvContraction,
    Geometry,
    Knobs,
    Load,
    gather_runs,
    schedule_convolution,
)

GEOMETRY = Geometry(
    block=16, operand_rows=16384, operand_bank_rows=4096, accumulator_rows=1024, separate_accumulator_space=True
)


def _expand(runs):
    rows = []
    for run in runs:
        for i in range(run.rows):
            rows.append(None if run.source is None else (run.source[0] + i * run.row_step, run.source[1]))
    return tuple(rows)


def _joinable(left, right):
    if left.source is None or right.source is None:
        return left.source is None and right.source is None
    if left.source[1] != right.source[1]:
        return False
    if left.rows == 1 and right.rows == 1:
        return right.source[0] > left.source[0]
    step = left.row_step if left.rows > 1 else right.row_step
    return (
        step > 0
        and right.source[0] == left.source[0] + left.rows * step
        and (right.rows == 1 or right.row_step == step)
    )


CONVS = [
    ConvContraction(1, 8, 8, 4, 3, 3, 8, padding=(1, 1, 1, 1)),
    ConvContraction(1, 8, 8, 64, 3, 3, 32, stride=(2, 2), padding=(1, 1, 1, 1)),
    ConvContraction(2, 7, 9, 20, 3, 3, 40, padding=(1, 1, 1, 1)),
    ConvContraction(1, 9, 9, 5, 3, 3, 17, padding=(2, 2, 2, 2), dilation=(2, 2)),
    ConvContraction(1, 6, 6, 33, 1, 1, 20),
]


@pytest.mark.parametrize("conv", CONVS, ids=lambda c: f"{c.in_h}x{c.in_w}x{c.ci}_s{c.stride[0]}_d{c.dilation[0]}")
def test_the_runs_of_every_real_gather_join_back_to_it_and_are_maximal(conv):
    schedule = schedule_convolution(conv, GEOMETRY, Knobs())
    gathers = [op.gather for op in schedule.ops if isinstance(op, Load) and op.role == LHS]
    assert gathers
    for gather in gathers:
        runs = gather_runs(gather)
        assert _expand(runs) == gather
        assert [r.offset for r in runs] == [sum(x.rows for x in runs[:i]) for i in range(len(runs))]
        for left, right in zip(runs, runs[1:]):
            assert not _joinable(left, right), (left, right)


def test_a_stride_two_row_of_pixels_is_one_run_at_step_two():
    conv = ConvContraction(1, 8, 8, 16, 3, 3, 16, stride=(2, 2), padding=(1, 1, 1, 1))
    schedule = schedule_convolution(conv, GEOMETRY, Knobs())
    steps = {
        run.row_step
        for op in schedule.ops
        if isinstance(op, Load) and op.role == LHS
        for run in gather_runs(op.gather)
        if run.source is not None and run.rows > 1
    }
    assert steps == {2}


def test_coalescing_cuts_the_loads_of_the_measured_probe():
    """8x8x64 -> 32, 3x3 s1: 2,304 one-row gathers become a few hundred runs."""
    conv = ConvContraction(1, 8, 8, 64, 3, 3, 32, padding=(1, 1, 1, 1))
    schedule = schedule_convolution(conv, GEOMETRY, Knobs())
    gathers = [op.gather for op in schedule.ops if isinstance(op, Load) and op.role == LHS]
    rows = sum(len(g) for g in gathers)
    runs = sum(len(gather_runs(g)) for g in gathers)
    assert rows == 36 * 4 * 16 and runs < rows // 4


def test_random_gathers_round_trip():
    rng = random.Random(7)
    for _ in range(500):
        gather = []
        row = rng.randrange(0, 50)
        for _ in range(rng.randrange(1, 17)):
            choice = rng.random()
            if choice < 0.2:
                gather.append(None)
            else:
                row += rng.choice((1, 1, 2, 3, -5))
                gather.append((max(row, 0), rng.choice((0, 0, 16))))
        runs = gather_runs(tuple(gather))
        assert _expand(runs) == tuple(gather)
        assert all(r.rows >= 1 and (r.row_step > 0 or r.rows == 1 or r.source is None) for r in runs)
