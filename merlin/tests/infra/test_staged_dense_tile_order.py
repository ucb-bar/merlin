"""Whole-output-tile traversal changes real checked effects, not source names."""

from dataclasses import replace

import pytest

from merlin.compile.scheduling.staged_dense import (
    AccumulatorReadout,
    AccumulatorZero,
    CompletionFence,
    ComputeConsume,
    DescriptorIssue,
    OperandLoad,
    StagedDenseContract,
    StagedDenseProblem,
    StagedDenseRewrite,
    eliminate_redundant_rhs_loads,
    order_output_tiles_n_before_m,
    rewrite_staged_dense,
)


def selected(**changes):
    return replace(
        StagedDenseContract(
            "fixed-facts",
            "fixed-effects",
            "signed-raw",
            "fixture-only",
            4,
            8,
            4,
            8,
            18,
            32,
            "read_at_compute_consume",
            "data_commands_preserve_pending_descriptor",
            "raw_accumulator_identity",
            True,
            True,
            False,
        ),
        **changes,
    )


def program(problem):
    operations = [CompletionFence()]
    for m in range(0, problem.m, 4):
        for n in range(0, problem.n, 4):
            mr, nr = min(4, problem.m - m), min(4, problem.n - n)
            operations.append(AccumulatorZero(0, mr, nr))
            for k in range(0, problem.k, 4):
                kr = min(4, problem.k - k)
                operations.extend(
                    (
                        OperandLoad("lhs", m, k, 0, mr, kr),
                        CompletionFence(),
                        OperandLoad("rhs", k, n, 4, kr, nr),
                        DescriptorIssue(4, 0, mr, kr, nr, True),
                        ComputeConsume(0, mr, kr),
                        CompletionFence(),
                    )
                )
            operations.extend((AccumulatorReadout(0, m, n, mr, nr), CompletionFence()))
    operations.append(CompletionFence())
    return tuple(operations)


@pytest.mark.parametrize("m", [1, 4, 5, 7, 8, 9])
@pytest.mark.parametrize("k,n", [(1, 1), (4, 4), (4, 5), (4, 8), (5, 7), (8, 8)])
def test_checked_complete_tile_order_covers_independent_tails_and_full_reductions(m, k, n):
    problem = StagedDenseProblem(m, k, n)
    before = program(problem)
    order = StagedDenseRewrite("new_tile_traversal", order_output_tiles_n_before_m)
    after, report = rewrite_staged_dense(before, problem=problem, contract=selected(), rewrites=(order,))
    outputs = [op for op in after if type(op) is AccumulatorReadout]
    assert [(op.output_col, op.output_row) for op in outputs] == sorted(
        (op.output_col, op.output_row) for op in outputs
    )
    assert report["checks"][-1]["products"] == m * k * n
    assert report["checks"][-1]["outputs"] == m * n
    assert len(after) == len(before)
    assert sum(type(op) is CompletionFence for op in after) == sum(type(op) is CompletionFence for op in before)
    assert program(problem) == before


def test_order_and_residency_are_noncommuting_under_the_same_complete_product_checker():
    problem = StagedDenseProblem(8, 4, 8)
    before = program(problem)
    order = StagedDenseRewrite("n_before_m", order_output_tiles_n_before_m)
    reuse = StagedDenseRewrite("rhs_residency", eliminate_redundant_rhs_loads)
    left, left_report = rewrite_staged_dense(before, problem=problem, contract=selected(), rewrites=(order, reuse))
    right, right_report = rewrite_staged_dense(before, problem=problem, contract=selected(), rewrites=(reuse, order))
    assert left != right
    assert len(left) == len(right) - 2
    assert left_report["checks"][-1]["products"] == right_report["checks"][-1]["products"] == 256
    assert left_report["checks"][-1]["outputs"] == right_report["checks"][-1]["outputs"] == 64


@pytest.mark.parametrize("premise", ["external_buffers_disjoint", "rhs_immutable"])
def test_absent_ownership_never_proposes_output_reordering(premise):
    problem = StagedDenseProblem(8, 4, 8)
    before = program(problem)
    assert order_output_tiles_n_before_m(before, problem, selected(**{premise: False})) is before


@pytest.mark.parametrize("damage", ["missing_zero", "overlapping_zero", "missing_fence", "duplicate_output"])
def test_incomplete_or_ambiguous_tiles_are_refused(damage):
    problem = StagedDenseProblem(8, 4, 8)
    before = list(program(problem))
    if damage == "missing_zero":
        del before[1]
    elif damage == "overlapping_zero":
        before.insert(2, before[1])
    elif damage == "missing_fence":
        index = next(i for i, op in enumerate(before) if type(op) is AccumulatorReadout)
        del before[index + 1]
    else:
        indexes = [i for i, op in enumerate(before) if type(op) is AccumulatorReadout]
        before[indexes[1]] = before[indexes[0]]
    with pytest.raises(ValueError):
        order_output_tiles_n_before_m(tuple(before), problem, selected())


def fence_runs_program():
    """Nonuniform real inter-tile fences plus separately anchored endpoints."""
    base = program(StagedDenseProblem(8, 4, 8))
    blocks = []
    at = 1
    while at < len(base) - 1:
        end = next(i for i in range(at, len(base)) if type(base[i]) is AccumulatorReadout) + 2
        blocks.append(base[at:end])
        at = end
    source_order = (3, 0, 2, 1)
    source_blocks = tuple(
        (*blocks[index], *((CompletionFence(),) * extra))
        for index, extra in zip(source_order, (1, 3, 2, 0), strict=True)
    )
    prefix, suffix = (CompletionFence(),) * 3, (CompletionFence(),) * 4
    before = (*prefix, *(op for block in source_blocks for op in block), *suffix)
    by_coordinate = sorted(
        source_blocks,
        key=lambda block: next((op.output_col, op.output_row) for op in block if type(op) is AccumulatorReadout),
    )
    expected = (*prefix, *(op for block in by_coordinate for op in block), *suffix)
    return before, expected


def test_nonuniform_complete_fence_runs_move_intact_and_endpoints_stay_anchored():
    before, expected = fence_runs_program()
    after, report = rewrite_staged_dense(
        before,
        problem=StagedDenseProblem(8, 4, 8),
        contract=selected(),
        rewrites=(StagedDenseRewrite("n_before_m", order_output_tiles_n_before_m),),
    )
    assert after == expected and after != before
    assert len(after) == len(before)
    assert sum(type(op) is CompletionFence for op in after) == sum(type(op) is CompletionFence for op in before)
    assert report["checks"][-1]["outputs"] == 64
    assert report["checks"][-1]["products"] == 256


def test_extra_fences_do_not_absorb_unknown_intertile_effects():
    before, _ = fence_runs_program()
    before = list(before)
    boundary = next(i for i, op in enumerate(before) if type(op) is AccumulatorReadout)
    before.insert(boundary + 2, object())
    with pytest.raises(ValueError):
        order_output_tiles_n_before_m(tuple(before), StagedDenseProblem(8, 4, 8), selected())


def test_intertile_fence_runs_require_readout_completion():
    before, _ = fence_runs_program()
    before = list(before)
    readout = next(i for i, op in enumerate(before) if type(op) is AccumulatorReadout)
    end = readout + 1
    while type(before[end]) is CompletionFence:
        end += 1
    assert type(before[end]) is AccumulatorZero
    del before[readout + 1 : end]
    with pytest.raises(ValueError):
        rewrite_staged_dense(
            tuple(before),
            problem=StagedDenseProblem(8, 4, 8),
            contract=selected(),
            rewrites=(StagedDenseRewrite("n_before_m", order_output_tiles_n_before_m),),
        )
