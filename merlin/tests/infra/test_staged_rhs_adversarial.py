"""Read-only review of original staged source/product verification and RHS policy."""

from dataclasses import replace

import pytest
from test_staged_dense import contract, program

from merlin.compile.scheduling.staged_dense import (
    AccumulatorReadout,
    AccumulatorZero,
    CompletionFence,
    ComputeConsume,
    DescriptorIssue,
    OperandLoad,
    StagedDenseProblem,
    StagedDenseRewrite,
    eliminate_redundant_rhs_loads,
    order_output_tiles_n_before_m,
    rewrite_staged_dense,
)
from merlin.compile.scheduling.staged_rhs_residency import place_rhs_stripe


def apply(ops, problem, selected, *functions):
    return rewrite_staged_dense(
        ops, problem=problem, contract=selected, rewrites=tuple(StagedDenseRewrite(fn.__name__, fn) for fn in functions)
    )


@pytest.mark.parametrize("dim", [3, 5])
@pytest.mark.parametrize("k", [11, 15, 16])
@pytest.mark.parametrize("n", [1, 5, 9])
def test_multiple_rhs_overwrites_are_resolved_at_actual_consume(dim, k, n):
    problem = StagedDenseProblem(dim + 1, k, n)
    selected = contract(dim=dim, operand_rows=8 * dim, primitive_bits=20)
    changed = []
    pending_load = None
    for op in program(problem, dim):
        if type(op) is OperandLoad and op.role == "rhs":
            pending_load = op
        if type(op) is ComputeConsume:
            # The descriptor may cross fences and two physical writes. Only
            # complete final current cells are valid for the consumed panel.
            wrong_k = dim if pending_load.source_row == 0 else 0
            changed.extend((replace(pending_load, source_row=wrong_k), pending_load))
        changed.append(op)
    after, report = apply(
        tuple(changed),
        problem,
        selected,
        order_output_tiles_n_before_m,
        place_rhs_stripe,
        eliminate_redundant_rhs_loads,
    )
    assert report["checks"][-1]["products"] == problem.m * k * n
    assert report["checks"][-1]["outputs"] == problem.m * n
    assert any(type(op) is OperandLoad and op.local_row > dim for op in after)


@pytest.mark.parametrize("damage", ["wrong_panel", "mixed_tail_columns", "missing_tail_reload"])
def test_late_wrong_or_mixed_tail_cells_cannot_be_admitted(damage):
    problem = StagedDenseProblem(7, 9, 9)
    before = list(program(problem))
    at = next(i for i, op in enumerate(before) if type(op) is ComputeConsume)
    if damage == "wrong_panel":
        before.insert(at, OperandLoad("rhs", 4, 0, 4, 4, 4))
    elif damage == "mixed_tail_columns":
        before.insert(at, OperandLoad("rhs", 0, 4, 4, 4, 1))
    else:
        before.insert(at, OperandLoad("rhs", 8, 0, 4, 1, 4))
    with pytest.raises(ValueError):
        apply(tuple(before), problem, contract(), place_rhs_stripe)


@pytest.mark.parametrize("dim", [3, 5])
def test_other_rhs_rows_and_lhs_aliases_remain_noop_even_if_legal(dim):
    problem = StagedDenseProblem(2 * dim + 1, 2 * dim + 1, dim + 1)
    # Use disjoint, noncanonical assigned locations. Their numerical validity
    # does not supply this pass's private canonical stripe assignment.
    original = []
    for op in program(problem, dim):
        if type(op) is OperandLoad:
            op = replace(op, local_row=(dim if op.role == "lhs" else 0))
        elif type(op) is DescriptorIssue:
            op = replace(op, rhs_row=0)
        elif type(op) is ComputeConsume:
            op = replace(op, lhs_row=dim)
        original.append(op)
    original = tuple(original)
    selected = contract(dim=dim, operand_rows=8 * dim, primitive_bits=20)
    final, _ = apply(original, problem, selected, place_rhs_stripe)
    assert final is original


def test_real_stationary_weight_lifetime_cannot_enter_stripe_path():
    problem = StagedDenseProblem(5, 8, 3)
    original = list(program(problem))
    # First complete output tile establishes K=4 tail panel. The next tile's
    # first K=0 requires ordinary weights, so keep it normal. Its second K=4
    # panel can explicitly retain the panel loaded by that first tile ONLY if
    # no intervening normal preload overwrites it, which is not true here.
    # This wrong retained program must fail the original independent check.
    consume = [i for i, op in enumerate(original) if type(op) is ComputeConsume][-1]
    issue = next(i for i in range(consume - 1, -1, -1) if type(original[i]) is DescriptorIssue)
    original[issue] = replace(original[issue], rhs_row=None)
    original[consume] = replace(original[consume], retain_weights=True)
    assert place_rhs_stripe(tuple(original), problem, contract()) == tuple(original)
    with pytest.raises(ValueError, match="source geometry"):
        apply(tuple(original), problem, contract(), place_rhs_stripe)


@pytest.mark.parametrize("available", [15, 16, 17])
def test_every_proposed_tail_address_stays_inside_explicit_joint_assignment(available):
    problem = StagedDenseProblem(7, 9, 9)
    original = program(problem)
    selected = contract(operand_rows=available)
    after, _ = apply(
        original, problem, selected, order_output_tiles_n_before_m, place_rhs_stripe, eliminate_redundant_rhs_loads
    )
    for op in after:
        if type(op) is OperandLoad:
            assert 0 <= op.local_row and op.local_row + op.rows <= available
    assert any(type(op) is OperandLoad and op.local_row > selected.dim for op in after) == (available >= 16)


def test_legal_retained_weights_keep_original_assignment_and_lifetime():
    problem = StagedDenseProblem(5, 8, 3)
    operations = [AccumulatorZero(0, 4, 3), AccumulatorZero(4, 1, 3)]
    for k in (0, 4):
        operations.extend(
            (
                OperandLoad("lhs", 0, k, 0, 4, 4),
                OperandLoad("rhs", k, 0, 4, 4, 3),
                DescriptorIssue(4, 0, 4, 4, 3, True),
                CompletionFence(),
                ComputeConsume(0, 4, 4),
                CompletionFence(),
                OperandLoad("lhs", 4, k, 0, 1, 4),
                DescriptorIssue(None, 4, 1, 4, 3, True),
                CompletionFence(),
                ComputeConsume(0, 1, 4, True),
                CompletionFence(),
            )
        )
    operations.extend(
        (AccumulatorReadout(0, 0, 0, 4, 3), CompletionFence(), AccumulatorReadout(4, 4, 0, 1, 3), CompletionFence())
    )
    original = tuple(operations)
    final, report = apply(original, problem, contract(), place_rhs_stripe)
    assert final is original
    assert report["checks"][-1]["retained_computes"] == 2
    assert report["checks"][-1]["products"] == 5 * 8 * 3
