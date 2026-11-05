"""Full source products, assigned capacity and pass-order controls for RHS stripes."""

from dataclasses import replace

import pytest
from test_staged_dense import contract, program

from merlin.compile.scheduling.staged_dense import (
    AccumulatorReadout,
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


def apply(operations, problem, selected, functions):
    return rewrite_staged_dense(
        operations,
        problem=problem,
        contract=selected,
        rewrites=tuple(StagedDenseRewrite(function.__name__, function) for function in functions),
    )


@pytest.mark.parametrize("dim", [3, 4])
@pytest.mark.parametrize("m_delta", [0, 1, 3])
@pytest.mark.parametrize("k_panels,k_tail", [(1, 0), (1, 1), (2, 0), (2, 1)])
@pytest.mark.parametrize("n_delta", [-1, 0, 1])
def test_shape_spectrum_preserves_complete_products_and_every_fence(dim, m_delta, k_panels, k_tail, n_delta):
    problem = StagedDenseProblem(dim + m_delta, dim * k_panels + k_tail, dim + n_delta)
    selected = contract(dim=dim, operand_rows=8 * dim)
    before = program(problem, dim)
    after, report = apply(
        before, problem, selected, (order_output_tiles_n_before_m, place_rhs_stripe, eliminate_redundant_rhs_loads)
    )
    assert report["checks"][-1]["products"] == problem.m * problem.k * problem.n
    assert report["checks"][-1]["outputs"] == problem.m * problem.n
    for kind in (CompletionFence, ComputeConsume, DescriptorIssue, AccumulatorReadout):
        assert sum(type(op) is kind for op in before) == sum(type(op) is kind for op in after)
    mt = (problem.m + dim - 1) // dim
    nt = (problem.n + dim - 1) // dim
    kt = (problem.k + dim - 1) // dim
    assert len(before) - len(after) == (mt - 1) * nt * kt
    if problem.k > dim:
        assert any(type(op) is OperandLoad and op.role == "rhs" and op.local_row > dim for op in after)


@pytest.mark.parametrize("available", [15, 16, 17])
def test_exact_joint_assignment_boundary_does_not_infer_free_hardware(available):
    problem = StagedDenseProblem(7, 9, 5)
    before = program(problem)
    selected = contract(operand_rows=available)
    placed = place_rhs_stripe(before, problem, selected)
    assert (placed is before) == (available < 16)
    _, report = apply(
        before, problem, selected, (order_output_tiles_n_before_m, place_rhs_stripe, eliminate_redundant_rhs_loads)
    )
    assert report["before_commands"] - report["after_commands"] == (0 if available < 16 else 6)


@pytest.mark.parametrize("premise", ["external_buffers_disjoint", "rhs_immutable"])
def test_missing_original_ownership_keeps_original_placement(premise):
    problem = StagedDenseProblem(7, 9, 5)
    before = program(problem)
    assert place_rhs_stripe(before, problem, contract(**{premise: False})) is before


def test_descriptor_uses_later_loaded_panel_at_consume():
    problem = StagedDenseProblem(7, 9, 5)
    before = list(program(problem))
    # Move every B load across descriptor issue and its preservation fence.
    for index in reversed([i for i, op in enumerate(before) if type(op) is OperandLoad and op.role == "rhs"]):
        load = before.pop(index)
        before.insert(index + 2, load)
    after, report = apply(
        tuple(before),
        problem,
        contract(),
        (order_output_tiles_n_before_m, place_rhs_stripe, eliminate_redundant_rhs_loads),
    )
    assert report["checks"][-1]["products"] == 7 * 9 * 5
    assert len(before) - len(after) == 6


def test_pass_order_changes_reuse_without_changing_required_work():
    problem = StagedDenseProblem(7, 9, 5)
    before = program(problem)
    good, good_report = apply(
        before, problem, contract(), (order_output_tiles_n_before_m, place_rhs_stripe, eliminate_redundant_rhs_loads)
    )
    early, early_report = apply(
        before, problem, contract(), (place_rhs_stripe, eliminate_redundant_rhs_loads, order_output_tiles_n_before_m)
    )
    assert len(before) - len(good) == 6
    assert len(early) == len(before)
    assert good_report["checks"][-1]["products"] == early_report["checks"][-1]["products"]


def test_other_valid_placement_requires_its_own_assignment_rule():
    problem = StagedDenseProblem(7, 9, 5)
    before = []
    for operation in program(problem):
        if type(operation) is OperandLoad:
            operation = replace(operation, local_row=operation.local_row + 12)
        elif type(operation) is DescriptorIssue:
            operation = replace(operation, rhs_row=operation.rhs_row + 12)
        elif type(operation) is ComputeConsume:
            operation = replace(operation, lhs_row=operation.lhs_row + 12)
        before.append(operation)
    before = tuple(before)
    assert place_rhs_stripe(before, problem, contract()) is before
    after, _ = apply(before, problem, contract(), (place_rhs_stripe,))
    assert after is before


def test_changed_product_or_unknown_effect_cannot_hide_behind_proposal():
    problem = StagedDenseProblem(7, 9, 5)
    before = list(program(problem))
    index = next(i for i, op in enumerate(before) if type(op) is OperandLoad and op.role == "rhs")
    before[index] = replace(before[index], source_row=1)
    with pytest.raises(ValueError, match="source geometry"):
        apply(tuple(before), problem, contract(), (place_rhs_stripe,))
    with pytest.raises(ValueError, match="unknown staged"):
        place_rhs_stripe((*program(problem), object()), problem, contract())
