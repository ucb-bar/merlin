"""Finite staged semantics and candidate code; no target qualification is issued."""

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
    rewrite_staged_dense,
)
from merlin.compile.scheduling.staged_dense_verification import verify_staged_dense


def contract(**changes):
    result = StagedDenseContract(
        "selected-facts",
        "selected-effects",
        "exact-signed-raw",
        "independent-fixture",
        4,
        32,
        16,
        8,
        18,
        32,
        "read_at_compute_consume",
        "data_commands_preserve_pending_descriptor",
        "raw_accumulator_identity",
        True,
        True,
        True,
    )
    return replace(result, **changes)


def program(problem, dim=4):
    result = []
    for m in range(0, problem.m, dim):
        for n in range(0, problem.n, dim):
            mr, nr = min(dim, problem.m - m), min(dim, problem.n - n)
            result.append(AccumulatorZero(0, mr, nr))
            for k in range(0, problem.k, dim):
                kr = min(dim, problem.k - k)
                result.extend(
                    (
                        OperandLoad("lhs", m, k, 0, mr, kr),
                        OperandLoad("rhs", k, n, dim, kr, nr),
                        DescriptorIssue(dim, 0, mr, kr, nr, True),
                        CompletionFence(),
                        ComputeConsume(0, mr, kr),
                        CompletionFence(),
                    )
                )
            result.extend((AccumulatorReadout(0, m, n, mr, nr), CompletionFence()))
    return tuple(result)


@pytest.mark.parametrize("m", [1, 4, 5, 7, 8, 9])
@pytest.mark.parametrize("k", [1, 4, 5, 8])
@pytest.mark.parametrize("n", [1, 3, 4, 5])
def test_full_source_spectrum_tails_and_legal_rhs_retention(m, k, n):
    problem = StagedDenseProblem(m, k, n)
    before = program(problem)
    after, report = rewrite_staged_dense(
        before,
        problem=problem,
        contract=contract(),
        rewrites=(StagedDenseRewrite("shape_policy", eliminate_redundant_rhs_loads),),
    )
    assert report["checks"][-1]["products"] == m * k * n
    assert report["checks"][-1]["outputs"] == m * n
    assert report["checks"][-1]["fences_preserving_pending_descriptor"] > 0
    removed = len(before) - len(after)
    assert removed == ((m - 1) // 4 if k <= 4 and n <= 4 else 0)


@pytest.mark.parametrize("premise", ["rhs_immutable", "external_buffers_disjoint"])
def test_missing_ownership_never_eliminates_load(premise):
    problem = StagedDenseProblem(7, 4, 3)
    original = program(problem)
    assert eliminate_redundant_rhs_loads(original, problem, contract(**{premise: False})) is original
    if premise == "external_buffers_disjoint":
        with pytest.raises(ValueError, match="ownership proof"):
            verify_staged_dense(original, problem=problem, contract=contract(**{premise: False}))


def test_descriptor_observes_cells_at_compute_not_at_issue_or_fence():
    problem = StagedDenseProblem(2, 8, 3)
    original = list(program(problem))
    consume = next(i for i, op in enumerate(original) if type(op) is ComputeConsume)
    original.insert(consume, OperandLoad("rhs", 4, 0, 4, 4, 3))
    with pytest.raises(ValueError, match="source geometry"):
        verify_staged_dense(tuple(original), problem=problem, contract=contract())


def test_fence_cannot_retire_a_lone_descriptor():
    problem = StagedDenseProblem(2, 4, 3)
    original = (*program(problem), DescriptorIssue(4, 0, 2, 4, 3, True), CompletionFence())
    with pytest.raises(ValueError, match="pending descriptors"):
        verify_staged_dense(original, problem=problem, contract=contract())


def test_retained_weights_survive_spad_clobber_only_with_explicit_ownership():
    problem = StagedDenseProblem(5, 4, 3)
    original = list(program(problem))
    issues = [i for i, op in enumerate(original) if type(op) is DescriptorIssue]
    consumes = [i for i, op in enumerate(original) if type(op) is ComputeConsume]
    original[issues[1]] = replace(original[issues[1]], rhs_row=None)
    original[consumes[1]] = replace(original[consumes[1]], retain_weights=True)
    original[issues[1] - 1] = OperandLoad("lhs", 0, 0, 4, 4, 4)
    report = verify_staged_dense(tuple(original), problem=problem, contract=contract())
    assert report["retained_computes"] == 1
    with pytest.raises(ValueError, match="retained weights"):
        verify_staged_dense(tuple(original), problem=problem, contract=contract(retained_weights=False))


@pytest.mark.parametrize("damage", ["drop_compute", "drop_init", "no_continue", "address", "readout", "unknown"])
def test_new_candidate_pass_cannot_erase_work_effects_or_ownership(damage):
    problem = StagedDenseProblem(3, 5, 3)
    original = program(problem)

    def authored(operations, _problem, _contract):
        candidate = list(operations)
        if damage == "drop_compute":
            del candidate[next(i for i, op in enumerate(candidate) if type(op) is ComputeConsume)]
        elif damage == "drop_init":
            del candidate[0]
        elif damage == "no_continue":
            issues = [i for i, op in enumerate(candidate) if type(op) is DescriptorIssue]
            candidate[issues[1]] = replace(candidate[issues[1]], continuation=False)
        elif damage == "address":
            candidate[1] = replace(candidate[1], local_row=32)
        elif damage == "readout":
            position = next(i for i, op in enumerate(candidate) if type(op) is AccumulatorReadout)
            candidate[position] = replace(candidate[position], output_col=1)
        else:
            candidate.insert(0, object())
        return tuple(candidate)

    with pytest.raises(ValueError):
        rewrite_staged_dense(
            original,
            problem=problem,
            contract=contract(),
            rewrites=(StagedDenseRewrite("new_transformation", authored),),
        )


def test_candidate_cannot_mutate_selected_resources_or_original_source():
    problem = StagedDenseProblem(3, 4, 3)

    def mutate(operations, _problem, selected):
        object.__setattr__(selected, "operand_rows", 4096)
        return operations

    with pytest.raises(ValueError, match="changed original"):
        rewrite_staged_dense(
            program(problem), problem=problem, contract=contract(), rewrites=(StagedDenseRewrite("invalid", mutate),)
        )


def test_candidate_order_changes_actual_proposal_under_same_fixed_checks():
    problem = StagedDenseProblem(7, 4, 3)
    original = program(problem)

    def add_redundant_panel(operations, _problem, _contract):
        index = next(i for i, op in enumerate(operations) if type(op) is OperandLoad and op.role == "rhs")
        return (*operations[:index], operations[index], *operations[index:])

    add = StagedDenseRewrite("new_panel_transform", add_redundant_panel)
    eliminate = StagedDenseRewrite("shape_residency", eliminate_redundant_rhs_loads)
    left, _ = rewrite_staged_dense(original, problem=problem, contract=contract(), rewrites=(add, eliminate))
    right, _ = rewrite_staged_dense(original, problem=problem, contract=contract(), rewrites=(eliminate, add))
    assert len(right) == len(left) + 1
    assert left != right


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "merlin.dense_schedule.v1"},
        {"preload_visibility": "at_issue"},
        {"fence_completion": "all_commands_retire"},
        {"readout": "scaled"},
        {"primitive_bits": 17},
    ],
)
def test_incompatible_or_unknown_staged_contract_refuses(change):
    with pytest.raises(ValueError):
        contract(**change)
