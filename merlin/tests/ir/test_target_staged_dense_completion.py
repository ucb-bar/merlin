"""The LLVM bridge preserves synchronization even when math alone is valid."""

from dataclasses import replace

import pytest
from test_target_staged_dense import Owner, contract, module, operations

from merlin.compile.scheduling import staged_dense as S
from merlin.llvmlower import target_staged_dense as T
from merlin.llvmlower import target_staged_dense_check as C


def apply(rewrite, owner=None):
    owner = owner or Owner()
    return T.rewrite_target_staged_dense(
        module(),
        problem=S.StagedDenseProblem(2, 1, 1),
        contract=contract(),
        adapter=T.StagedDenseTargetAdapter(owner.project, owner.emit, complete_effects=True),
        rewrites=(S.StagedDenseRewrite("proposal", rewrite),),
    )


def damaged(ops, mode):
    result = list(ops)
    fence = next(i for i, op in enumerate(result) if type(op) is S.CompletionFence)
    if mode == "drop_wait":
        del result[fence]
    elif mode.startswith("move_wait_"):
        wait = result.pop(fence)
        destination = {"move_wait_lhs": 1, "move_wait_rhs": 2, "move_wait_compute": 5}[mode]
        result.insert(destination, wait)
    elif mode == "add_wait":
        result.insert(1, S.CompletionFence())
    elif mode == "add_rhs":
        rhs = next(op for op in result if type(op) is S.OperandLoad and op.role == "rhs")
        result.insert(2, rhs)
    return tuple(result)


@pytest.mark.parametrize(
    "mode", ["drop_wait", "move_wait_lhs", "move_wait_rhs", "move_wait_compute", "add_wait", "add_rhs"]
)
def test_authored_pass_cannot_change_waits_or_add_effects_before_emitter(mode):
    class Counting(Owner):
        emissions = 0

        def emit(self, source, before, after):
            self.emissions += 1
            return super().emit(source, before, after)

    def rewrite(ops, _problem, _contract):
        return damaged(ops, mode)

    owner = Counting()
    with pytest.raises(ValueError, match="staged LLVM"):
        apply(rewrite, owner)
    assert owner.emissions == 0


@pytest.mark.parametrize(
    "mode", ["drop_wait", "move_wait_lhs", "move_wait_rhs", "move_wait_compute", "add_wait", "add_rhs"]
)
def test_actual_emission_cannot_retag_bad_selected_wait_sequence(mode):
    ops = damaged(operations(), mode)
    owner = Owner()
    with pytest.raises(ValueError, match="staged LLVM"):
        T.verify_target_staged_dense_emission(
            module(),
            module(ops),
            problem=S.StagedDenseProblem(2, 1, 1),
            contract=contract(),
            adapter=T.StagedDenseTargetAdapter(owner.project, owner.emit, complete_effects=True),
            selected_operations=ops,
        )


def fence_case():
    ops = operations()
    first, last = ops[:9], ops[9:]
    prefix = (S.CompletionFence(),) * 2
    suffix = (S.CompletionFence(),) * 2
    first = (*first, S.CompletionFence(), S.CompletionFence())
    return (*prefix, *first, *last, *suffix), (*prefix, *last, *first, *suffix)


def test_moved_long_closing_run_is_not_misclassified_as_new_global_suffix():
    source, actual = fence_case()
    report = C.verify_staged_completion_preservation(source, actual)
    assert report == {
        "output_tiles": 2,
        "removed_rhs_loads": 0,
        "anchored_prefix_fences": 2,
        "anchored_suffix_fences": 2,
        "scope": "preserved source synchronization placement; original completion law remains required",
    }
    assert sum(type(op) is S.CompletionFence for op in source) == sum(type(op) is S.CompletionFence for op in actual)


@pytest.mark.parametrize("boundary", ["prefix", "suffix", "tile_closing"])
def test_anchored_or_tile_completion_cannot_be_dropped(boundary):
    source, actual = fence_case()
    actual = list(actual)
    index = {"prefix": 0, "suffix": len(actual) - 1, "tile_closing": len(actual) - 3}[boundary]
    assert type(actual[index]) is S.CompletionFence
    del actual[index]
    with pytest.raises(ValueError):
        C.verify_staged_completion_preservation(source, tuple(actual))


def test_only_local_rows_are_ignored_and_math_still_checks_placement():
    source = operations()

    def assigned(op):
        if type(op) in (S.OperandLoad, S.AccumulatorZero):
            return replace(op, local_row=op.local_row + 3)
        if type(op) is S.DescriptorIssue:
            return replace(op, rhs_row=op.rhs_row + 3, accumulator_row=op.accumulator_row + 3)
        if type(op) is S.ComputeConsume:
            return replace(op, lhs_row=op.lhs_row + 3)
        if type(op) is S.AccumulatorReadout:
            return replace(op, accumulator_row=op.accumulator_row + 3)
        return op

    actual = tuple(assigned(op) for op in source)
    assert C.verify_staged_completion_preservation(source, actual)["output_tiles"] == 2
    # This proof checks ordering only; existing independent geometry still refuses.
    owner = Owner()
    with pytest.raises(ValueError, match="rectangle"):
        T.verify_target_staged_dense_emission(
            module(),
            module(actual),
            problem=S.StagedDenseProblem(2, 1, 1),
            contract=contract(),
            adapter=T.StagedDenseTargetAdapter(owner.project, owner.emit, complete_effects=True),
            selected_operations=actual,
        )


@pytest.mark.parametrize("damage", ["external", "continuation", "retain", "unknown"])
def test_logical_work_flags_and_unknown_effects_remain_exact(damage):
    source = operations()
    actual = list(source)
    if damage == "external":
        actual[1] = replace(actual[1], source_row=1)
    if damage == "continuation":
        actual[3] = replace(actual[3], continuation=False)
    if damage == "retain":
        actual[5] = replace(actual[5], retain_weights=True)
    if damage == "unknown":
        actual[5] = object()
    with pytest.raises(ValueError):
        C.verify_staged_completion_preservation(source, tuple(actual))


def test_original_checker_binding_frozen_before_first_projection(monkeypatch):
    class Mutator(Owner):
        def project(self, actual):
            monkeypatch.setattr(C, "_normalized", lambda op: ())
            return super().project(actual)

    with pytest.raises(ValueError, match="binding"):
        apply(S.eliminate_redundant_rhs_loads, Mutator())


def test_real_legal_rhs_removal_retains_every_other_operation_and_wait():
    result = apply(S.eliminate_redundant_rhs_loads)
    proof = result.report["emission"]["completion_preservation"]
    assert proof["removed_rhs_loads"] == 1 and proof["output_tiles"] == 2
    result.revalidate()


def test_whole_tile_permutation_with_long_last_run_survives_actual_llvm_reprojection():
    source, selected = fence_case()

    def reorder(ops, _problem, _contract):
        assert ops == source
        return selected

    owner = Owner()
    result = T.rewrite_target_staged_dense(
        module(source),
        problem=S.StagedDenseProblem(2, 1, 1),
        contract=contract(),
        adapter=T.StagedDenseTargetAdapter(owner.project, owner.emit, complete_effects=True),
        rewrites=(S.StagedDenseRewrite("whole_tiles", reorder),),
    )
    assert result.selected_operations == selected
    assert result.report["emission"]["completion_preservation"]["anchored_suffix_fences"] == 2
    result.revalidate()
