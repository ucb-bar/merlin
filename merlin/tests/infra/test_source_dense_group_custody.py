"""Source-only issuance checks using three independently verified dense groups.

The neutral declarations exercise provenance custody, not target qualification.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_compact_dense_group import fixture, selected_fixture
from test_shared_lhs_source import CASES, source

from merlin.compile.scheduling.compact_dense_group import implementation_paths
from merlin.compile.scheduling.dense_source import DenseSourceUnavailable, read_source_dense
from merlin.llvmlower.source_dense_group import (
    SourceDenseGrouping,
    SourceDenseGroupResult,
    schedule_source_dense_group,
)


@pytest.fixture(scope="module")
def checked_groups(tmp_path_factory):
    selected, model = selected_fixture(tmp_path_factory.mktemp("source-group-custody"))

    def build(case, actual):
        module = source(case)
        operations = []
        for operation in module.walk():
            try:
                read_source_dense(operation, selected_contracts=selected, max_source_elements=4096)
            except DenseSourceUnavailable:
                continue
            operations.append(operation)
        assert len(operations) == 2
        selection = SimpleNamespace(
            lhs_grouping=SourceDenseGrouping(actual[2], actual[5], actual[6], actual[4], max_members=2),
            selected_contracts=selected,
            max_source_elements=4096,
            resources=actual[1],
            static_budget=160,
            adapter=actual[3],
            external_buffers_disjoint=True,
            reservations=(),
            verification_timeout_ms=30000,
        )
        result = schedule_source_dense_group(module, tuple(operations), selection, lambda: None)
        assert result is not None
        result.revalidate()
        result.compiled.revalidate()
        return result

    original = build(CASES[0], model)
    same_shape_model = (
        tuple(replace(problem, n=5) for problem in model[0]),
        *model[1:5],
        *fixture(same_shape=True)[5:],
    )
    different_shape = build(replace(CASES[0], shape=(5, 5, 5, 5)), same_shape_model)
    shifted = tuple(replace(storage, rhs=replace(storage.rhs, base=storage.rhs.base + 2)) for storage in model[2])
    different_assignment = build(CASES[0], (*model[:2], shifted, *model[3:]))
    return original, different_shape, different_assignment


def test_actual_checked_groups_have_distinct_source_or_resource_contexts(checked_groups):
    original, shape, assignment = checked_groups
    assert original.compiled.kernel.args != shape.compiled.kernel.args
    assert original.compiled.kernel.args == assignment.compiled.kernel.args
    assert (
        original.compiled.checked_emission.context.resources != assignment.compiled.checked_emission.context.resources
    )
    for result in checked_groups:
        assert result.compiled.implementation_paths() == implementation_paths()
        assert set(result.compiled.implementation_paths()) <= {path for path, _ in result.source_pins}


@pytest.mark.parametrize("foreign_index", [1, 2], ids=["different-N", "same-shape-different-RHS-slots"])
def test_issued_source_group_rejects_genuinely_checked_foreign_compilation(checked_groups, foreign_index):
    original, *foreign = checked_groups
    other = foreign[foreign_index - 1]
    fields = {"compiled": original.compiled, "lowered_stamp": original.lowered_stamp}
    try:
        # Both foreign compilation and its matching checked stamp are valid.
        # The original source result must still retain its own exact program.
        object.__setattr__(original, "compiled", other.compiled)
        object.__setattr__(original, "lowered_stamp", other.lowered_stamp)
        with pytest.raises(ValueError, match="source-to-program binding changed"):
            original.revalidate()
    finally:
        for name, value in fields.items():
            object.__setattr__(original, name, value)
    original.revalidate()


@pytest.mark.parametrize("field", ["operation_ordinals", "source_passes", "selection_identity", "source_pins"])
def test_issued_group_retains_original_provenance(checked_groups, field):
    original = checked_groups[0]
    before = getattr(original, field)
    changed = {
        "operation_ordinals": tuple(reversed(original.operation_ordinals)),
        "source_passes": (),
        "selection_identity": "0" * 64,
        "source_pins": (),
    }[field]
    assert changed != before
    try:
        object.__setattr__(original, field, changed)
        with pytest.raises(ValueError, match="source-to-program binding changed"):
            original.revalidate()
    finally:
        object.__setattr__(original, field, before)
    original.revalidate()


def test_replacement_guard_cannot_acquire_issued_identity(checked_groups):
    original = checked_groups[0]
    before = original._guard
    calls = []
    try:
        object.__setattr__(original, "_guard", lambda candidate: calls.append(candidate))
        with pytest.raises(ValueError, match="issuance or guard changed"):
            original.revalidate()
        assert calls == []
    finally:
        object.__setattr__(original, "_guard", before)
    original.revalidate()


def test_dataclass_clone_does_not_inherit_issuance(checked_groups):
    with pytest.raises(ValueError, match="issuance or guard changed"):
        replace(checked_groups[0]).revalidate()


def test_retained_validation_rejects_replaced_result_method_before_dispatch(checked_groups, monkeypatch):
    original = checked_groups[0]
    retained = original.revalidate
    before = SourceDenseGroupResult.emit_c
    calls = []
    try:
        monkeypatch.setattr(SourceDenseGroupResult, "emit_c", lambda *args, **kwargs: calls.append(args))
        with pytest.raises(ValueError, match="implementation function or code changed"):
            retained()
        assert calls == []
    finally:
        monkeypatch.setattr(SourceDenseGroupResult, "emit_c", before)
    original.revalidate()
