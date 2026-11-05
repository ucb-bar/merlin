"""Shared fusion selection preserves complete source coverage and dependencies."""

from dataclasses import replace

import pytest

from merlin.compile.application_plan import application_plan, select_application_plan
from merlin.compile.semantic_regions import outline_semantic_regions
from merlin.semantic_compiler.target_binding import NativeCompilationError
from merlin.tests.ir.test_semantic_regions import graph, plan_for


def alternatives(source, groups, prefix):
    return tuple(replace(row, id=prefix + row.id) for row in plan_for(source, groups).selected)


def test_first_valid_partition_rejects_cyclic_fusion_and_preserves_shared_outputs():
    source = graph()
    singles = alternatives(source, ((0,), (1,), (2,), (3,)), "single_")
    cyclic = alternatives(source, ((0, 2, 3), (1,)), "cyclic_")[0]
    fused = alternatives(source, ((0, 2), (1,), (3,)), "fused_")[0]
    choices = (cyclic, fused, *singles)
    plan = select_application_plan(source, choices)
    assert fused in plan.selected and cyclic not in plan.selected
    assert select_application_plan(source, reversed(choices)) == plan
    outlined = outline_semantic_regions(source, entry="forward", plan=plan)
    assert outlined.emission.dispatch.results == list(source.outputs)


def test_work_and_coverage_failures_do_not_call_legacy_planning():
    source = graph()
    rows = alternatives(source, ((0,), (1,), (2,), (3,)), "single_")
    with pytest.raises(NativeCompilationError, match="work limit"):
        select_application_plan(source, rows, max_states=1)

    class Provider:
        def application_region_alternatives(self, source):
            return ()

        def plan(self, source):
            raise AssertionError("must not fall back after shared selection fails")

    with pytest.raises(NativeCompilationError, match="covers every node"):
        application_plan(source, Provider())


def test_shared_planning_is_invoked_without_legacy_selection():
    source = graph()
    expected = plan_for(source, ((0, 2), (1,), (3,)))

    class Provider:
        def application_region_alternatives(self, source):
            return expected.selected

        def plan(self, source):
            raise AssertionError("enumerating providers use Merlin's selector")

    assert application_plan(source, Provider()).selected == expected.selected
