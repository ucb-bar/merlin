"""Representative planning is source-bound and keeps host and tensor identities."""

from dataclasses import replace

import pytest
from test_shared_input_schedule import source

from merlin.compile.model_coverage import plan_model_coverage
from merlin.xdsl_dialects.lowering.compute_groups import Group


def fixture(*, different_lhs=False, widths=(7, 3, 7)):
    module = source(5, 11, widths, different_lhs=different_lhs, epilogue=True)
    roots = [op for op in module.walk() if op.name == "linalg.generic"]
    host = next(
        op for op in module.walk() if op.name == "arith.addi" and op.results[0].type == roots[0].results[0].type
    )
    groups = [
        Group(i, "selected_unit", root, [root], ["contraction"], macs=5 * 11 * n)
        for i, (root, n) in enumerate(zip(roots, widths))
    ]
    groups.append(Group(3, "host", None, [host], ["residual_add"]))
    return module, groups


def test_representatives_keep_tails_and_host_work_and_do_not_claim_equivalence():
    module, groups = fixture()
    plan = plan_model_coverage(module, groups)
    assert len(plan["profiles"]) == 3
    assert plan["profiles"][0]["members"] == [0, 2]
    assert plan["profiles"][1]["members"] == [1]
    assert plan["placement_counts"] == {"selected_unit": 3, "host": 1}
    assert not plan["performance_qualification"]
    assert "no numerical substitution" in plan["scope"]


def test_original_shared_input_identity_and_all_external_observers_survive():
    module, groups = fixture()
    boundaries = plan_model_coverage(module, groups)["boundaries"]
    lhs = [v for v in boundaries if v["shape"] == [5, 11]]
    assert len(lhs) == 1 and lhs[0]["consumer_groups"] == [0, 1, 2]
    assert lhs[0]["logical_bytes"] == 55
    first = next(v for v in boundaries if v["producer_group"] == 0)
    assert first["consumer_groups"] == [3]
    assert first["logical_bytes"] == 140
    assert first["completion"] == "not_established"
    assert first["alias_ownership"] == "not_established"
    assert sum(v["external_output"] for v in boundaries) == 3
    assert not any(v["shape"] == [] for v in boundaries)  # no invented scalar body inputs


def test_same_type_different_source_inputs_remain_distinct():
    module, groups = fixture(different_lhs=True)
    lhs = [v for v in plan_model_coverage(module, groups)["boundaries"] if v["shape"] == [5, 11]]
    assert sorted(v["consumer_groups"] for v in lhs) == [[0], [1, 2]]


def test_consumer_closure_groups_use_source_sink_order_without_inventing_contiguity():
    module, groups = fixture()
    combined = replace(groups[0], members=groups[0].members + groups[3].members, stages=["contraction", "residual_add"])
    plan = plan_model_coverage(module, [groups[1], groups[2], combined])
    assert [row["index"] for row in plan["groups"]] == [1, 2, 0]
    assert plan["groups"][-1]["stages"] == ["contraction", "residual_add"]


@pytest.mark.parametrize("failure", ["overlap", "foreign", "root", "order", "indices", "nested"])
def test_inconsistent_source_grouping_refuses(failure):
    module, groups = fixture()
    if failure == "overlap":
        groups[1] = replace(groups[1], root=groups[0].root, members=groups[0].members)
    elif failure == "foreign":
        other, other_groups = fixture()
        groups[1] = other_groups[1]
    elif failure == "root":
        groups[0] = replace(groups[0], root=groups[1].root)
    elif failure == "order":
        groups = list(reversed(groups))
    elif failure == "indices":
        groups[1] = replace(groups[1], index=0)
    else:
        child = next(op for op in groups[0].root.walk() if op.name == "arith.muli")
        groups[1] = replace(groups[1], root=child, members=[child])
    with pytest.raises(ValueError):
        plan_model_coverage(module, groups)


def test_source_change_changes_identity_and_plan_is_json_safe():
    import json

    module, groups = fixture()
    plan = plan_model_coverage(module, groups)
    changed, changed_groups = fixture(widths=(7, 4, 7))
    assert plan["source_ir_sha256"] != plan_model_coverage(changed, changed_groups)["source_ir_sha256"]
    assert json.loads(json.dumps(plan)) == plan
