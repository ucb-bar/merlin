from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback.private_group_provenance import join_routed_source_groups

from merlin.targetgen.owned_host_model_execution import _independently_eligible_group_ids


def test_roster_uses_eligible_members_not_planner_or_root_placement():
    groups = [SimpleNamespace(index=i) for i in (1, 2, 3)]
    placement = {"eligible_source_regions": [
        {"source_group": 1, "source_op_index": 7},
        {"source_group": 1, "source_op_index": 8},
        {"source_group": 3, "source_op_index": 75},
    ]}
    assert _independently_eligible_group_ids(placement, groups) == {1, 3}
    source = {"eligible_groups": [1, 3], "eligible_group_provenance": [
        {"source_group": i, "source_region_id": f"r{i}", "source_node_ids": [f"n{i}"]}
        for i in (1, 3)
    ]}
    assigned = [{"group": i, "source_region_id": f"r{i}", "source_node_ids": [f"n{i}"]}
                for i in (1, 3)]
    assert join_routed_source_groups(assigned, source) == {1: 1, 3: 3}
    with pytest.raises(ValueError, match="omit independently eligible"):
        join_routed_source_groups(assigned[:1], source)
    assigned[1]["source_node_ids"] = ["other"]
    with pytest.raises(ValueError):
        join_routed_source_groups(assigned, source)


def test_roster_rejects_missing_original_group_and_boolean_identity():
    groups = [SimpleNamespace(index=1)]
    for index in (2, True):
        with pytest.raises(ValueError):
            _independently_eligible_group_ids({"eligible_source_regions": [{"source_group": index}]}, groups)
