"""Optional offload never substitutes for required work or escapes source auditing."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback import private_group_provenance as provenance
from xdsl.dialects.builtin import ArrayAttr, StringAttr, TensorType, i8


def _group(index, *, placement="device", identity=True):
    root = SimpleNamespace(
        attributes={
            "prov.region_id": StringAttr(f"region-{index}"),
            "prov.source_node_ids": ArrayAttr([StringAttr(f"node-{index}")]),
        } if identity else {},
        results=[SimpleNamespace(type=TensorType(i8, [4, 8]))],
    )
    return SimpleNamespace(index=index, placement=placement, root=root, members=[root])


def _roster():
    return provenance.source_group_roster(
        [_group(4), _group(8), _group(9), _group(12, placement="host")],
        {4}, host_placement="host",
    )


def _calls(source, *, optional=()):
    return [
        {"group": 100 + row["source_group"], **row}
        for row in source["eligible_group_provenance"] + source["optional_group_provenance"]
        if row["source_group"] in set(source["eligible_groups"]) | set(optional)
    ]


def test_optional_groups_can_be_omitted_or_routed_without_changing_mandatory_coverage():
    source = _roster()
    assert source["eligible_groups"] == [4]
    assert source["optional_groups"] == [8, 9]
    assert source["eligible_group_metrics"] == {4: {"elements": 32, "element_bytes": 1}}
    assert source["optional_group_metrics"] == {
        8: {"elements": 32, "element_bytes": 1}, 9: {"elements": 32, "element_bytes": 1},
    }
    before = deepcopy(source)
    for optional in ((), (8,), (9,), (8, 9)):
        calls = _calls(source, optional=optional)
        joined = provenance.join_routed_source_groups(calls, source)
        assert set(joined.values()) == {4, *optional}
        assert set(joined.values()) & set(source["eligible_groups"]) == {4}
    assert source == before
    with pytest.raises(ValueError, match="omit independently eligible"):
        provenance.join_routed_source_groups(_calls(source, optional=(8,))[1:], source)


def test_optional_roster_is_source_bound_even_when_not_selected():
    source = _roster()
    calls = _calls(source, optional=(8,))
    for changed in (
        [*calls, {"group": 199, "source_region_id": "invented", "source_node_ids": ["node-99"]}],
        [calls[0], {**calls[1], "source_node_ids": ["changed"]}],
        [*calls, {**calls[1], "group": 200}],
    ):
        with pytest.raises(ValueError, match="provenance|ambiguous"):
            provenance.join_routed_source_groups(changed, source)
    for mutate in (
        lambda s: s.update(optional_groups=[4, 8, 9]),
        lambda s: s.update(optional_groups=[8, 8]),
        lambda s: s.update(optional_groups=[True]),
        lambda s: s["optional_group_provenance"][0].update(source_region_id="region-4"),
        lambda s: s["optional_group_provenance"][0].update(source_group=99),
        lambda s: s["optional_group_provenance"][0].update(source_node_ids=[]),
    ):
        changed = deepcopy(source)
        mutate(changed)
        with pytest.raises(ValueError, match="ambiguous|provenance"):
            provenance.join_routed_source_groups(_calls(source), changed)


def test_unbound_optional_root_can_remain_host_but_cannot_be_emitted():
    source = provenance.source_group_roster(
        [_group(4), _group(8, identity=False)], {4}, host_placement="host"
    )
    assert source["optional_groups"] == []
    assert source["optional_group_refusals"][0]["source_group"] == 8
    calls = _calls(source)
    assert provenance.join_routed_source_groups(calls, source) == {104: 4}
    with pytest.raises(ValueError, match="independently bound source"):
        provenance.join_routed_source_groups([
            *calls, {"group": 108, "source_region_id": "region-8", "source_node_ids": ["node-8"]},
        ], source)
    with pytest.raises(ValueError, match="provenance"):
        provenance.source_group_roster([_group(4, identity=False)], {4}, host_placement="host")
    with pytest.raises(ValueError, match="assigned to host"):
        provenance.source_group_roster([_group(4, placement="host")], {4}, host_placement="host")


def test_host_selection_does_not_excuse_missing_required_work_or_other_device_refusals():
    source = _roster()
    source["accelerator_source_demand"] = {
        "status": "source_demand_derived", "n_unknown_host_groups": 0, "n_missing_direct_groups": 0,
        "n_unresolved_raw_regions": 0, "demanded_group_provenance": source["eligible_group_provenance"],
        "n_demanded_groups": 1,
    }
    emitted = {
        "routed": _calls(source, optional=(8,)),
        "skipped": [["optional-9", "the placement did not select this group's root"]],
    }
    assert provenance.verify_routed_source_coverage(emitted, source) == {104: 4, 108: 8}
    for changed in (
        {**emitted, "routed": emitted["routed"][1:]},
        {**emitted, "skipped": [["optional-9", "unsupported numeric format"]]},
    ):
        with pytest.raises(ValueError, match="omit independently eligible|outside intended host selection"):
            provenance.verify_routed_source_coverage(changed, source)
