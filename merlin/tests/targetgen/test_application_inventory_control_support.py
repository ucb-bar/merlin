"""Assertions remain counted support obligations, not independent compute sites."""

import pytest

from merlin.targetgen.application_inventory import application_demand_inventory
from merlin.targetgen.operation_accounting import build_operation_accounting


@pytest.mark.parametrize("source, count", [("aten.index_put_.default", 1), (None, 1), ("shape_guard", 3)])
def test_assert_guards_remain_unverified_support(tmp_path, source, count):
    capture = tmp_path / "model.mlir"
    attrs = f'{{prov.aten = "{source}"}}' if source is not None else ""
    guard = f'"cf.assert"(%condition) <{{msg = "index is in bounds"}}> {attrs} : (i1) -> ()'
    capture.write_text(
        "module { func.func @forward(%condition: i1) {\n"
        + "\n".join([guard] * count)
        + "\nfunc.return } }"
    )
    raw = capture.read_bytes()
    inventory = application_demand_inventory(
        {"probe": capture}, "test_device", detailed=True, capability_contract={"name": "test_device"}
    )
    app = inventory["applications"]["probe"]
    guards = [row for row in app["signatures"] if row["mlir_operation"] == "cf.assert"]
    assert sum(row["count"] for row in guards) == count
    assert all(row["disposition"] == "support_required" for row in guards)
    assert app["status"] == "inventoried"
    assert sum(row["count"] for row in app["signatures"]) == app["n_operations"] == count + 3
    accounting = build_operation_accounting(inventory, capability_contract={"name": "test_device"})
    observed = [
        row for row in accounting["applications"]["probe"]["signatures"]
        if row["observed_signature"]["mlir_operation"] == "cf.assert"
    ]
    assert all(row["classification"] == "support_lowering_required" for row in observed)
    assert all(row["hardware_admission"]["status"] == "not_applicable" for row in observed)
    assert all(row["lowering_status"] == "unverified" for row in observed)
    assert capture.read_bytes() == raw


def test_an_unknown_control_operation_is_not_licensed(tmp_path):
    capture = tmp_path / "model.mlir"
    capture.write_text(
        'module { func.func @forward(%condition: i1) { '
        '"cf.unqualified_guard"(%condition) : (i1) -> () func.return } }'
    )
    inventory = application_demand_inventory(
        {"probe": capture}, "test_device", detailed=True, capability_contract={"name": "test_device"}
    )
    app = inventory["applications"]["probe"]
    assert app["status"] == "incomplete"
    assert app["counts"]["unclassified"] == 1
