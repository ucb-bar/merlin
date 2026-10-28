"""The derivation inventory takes its operation graphs from pooled jobs without changing a byte."""

from __future__ import annotations

import pytest

from merlin.common import parse_pool as PP
from merlin.targetgen import application_inventory as A

SOURCE = "module {{ func.func @f() {{ %x = arith.constant {value} : i32 func.return }} }}"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    PP.clear_memo()
    monkeypatch.delenv(PP.WORKERS_ENV, raising=False)
    yield
    PP.clear_memo()


CONTRACT = {"name": "test_device", "compute_units": []}


def inventory(paths):
    return A.application_demand_inventory(
        paths, "test_device", detailed=True, include_graph=True, capability_contract=CONTRACT
    )


def captures(tmp_path):
    paths = {}
    for index, label in enumerate(("a", "b", "c")):
        directory = tmp_path / label
        directory.mkdir()
        (directory / "model.mlir").write_text(SOURCE.format(value=index + 2))
        paths[label] = directory / "model.mlir"
    return paths


def test_pooled_inventory_equals_serial_inventory(tmp_path, monkeypatch):
    paths = captures(tmp_path)
    monkeypatch.setenv(PP.WORKERS_ENV, "0")
    serial = inventory(paths)
    monkeypatch.setenv(PP.WORKERS_ENV, "2")
    pooled = inventory(paths)
    assert pooled == serial and all(row["operation_graph"] for row in pooled["applications"].values())


def test_a_capture_changed_after_the_start_is_graphed_from_the_bytes_read_at_its_turn(tmp_path, monkeypatch):
    paths = captures(tmp_path)
    monkeypatch.setenv(PP.WORKERS_ENV, "2")
    original = A._application_operation_inventory

    def mutate_second(path, *args, **kwargs):
        if path == paths["a"]:
            paths["b"].write_text(SOURCE.format(value=41))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(A, "_application_operation_inventory", mutate_second)
    pooled = inventory(paths)
    monkeypatch.setattr(A, "_application_operation_inventory", original)
    monkeypatch.setenv(PP.WORKERS_ENV, "0")
    assert pooled == inventory(paths)
