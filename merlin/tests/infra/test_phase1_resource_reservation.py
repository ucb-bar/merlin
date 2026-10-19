"""Reservation must restore both host slices when startup partially fails."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def reservation():
    root = Path(__file__).resolve()
    if (root.parent / "resource_reservation.py").exists():
        source = root.parent / "resource_reservation.py"
    else:
        source = next(
            parent / "examples/gemmini/phase1/launch/resource_reservation.py"
            for parent in root.parents
            if (parent / "examples/gemmini/phase1/launch/resource_reservation.py").exists()
        )
    spec = importlib.util.spec_from_file_location("cpu_reservation_test_module", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def state_file(tmp_path, *, acquired=False):
    path = tmp_path / "reservation.json"
    record = {
        "schema": "merlin.operator.cpu_reservation.v1",
        "unit": "merlinphase1test",
        "status": "prepared",
        "assigned_cpus": [2, 3],
        "online_cpus": [0, 1, 2, 3],
        "host_original_allowed_cpus": {"system.slice": "", "user.slice": "0-3"},
        "host_original_effective_cpus": {"system.slice": [0, 1, 2, 3], "user.slice": [0, 1, 2, 3]},
    }
    if acquired:
        record["acquire_started_at"] = "test"
    path.write_text(json.dumps(record))
    return path


def test_partial_acquisition_failure_restores_both_original_masks(
    reservation, monkeypatch, tmp_path
):
    path = state_file(tmp_path)
    masks = {"system.slice": "", "user.slice": "0-3"}
    writes = []
    monkeypatch.setattr(reservation.os, "geteuid", lambda: 0)
    monkeypatch.setattr(reservation, "allowed", lambda unit: masks[unit])
    monkeypatch.setattr(
        reservation,
        "effective_cpus",
        lambda unit: reservation.cpus(masks[unit]) if masks[unit] else set(range(4)),
    )

    def set_allowed(unit, value):
        writes.append((unit, value))
        if unit == "user.slice" and value == "0,1":
            raise RuntimeError("injected second-slice failure")
        masks[unit] = value

    monkeypatch.setattr(reservation, "set_allowed", set_allowed)
    with pytest.raises(RuntimeError, match="injected"):
        reservation.acquire(path)
    assert masks == {"system.slice": "", "user.slice": "0-3"}
    assert writes[-3:] == [("system.slice", "0,1,2,3"), ("system.slice", ""), ("user.slice", "0-3")]
    assert json.loads(path.read_text())["release_complete"] is True


def test_release_attempts_second_slice_even_when_first_restore_fails(
    reservation, monkeypatch, tmp_path
):
    path = state_file(tmp_path, acquired=True)
    writes = []
    monkeypatch.setattr(reservation.os, "geteuid", lambda: 0)
    monkeypatch.setattr(reservation, "allowed", lambda unit: "0-3")
    monkeypatch.setattr(reservation, "effective_cpus", lambda unit: set(range(4)))

    def set_allowed(unit, value):
        writes.append((unit, value))
        if unit == "system.slice":
            raise RuntimeError("injected restoration failure")

    monkeypatch.setattr(reservation, "set_allowed", set_allowed)
    with pytest.raises(RuntimeError, match="system.slice"):
        reservation.release(path)
    assert writes == [("system.slice", "0,1,2,3"), ("user.slice", "0-3")]
    record = json.loads(path.read_text())
    assert record["release_complete"] is False
    assert record["release_errors"] == ["system.slice: injected restoration failure"]


def test_unstarted_service_does_not_overwrite_unrelated_mask_change(
    reservation, monkeypatch, tmp_path
):
    path = state_file(tmp_path)
    monkeypatch.setattr(reservation.os, "geteuid", lambda: 0)
    monkeypatch.setattr(reservation, "set_allowed", lambda *_: pytest.fail("no lease acquired"))
    reservation.release(path)
    assert json.loads(path.read_text())["status"] == "released_without_acquisition"


def test_child_requires_actual_reserved_affinity_before_execution(
    reservation, monkeypatch, tmp_path
):
    path = state_file(tmp_path)
    monkeypatch.setattr(reservation.os, "sched_getaffinity", lambda _: {0, 1})
    monkeypatch.setattr(reservation.os, "execvpe", lambda *_: pytest.fail("unreserved execution"))
    with pytest.raises(ValueError, match="exact reserved CPUs"):
        reservation.child(path)


def test_empty_systemd_property_alone_does_not_count_as_kernel_release(
    reservation, monkeypatch, tmp_path
):
    path = state_file(tmp_path, acquired=True)
    masks = {"system.slice": "0,1", "user.slice": "0,1"}
    kernel = {unit: {0, 1} for unit in masks}
    monkeypatch.setattr(reservation.os, "geteuid", lambda: 0)
    monkeypatch.setattr(reservation, "allowed", lambda unit: masks[unit])
    monkeypatch.setattr(reservation, "effective_cpus", lambda unit: kernel[unit])

    def set_allowed(unit, value):
        masks[unit] = value
        if value:
            kernel[unit] = reservation.cpus(value)

    monkeypatch.setattr(reservation, "set_allowed", set_allowed)
    reservation.release(path)
    assert masks == {"system.slice": "", "user.slice": "0-3"}
    assert all(value == set(range(4)) for value in kernel.values())
    assert json.loads(path.read_text())["release_complete"] is True
