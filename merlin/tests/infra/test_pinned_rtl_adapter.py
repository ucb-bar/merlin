"""The cert adapter reopens the selected pin around the actual oracle call."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from merlin.runtime.backends import base
from merlin.targetgen import capsule_runner, rtl_simulator_execution


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(base, "get_backend", lambda _: SimpleNamespace(available=lambda _: True))
    calls = []

    @contextmanager
    def slot(**kwargs):
        calls.append("slot")
        yield 90

    monkeypatch.setattr(rtl_simulator_execution, "verilator_slot", slot)

    def execute(*args, **kwargs):
        calls.append("execute")
        assert kwargs["timeout"] <= 90
        return {"oracle": {"kind": "rtl_verilator"}}

    monkeypatch.setattr(capsule_runner.oot_compile, "run_on_oracle", execute)
    return capsule_runner.simulator_adapter("verilator", "example"), calls


def test_verifies_on_both_sides_and_retains_provenance(adapter, monkeypatch):
    run, calls = adapter

    def verify(*args):
        calls.append("verify")
        return {"all_pins_ok": True, "execution_identity": {"binary": "same"}}

    monkeypatch.setattr(rtl_simulator_execution, "verify_selected", verify)
    result = run({}, "", None, 100)
    assert calls == ["slot", "verify", "execute", "verify"]
    assert result["oracle"]["provenance"] == {"all_pins_ok": True, "execution_identity": {"binary": "same"}}


def test_changed_hardware_cannot_return_a_verdict(adapter, monkeypatch):
    run, _ = adapter
    observations = iter([{"execution_identity": {"binary": "before"}}, {"execution_identity": {"binary": "after"}}])
    monkeypatch.setattr(rtl_simulator_execution, "verify_selected", lambda *args: next(observations))
    with pytest.raises(capsule_runner.OracleUnavailable, match="changed during execution"):
        run({}, "", None, 100)


def test_repository_bookkeeping_does_not_change_hardware_identity(adapter, monkeypatch):
    run, _ = adapter
    observations = iter(
        [
            {"execution_identity": {"binary": "same"}, "merlin": {"dirty_files": 1}},
            {"execution_identity": {"binary": "same"}, "merlin": {"dirty_files": 2}},
        ]
    )
    monkeypatch.setattr(rtl_simulator_execution, "verify_selected", lambda *args: next(observations))
    assert run({}, "", None, 100)["oracle"]["provenance"]["execution_identity"] == {"binary": "same"}
