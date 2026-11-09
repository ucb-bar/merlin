from __future__ import annotations

import sys
from types import ModuleType

import pytest

from merlin.runtime.backends import base
from merlin.runtime.backends.base import BackendInfo, BackendKind, TargetClass


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(base, "_REGISTRY", {})
    monkeypatch.setattr(base, "_LOAD_FAILURES", {})
    monkeypatch.setattr(base, "_ensure_discovered", lambda: None)
    modules = {}
    for name, classification, kind in (
        ("array", TargetClass.NPU, BackendKind.KERNEL),
        ("threads", TargetClass.GPU, BackendKind.KERNEL),
        ("cpu", TargetClass.CPU, BackendKind.KERNEL),
        ("model", TargetClass.CPU, BackendKind.WHOLE_MODEL),
        ("route", TargetClass.CPU, BackendKind.MATMUL_ROUTE),
    ):
        module = ModuleType(f"_synthetic_backend_{name}")
        monkeypatch.setitem(sys.modules, module.__name__, module)
        base.register(BackendInfo(name, classification, kind, module.__name__))
        modules[name] = module
    return modules


def test_execution_capabilities_preserve_true_false_and_unknown(registry):
    registry["array"].EXECUTION_CAPABILITIES = {
        name: f"synthetic evidence for {name}" for name in base.EXECUTION_CAPABILITIES
    }
    declared = base.execution_capability_facts("array")
    omitted = base.execution_capability_facts("cpu")
    missing = base.execution_capability_facts("unregistered_fixture")
    for name in base.EXECUTION_CAPABILITIES:
        assert declared[name]["satisfied"] is True
        assert declared[name]["tier"] == "backend_declared"
        assert "synthetic evidence" in declared[name]["evidence"]
        assert omitted[name]["satisfied"] is False
        assert omitted[name]["tier"] == "backend_declared"
        assert missing[name]["satisfied"] is None
        assert missing[name]["tier"] == "not_established"
        assert missing[name]["missing"]
