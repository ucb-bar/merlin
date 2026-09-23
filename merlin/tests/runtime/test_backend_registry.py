"""Shared registry contracts use synthetic registrations, never target implementations."""

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


def test_registry_taxonomy(registry):
    assert set(base.list_backends()) == set(registry)
    assert base.class_of("array") is TargetClass.NPU
    assert base.class_of("threads") is TargetClass.GPU
    assert base.class_of("cpu") is TargetClass.CPU
    assert set(base.backends_of_class(TargetClass.NPU)) == {"array"}
    assert set(base.backends_of_class(TargetClass.GPU)) == {"threads"}
    assert set(base.backends_of_class(TargetClass.CPU)) == {"cpu", "model", "route"}


def test_backend_kinds(registry):
    assert base.info("array").kind is BackendKind.KERNEL
    assert base.info("model").kind is BackendKind.WHOLE_MODEL
    assert base.info("route").kind is BackendKind.MATMUL_ROUTE
    assert {name for name in base.list_backends() if base.info(name).kind is BackendKind.MATMUL_ROUTE} == {"route"}


def test_get_backend_lazy_import(registry, monkeypatch):
    observed = []

    def load(name):
        observed.append(name)
        return registry["array"]

    monkeypatch.setattr(base.importlib, "import_module", load)
    base.list_backends()
    base.info("array")
    assert observed == []
    assert base.get_backend("array") is registry["array"]
    assert observed == [registry["array"].__name__]


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


def test_unloadable_registered_backend_is_unknown(registry):
    base.register(BackendInfo("broken", TargetClass.NPU, BackendKind.KERNEL, "_absent_fixture_backend"))
    for fact in base.execution_capability_facts("broken").values():
        assert fact["satisfied"] is None
        assert fact["missing"]
        assert "ModuleNotFoundError" in fact["evidence"]


def test_parse_console_shared_protocol():
    outs, raw = base.parse_console("OUT Y0 2 2 1 2 3 4\nMETRIC cycles 100\nDONE\n")
    assert outs == {"Y0": [[1, 2], [3, 4]]} and raw == {"cycles": 100}
    outs, raw = base.parse_console(
        "OUT Y0 1 1 7\n%Warning: junk\nMETRIC broken\nMETRIC cycles 5\nDONE\n",
        strip_warnings=True,
        tolerant_metric=True,
    )
    assert outs == {"Y0": [[7]]} and raw == {"cycles": 5}
    with pytest.raises(ValueError):
        base.parse_console("OUT Y0 1 1 5\n", error_cls=ValueError)
    with pytest.raises(ValueError):
        base.parse_console("OUT Y0 2 2 1 2\nDONE\n", error_cls=ValueError)
    outs, _ = base.parse_console("OUT Y 1 1 1.5\nDONE\n", value_parser=float)
    assert outs == {"Y": [[1.5]]}
