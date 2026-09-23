"""Installed measurement support uses explicit inputs, never native bench defaults."""

from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import measurement_support as MS

from merlin.runtime.backends import base


def test_counter_environment_restores_after_exception(monkeypatch):
    monkeypatch.setenv("MERLIN_HW_COUNTERS", "original")
    monkeypatch.delenv("MERLIN_HW_COUNTER_UNIT", raising=False)
    with pytest.raises(RuntimeError, match="synthetic"):
        with MS.counter_environment(enabled=True, unit="BYTES"):
            assert MS.os.environ["MERLIN_HW_COUNTER_UNIT"] == "BYTES"
            raise RuntimeError("synthetic")
    assert MS.os.environ["MERLIN_HW_COUNTERS"] == "original"
    assert "MERLIN_HW_COUNTER_UNIT" not in MS.os.environ


@pytest.mark.parametrize("target", ["endpoint-one", "endpoint-two"])
def test_probe_routes_explicit_selected_provider(monkeypatch, target):
    seen = []
    artifact = {"inputs": {"circt_core_hw": {"sha256": "a" * 64}}, "counter_facts": []}
    monkeypatch.setattr(
        base, "get_backend", lambda selected: seen.append(selected) or SimpleNamespace(__name__=selected)
    )

    def imported(name):
        assert name == f"{target}.counter_byte_bindings"
        return SimpleNamespace(probe_counter_byte_bindings=lambda: artifact)

    monkeypatch.setattr(MS.importlib, "import_module", imported)
    assert MS.probe_counter_byte_bindings({"circt_core_hw": {"sha256": "a" * 64}}, target=target) == artifact
    assert seen == [target]


def test_missing_probe_is_unknown_without_fallback(monkeypatch):
    def missing(_target):
        raise LookupError("selected provider unavailable")

    monkeypatch.setattr(base, "get_backend", missing)
    result = MS.probe_counter_byte_bindings({}, target="missing-endpoint")
    assert result == {"status": "unknown", "counter_facts": [], "why": "LookupError: selected provider unavailable"}
