"""Bound runtime selection must not consult a legacy engine environment."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.targetgen import gsim_emulator as GE


def test_bound_firrtl_status_and_resolution_do_not_resolve_environment(monkeypatch):
    monkeypatch.setattr(GE, "emulator_path", lambda *a, **k: pytest.fail("legacy selector reached"))
    monkeypatch.delenv("MERLIN_RTL_FACTS", raising=False)
    model = GE.Resolution(
        "synthetic", Path("/selected/binary"), "contract", True, "selected", receipt_status="bound", digest="a" * 64
    )
    backend = SimpleNamespace(
        gsim_resolution=lambda: model, gsim_selected_firrtl_status=lambda: (True, "bound original FIRRTL")
    )
    assert GE.selected_firrtl_status("synthetic", backend=backend) == (True, "bound original FIRRTL")
    assert GE.resolve("synthetic", backend=backend) is model
    backend.gsim_selected_firrtl_status = lambda: (False, "changed")
    assert GE.selected_firrtl_status("synthetic", backend=backend) == (False, "changed")


def test_malformed_bound_callbacks_cannot_fall_back(monkeypatch):
    monkeypatch.setattr(GE, "emulator_path", lambda *a, **k: pytest.fail("legacy selector reached"))
    for value in (False, lambda: (1, "changed"), lambda: [True, "selected"]):
        assert (
            GE.selected_firrtl_status("synthetic", backend=SimpleNamespace(gsim_selected_firrtl_status=value))[0]
            is False
        )
    for value in (False, lambda: {}, lambda: GE.Resolution("other", Path("/selected"), "contract", True, "selected")):
        with pytest.raises(ValueError):
            GE.resolve("synthetic", backend=SimpleNamespace(gsim_resolution=value))


def test_chipyard_policy_consumes_bound_firrtl_status(monkeypatch):
    from merlin.runtime.backends import base
    from merlin.targetgen import oracle_policy, rtl_engine_policy

    backend = SimpleNamespace(
        gsim_status=lambda: (True, "selected binary"),
        gsim_selected_firrtl_status=lambda: (True, "bound original FIRRTL"),
    )
    monkeypatch.setattr(base, "get_backend", lambda target: backend)
    monkeypatch.setattr(GE, "emulator_path", lambda *a, **k: pytest.fail("legacy selector reached"))
    monkeypatch.setattr(rtl_engine_policy, "select", lambda target, probes: {"status": probes["gsim"]()})
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE", raising=False)
    assert oracle_policy.chipyard_l3_selection("synthetic")["status"] == (
        True,
        "selected binary; bound original FIRRTL",
    )
