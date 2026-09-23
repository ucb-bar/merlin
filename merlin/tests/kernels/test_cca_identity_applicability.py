"""Identity classification is architecture-level metadata, never a compiler lever."""

from types import SimpleNamespace

import pytest

from merlin.kernels import cca_contract as cc


@pytest.mark.parametrize("backend", ["synthetic_compute", "synthetic_spatial"])
def test_compute_identity_is_not_a_lever_even_if_a_backend_routes_it(monkeypatch, backend):
    spec = cc.FIELD_REGISTRY["compute.op"]
    assert spec.classification == cc.IDENTITY
    assert spec.backends == ("compute",)

    # Bypass discovery entirely: no provider, target facts, simulator or code-generation hooks.
    routes = [
        SimpleNamespace(axis="compute.op", action_class="PASS"),
        SimpleNamespace(axis="compute.accumulator_resident", action_class="PASS"),
    ]
    monkeypatch.setattr(cc, "_routes", lambda selected: routes)
    assert cc.leverable_axes(backend) == {"compute.accumulator_resident"}
    report = cc.check_bijection(backend)
    assert report.orphan_routes == ["compute.op"]
    assert report.orphan_fields == []
    assert report.unclassified == []


def test_missing_route_is_not_invented_for_an_arbitrary_compute_target(monkeypatch):
    monkeypatch.setattr(cc, "_routes", lambda selected: [])
    assert cc.leverable_axes("synthetic_compute") == set()
    assert cc.check_bijection("synthetic_compute").clean
