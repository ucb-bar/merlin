"""Source-positive lane selection never invents execution or changes capsule claims."""

import copy

import pytest

from merlin.targetgen import native_model_execution as native
from merlin.targetgen.owned_host_model_selection import owned_model_required


def test_declared_lane_uses_fresh_source_denominator_without_changing_flag(monkeypatch):
    capsule = {"semantic": {"must_accelerate": False}, "lanes": {"require": ["on_mesh"]}}
    original, calls = copy.deepcopy(capsule), []

    def source(c, *, target):
        calls.append((c, target))
        return {"n_eligible": 2}

    monkeypatch.setattr(native, "independent_frozen_source_eligibility", source)
    assert owned_model_required(capsule, target="fixture")
    assert calls == [(capsule, "fixture")]
    assert capsule == original


@pytest.mark.parametrize("eligible", [0, -1, True, "2", None])
def test_ineligible_or_malformed_denominator_keeps_existing_route(monkeypatch, eligible):
    monkeypatch.setattr(native, "independent_frozen_source_eligibility", lambda *a, **k: {"n_eligible": eligible})
    assert not owned_model_required({"lanes": {"require": ["on_mesh"]}}, target="fixture")


def test_no_lane_obligation_never_requests_source_census(monkeypatch):
    monkeypatch.setattr(native, "independent_frozen_source_eligibility", lambda *a, **k: pytest.fail("unexpected census"))
    assert not owned_model_required({"lanes": {"require": ["host"]}}, target="fixture")
    assert owned_model_required({"semantic": {"must_accelerate": True}}, target="fixture")
