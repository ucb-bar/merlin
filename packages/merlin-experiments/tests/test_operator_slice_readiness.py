"""Readiness follows the native operator route, not the capsule kind alone."""
from types import SimpleNamespace

import pytest

from merlin.targetgen import capsule_runner as CR
from merlin.targetgen import capsule_scheduling as CS
from merlin_experiments.phase1 import joint_control
from merlin_experiments.phase1.feedback import compiler_profile_runtime


def _readiness(monkeypatch, capsules, *, adapters=None):
    monkeypatch.setattr(joint_control, "selected", lambda: {"version": 1})
    owner = SimpleNamespace(source_sha256="a" * 64, owner=SimpleNamespace(
        snapshot_digest="b" * 64, profile={"sha256": "c" * 64}))
    monkeypatch.setattr(compiler_profile_runtime, "active_original_execution", lambda: owner)
    return CR._candidate_boundary_readiness(
        capsules, SimpleNamespace(manifest={"language": "python"}),
        oracle_adapters={"L2": object()} if adapters is None else adapters)


def _cap(name, kind="model_slice", **extra):
    return {"name": name, "kind": kind, "label": "public", **extra}


def test_operator_slices_share_bounded_ordering_without_cross_case_credit(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(CS.time, "monotonic", lambda: now[0])
    isa, component = _cap("movement", "isa"), _cap("contraction")
    readiness = _readiness(monkeypatch, [isa, component])
    assert readiness.route(isa) == readiness.route(component) is not None
    readiness.started(isa)
    assert not readiness.allows(component)
    now[0] += 30.0
    assert readiness.allows(component)  # Slow probes cannot indefinitely block fanout.
    result = readiness.completed("movement", {"status": "fail"})
    assert result["status"] == "fail"
    assert result["boundary_readiness"]["cross_case_pass_credit"] is False
    assert result["boundary_readiness"]["route_global_failure_proven"] is False


def test_connected_full_host_refusal_and_nonpublic_routes_are_excluded(monkeypatch):
    capsules = [
        _cap("component", operation={"op": "matmul"}, semantic={"must_accelerate": True}),
        _cap("connected", operation={"op": "model"}, semantic={"must_accelerate": True}),
        _cap("seam", operation={"op": "host_island_seam"}, semantic={"must_accelerate": True}),
        _cap("full", "model"),
        _cap("host", software_screen={"status": "admitted", "decisions": [
            {"placement": "host", "status": "admitted"}]}),
        _cap("refuse", expected={"outcome": "refuse"}),
        _cap("hidden", label="hidden"),
    ]
    readiness = _readiness(monkeypatch, capsules)
    assert set(readiness.routes) == {"component"}
    assert all(readiness.allows(cap) for cap in capsules[1:])


@pytest.mark.parametrize("adapters", [{}, {"L3": object()}, {"L2": object(), "L3": object()}])
def test_other_execution_scopes_keep_legacy_ordering(monkeypatch, adapters):
    assert _readiness(monkeypatch, [_cap("component")], adapters=adapters) is None
