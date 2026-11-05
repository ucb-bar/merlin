"""Installed measured calibration uses exact recorded work, without native imports."""

import builtins
import subprocess

import pytest


@pytest.fixture(autouse=True)
def no_native(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name in {"perf_model", "perf_agent_stage", "perf_capsule_verdict"}:
            pytest.fail("installed calibration imported native owner")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("native launch"))


def test_capsule_verdict_import_is_installed_and_unknown_stays_refused():
    from merlin_experiments.phase2 import capsule_verdict as V

    assert V.ceiling_dispersion([]) is None
    assert (
        V.capsule_verdict(
            capsule="fixture",
            declared_macs=None,
            achievable_rate=None,
            baseline_cycles=None,
            candidate_cycles=None,
            dispersion=None,
        )["verdict"]
        == V.REFUSED
    )
