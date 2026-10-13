"""Launch plans retain explicit engine-cap selections independently of later environment drift."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments import runner
from merlin_experiments.adapters import PHASE1_MODULE
from merlin_experiments.spec import ExperimentSpec

from merlin.llvmlower import toolchain
from merlin.targetgen import target_registry


@pytest.mark.parametrize("spike_cap", [None, "15"])
def test_phase1_plan_freezes_only_explicit_spike_cap(tmp_path, monkeypatch, spike_cap):
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    if spike_cap is None:
        monkeypatch.delenv("MERLIN_SPIKE_MAX_SLOTS", raising=False)
    else:
        monkeypatch.setenv("MERLIN_SPIKE_MAX_SLOTS", spike_cap)
    monkeypatch.setenv("MERLIN_GSIM_MAX_SLOTS", "15")
    monkeypatch.setattr(target_registry, "effective_target_path", lambda: "fixture-support")
    monkeypatch.setattr(toolchain, "clang_for", lambda root, env: Path("clang"))
    monkeypatch.setattr(runner, "_phase1_source_inputs", lambda command: {})
    monkeypatch.setattr(runner, "_phase1_operator_inputs", lambda command: {})
    adapter = SimpleNamespace(
        name="capsule_bench",
        resolve=lambda spec, config, root, destination: {
            "module": PHASE1_MODULE, "env": {},
            "entrypoint": str(tmp_path / "entrypoint.py"),
            "inputs": {"descriptor": str(tmp_path / "not-materialized.yaml")},
        },
    )
    monkeypatch.setitem(runner.ADAPTERS, "capsule_bench", adapter)
    spec = ExperimentSpec(tmp_path / "definition.yaml", {
        "id": "engine-cap-fixture", "target": "synthetic",
        "phases": {"1": {"adapter": "capsule_bench", "config": {}}},
    })
    plan = runner.resolve_plan(spec, phase="1", run_dir=tmp_path / "run")
    frozen = plan["phases"]["1"]["env"]
    assert frozen["MERLIN_GSIM_MAX_SLOTS"] == "15"
    if spike_cap is None:
        assert "MERLIN_SPIKE_MAX_SLOTS" not in frozen
    else:
        assert frozen["MERLIN_SPIKE_MAX_SLOTS"] == spike_cap
        monkeypatch.setenv("MERLIN_SPIKE_MAX_SLOTS", "2")
        assert frozen["MERLIN_SPIKE_MAX_SLOTS"] == "15"
