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


@pytest.mark.parametrize(("engine_key", "selected"), [
    ("MERLIN_SPIKE_MAX_SLOTS", None), ("MERLIN_SPIKE_MAX_SLOTS", "15"),
    ("MERLIN_GSIM_STARTUP_PRELOAD", None), ("MERLIN_GSIM_STARTUP_PRELOAD", "1"),
])
def test_phase1_plan_freezes_only_explicit_engine_controls(tmp_path, monkeypatch, engine_key, selected):
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    if selected is None:
        monkeypatch.delenv(engine_key, raising=False)
    else:
        monkeypatch.setenv(engine_key, selected)
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
    if selected is None:
        assert engine_key not in frozen
    else:
        assert frozen[engine_key] == selected
        monkeypatch.setenv(engine_key, "0")
        assert frozen[engine_key] == selected
