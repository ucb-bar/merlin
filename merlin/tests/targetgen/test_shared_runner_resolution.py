"""Core resolver ownership preserves legacy evaluation overrides without importing them."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import asdict

import pytest

from merlin.common.paths import python_import_roots
from merlin.targetgen import capsule_runner as evaluator
from merlin.targetgen import program_engine_policy as engines
from merlin.targetgen import program_oracle, runner_config, target_experiment, tier_affordability


def _no_manifest(_target):
    raise ValueError("no manifest")


def test_legacy_exports_are_exact_shared_identities():
    assert evaluator._config_for_target is runner_config._config_for_target
    assert evaluator._default_config is runner_config._default_config
    assert evaluator._TIER_SIM is runner_config.CONVENTIONAL_TIER_SIM
    assert evaluator._RTL_TIERS is runner_config.CONVENTIONAL_RTL_TIERS
    assert program_oracle._model_venv_python is engines._model_venv_python


@pytest.mark.parametrize("suite", [None, "", "explicit-suite"])
def test_missing_manifest_preserves_every_default_field(monkeypatch, suite):
    monkeypatch.setattr(target_experiment, "load_capability_manifest", _no_manifest)
    assert asdict(runner_config.selected_runner_config("fixture", suite, "f32")) == {
        "target": "fixture",
        "suite": suite or "fixture-capsule-bench",
        "dtype": "f32",
        "fourth_output_name": "lowered.llvm.mlir",
        "tier_sim": {"L2": "spike", "L3": "elaborated_rtl", "L4": "vcs", "L5": "firesim"},
        "rtl_tiers": frozenset({"L3", "L4", "L5"}),
        "oracle_tiers": ("L2", "L3", "L4", "L5"),
        "perf_fields": (),
        "trace_gate": "rocc_insn",
        "force_match_policy": None,
    }


def test_manifest_conversion_and_conversion_failure_preserve_fallback(monkeypatch):
    manifest, expected = object(), object()
    monkeypatch.setattr(target_experiment, "load_capability_manifest", lambda target: manifest)

    def convert(value):
        assert value is manifest
        return expected

    monkeypatch.setattr(runner_config, "runner_config_from_manifest", convert)
    assert runner_config.selected_runner_config("fixture", "unused", "unused") is expected
    monkeypatch.setattr(runner_config, "runner_config_from_manifest", _no_manifest)
    assert runner_config.selected_runner_config("fixture", None, "f32").dtype == "f32"


def test_legacy_config_and_fallback_replacements_remain_visible(monkeypatch):
    expected = object()
    monkeypatch.setattr(evaluator, "_config_for_target", lambda *args: expected)
    assert runner_config.selected_runner_config("fixture", None, "f32") is expected
    monkeypatch.setattr(evaluator, "_config_for_target", runner_config._config_for_target)
    monkeypatch.setattr(target_experiment, "load_capability_manifest", _no_manifest)
    monkeypatch.setattr(evaluator, "_default_config", lambda *args: expected)
    assert runner_config.selected_runner_config("fixture", None, "f32") is expected


def test_shared_config_override_not_shadowed_by_legacy_reexport(monkeypatch):
    expected = object()
    monkeypatch.setattr(runner_config, "_config_for_target", lambda *args: expected)
    assert runner_config.selected_runner_config("fixture", None, "f32") is expected


def test_legacy_map_replacement_and_mutation_preserve_consumers(monkeypatch):
    mapping = {"L2": "fixture-engine", "L3": "other-engine"}
    monkeypatch.setattr(evaluator, "_TIER_SIM", mapping)
    monkeypatch.setattr(evaluator, "_RTL_TIERS", {"L3"})
    config = runner_config._default_config("fixture", "suite", "f32")
    assert config.tier_sim == mapping and config.tier_sim is not mapping
    assert config.rtl_tiers == frozenset({"L3"})
    assert "l2" in tier_affordability._engine_vocabulary()[1]
    assert "l4" not in tier_affordability._engine_vocabulary()[1]
    mapping["L5"] = "last-engine"
    assert runner_config.conventional_tier_sim()["L5"] == "last-engine"
    assert "l5" in tier_affordability._engine_vocabulary()[1]


def test_model_interpreter_and_missing_error_preserve_legacy_registry(monkeypatch, tmp_path):
    monkeypatch.setattr(program_oracle, "ext_path", lambda name: tmp_path)
    python = tmp_path / ".venv" / "bin" / "python"
    with pytest.raises(program_oracle.OracleUnavailable) as exc:
        engines.selected_model_venv_python("fixture")
    assert str(exc.value) == f"model venv python absent: {python} (run `uv sync` in {tmp_path})"
    python.parent.mkdir(parents=True)
    python.touch()
    assert engines.selected_model_venv_python("fixture") == python
    monkeypatch.setattr(program_oracle, "_model_venv_python", lambda model: tmp_path / model)
    assert engines.selected_model_venv_python("replacement") == tmp_path / "replacement"


def test_core_model_override_not_shadowed_by_legacy_reexport(monkeypatch, tmp_path):
    monkeypatch.setattr(engines, "_model_venv_python", lambda model: tmp_path / model)
    assert engines.selected_model_venv_python("fixture") == tmp_path / "fixture"


def test_new_config_owner_is_in_grading_source_commitment():
    from merlin.targetgen import tier_cache

    assert "merlin.targetgen.runner_config" in tier_cache._GRADING_MODULES


def test_core_resolvers_run_with_evaluation_imports_blocked(tmp_path):
    script = r"""
import importlib.abc
import sys
from pathlib import Path
from types import SimpleNamespace
class BlockEvaluation(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "aet" or fullname.startswith("aet.") or fullname in {
            "merlin.targetgen.capsule_runner", "merlin.targetgen.capsule_golden",
        }:
            raise AssertionError("optional evaluation import: " + fullname)
sys.meta_path.insert(0, BlockEvaluation())
from merlin.targetgen import runner_config as config, program_engine_policy as engines
from merlin.targetgen import isa_taxonomy, mesh_program_run, target_experiment, tier_affordability
def absent(target):
    raise ValueError("absent")
target_experiment.load_capability_manifest = absent
assert config.selected_runner_config("fixture", None, "f32").fourth_output_name == "lowered.llvm.mlir"
assert tier_affordability._engine_vocabulary()[1] >= {"l2", "l3", "l4", "l5"}
assert "l0" not in tier_affordability._engine_vocabulary()[1]
engines.ext_path = lambda name: Path(sys.argv[1])
try:
    engines.selected_model_venv_python("fixture")
except engines.OracleUnavailable:
    pass
else:
    raise AssertionError("missing interpreter accepted")
isa_taxonomy._isa_def_path = lambda te: Path(__import__("json").__file__)
engines.selected_model_venv_python = lambda name: Path(sys.executable)
isa_taxonomy.subprocess.run = lambda *args, **kwargs: SimpleNamespace(returncode=1)
assert isa_taxonomy.derive_isa_taxonomy(SimpleNamespace(target="fixture"), model_ext="fixture") == {}
assert "merlin.targetgen.capsule_runner" not in sys.modules
assert "merlin.targetgen.program_oracle" not in sys.modules
from merlin.targetgen import tier_cache, oracle_policy
oracle_policy.selected_l3_engine_report = lambda target: {"available": True, "engine": "fixture-engine"}
assert tier_cache._engine_token("fixture", "L3", True) == "fixture-engine"
assert "merlin.targetgen.capsule_runner" not in sys.modules
assert "merlin.targetgen.program_oracle" not in sys.modules
from merlin.targetgen import program_oracle as program, oracle_policy, isa_model
oracle_policy._endpoint_of = lambda target: ("external_backend", "fixture")
isa_model.isa_model_for_target = lambda target: object()
program.run_program_oracle_smoke = lambda *args, **kwargs: {"ok": True, "reason": "fixture result"}
probe = SimpleNamespace(
    fixture={"kind": "named_program", "name": "fixture"},
    requirements={"operations": [{"domain": "instruction", "operation": "fixture"}]},
)
result = program.run_capability_probe(te=SimpleNamespace(target="fixture"), probe=probe, workdir=Path(sys.argv[1]))
assert result["observations"][0]["status"] == "supported"
assert "merlin.targetgen.capsule_runner" not in sys.modules
"""
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(str(p) for p in python_import_roots()),
        MERLIN_EXT_FIXTURE=str(tmp_path),
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
