"""Tier inventory is host metadata, independent of optional adapter construction."""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from merlin_experiments.phase0.declarations import all_declarations, for_target

from merlin.common.paths import python_import_roots, repo_root
from merlin.targetgen import capsule_runner as evaluator
from merlin.targetgen import oracle_policy as policy
from merlin.targetgen import program_engine_policy as program_policy
from merlin.targetgen import program_oracle


def _selected(engine="gsim"):
    return {
        "engine": engine,
        "fidelity": "elaborated_rtl",
        "reason": "fixture engine",
        "considered": [{"engine": engine, "available": True, "reason": "fixture engine"}],
        "passed_over": [],
    }


def _absent(*args, **kwargs):
    raise program_policy.OracleUnavailable("fixture engine absent")


def test_plan_owns_nested_selection_and_returns_independent_serializable_copies():
    original = _selected()
    original["considered"][0]["paths"] = ["fixture-engine"]
    plan = policy.OracleTierPlan(("L2", "L3"), selection=original)
    expected = json.loads(json.dumps(original))
    original["considered"][0]["paths"].append("mutated input")
    original["passed_over"].append("mutated input")
    returned = plan.selection
    returned["considered"][0]["paths"].clear()
    returned["passed_over"].append("mutated result")
    returned["engine"] = "different-engine"
    assert plan.selection == expected
    assert isinstance(plan.selection, dict)
    assert isinstance(plan.selection["considered"], list)
    assert plan.selection is not plan.selection
    copied = replace(plan, route="program")
    assert copied.selection == expected and copied.route == "program"
    record = asdict(plan)
    assert json.loads(json.dumps(record))["selection"] == expected
    record["selection"]["considered"][0]["paths"].append("mutated serialization")
    assert plan.selection == copied.selection == expected


@pytest.fixture
def isolated(monkeypatch):
    monkeypatch.setattr(policy, "_ensure_sim_metadata_discovered", lambda: None)
    monkeypatch.setattr(evaluator, "_ensure_sim_oracles_discovered", lambda: None)
    monkeypatch.setattr(evaluator, "_endpoint_of", lambda target: ("inline_asm_insn", None))
    monkeypatch.setattr(evaluator, "_bespoke_sim_via", lambda target: "chipyard")


@pytest.mark.parametrize("available", [True, False])
def test_chipyard_inventory_matches_evaluator_including_existing_arc_fallback(isolated, monkeypatch, available):
    monkeypatch.setattr(evaluator, "chipyard_l3_selection", (lambda target: _selected()) if available else _absent)
    plan = policy.oracle_tier_plan("fixture")
    adapters = evaluator.oracle_adapters("fixture")
    assert plan.tiers == tuple(sorted(adapters)) == ("L2", "L3")
    assert (plan.selection is not None) is available
    assert (adapters["L3"].__qualname__.startswith("mlc_arc_adapter")) is not available


@pytest.mark.parametrize("available", [True, False])
def test_program_inventory_matches_evaluator_without_constructing_adapters(isolated, monkeypatch, available):
    monkeypatch.setattr(evaluator, "_endpoint_of", lambda target: ("external_backend", "fixture_model"))
    monkeypatch.setattr(program_oracle, "select_rtl_engine", (lambda target: _selected()) if available else _absent)
    monkeypatch.setattr(program_oracle, "_rtl_engine_dir", lambda target, engine: None)
    plan = policy.oracle_tier_plan("fixture", "")
    adapters = evaluator.oracle_adapters("fixture", "")
    assert plan.tiers == tuple(sorted(adapters)) == (("L2", "L3") if available else ("L2",))
    assert plan.model_ext == "fixture_model"


def test_plan_requires_program_model_binding(isolated, monkeypatch):
    monkeypatch.setattr(evaluator, "_endpoint_of", lambda target: ("external_backend", None))
    for resolve in (policy.oracle_tier_plan, evaluator.oracle_adapters):
        with pytest.raises(ValueError, match="runner.model_ext"):
            resolve("fixture", "")


def test_explicit_empty_sim_is_not_reresolved(isolated):
    assert policy.oracle_tier_plan("fixture", "").tiers == ("L3",)
    assert policy.oracle_tier_plan("fixture", "").sim_via == ""


def test_exclusive_metadata_owns_its_tiers_without_calling_adapter_or_program_route(isolated, monkeypatch):
    def forbidden(target):
        raise AssertionError("metadata instantiated an execution adapter")

    oracle = policy._SimOracle(
        forbidden,
        lambda target: (False, "fixture"),
        True,
        tier_plan=lambda target: policy.OracleTierPlan(("custom-tier",)),
    )
    monkeypatch.setitem(policy._SIM_ORACLES, "fixture-exclusive", oracle)
    monkeypatch.setattr(evaluator, "_endpoint_of", forbidden)
    plan = policy.oracle_tier_plan("fixture", "fixture-exclusive")
    assert plan.tiers == ("custom-tier",) and plan.route == "exclusive"


def test_legacy_plugin_evaluation_survives_metadata_refusal(isolated, monkeypatch):
    adapters = {"plugin-tier": lambda *args: None}
    oracle = policy._SimOracle(lambda target: adapters, lambda target: (False, "fixture"), True)
    monkeypatch.setitem(policy._SIM_ORACLES, "legacy-fixture", oracle)
    assert evaluator.oracle_adapters("fixture", "legacy-fixture") is adapters
    with pytest.raises(policy.OracleMetadataUnavailable, match="tier_plan="):
        policy.oracle_tier_plan("fixture", "legacy-fixture")


def test_plugin_requirement_inference_defaults_to_explicit_refusal(isolated, monkeypatch):
    oracle = policy._SimOracle(
        lambda target: {"L3": lambda *args: None},
        lambda target: (False, "fixture"),
        True,
        tier_plan=lambda target: policy.OracleTierPlan(("L3",)),
    )
    monkeypatch.setitem(policy._SIM_ORACLES, "unproved-fixture", oracle)
    assert policy.oracle_tier_plan("fixture", "unproved-fixture").tiers == ("L3",)
    with pytest.raises(policy.OracleMetadataUnavailable, match="datapath.required_oracle_tiers"):
        policy.inferred_oracle_tiers("fixture", "unproved-fixture")


def test_muon_factory_failure_cannot_silently_add_a_required_tier(monkeypatch):
    from merlin.runtime.backends.base import get_backend
    from merlin.targetgen.corpus_spec import derive_binding, profile_datapath
    from merlin.targetgen.target_experiment import load_target_experiment

    plugin = get_backend("muon").muon_oracles
    te = load_target_experiment(
        repo_root() / "merlin/experiments/capsule_bench/targets/radiance/target_experiment.yaml"
    )
    for key in ("MERLIN_MUON_L3_VERILATOR_ALSO", "MERLIN_EXEC_SMOKE", "MERLIN_MUON_SKIP_RTL_L3"):
        monkeypatch.setenv(key, "0")
    monkeypatch.setattr(plugin, "l3_selection", lambda target: _selected())
    monkeypatch.setattr(plugin, "gsim_muon_adapter", _absent)
    # The legacy evaluator drops the selected L3 after construction fails. Its
    # behavior must remain unchanged; metadata cannot guess the resulting keys.
    assert sorted(evaluator.oracle_adapters(te.target, te.sim_via)) == ["L2"]
    advertised = policy.oracle_tier_plan(te.target, te.sim_via)
    assert advertised.tiers == ("L2", "L3")
    assert advertised.requirements_inference_safe is False
    with pytest.raises(policy.OracleMetadataUnavailable, match="datapath.required_oracle_tiers"):
        derive_binding(te, {})
    from merlin.common.yaml import load_yaml

    explicit = profile_datapath(load_yaml(for_target(te.target).recipe))
    assert derive_binding(te, explicit).tiers == explicit["required_oracle_tiers"] == ["L0", "L1", "L2"]


def test_all_explicit_catalog_profile_requirements_bypass_metadata(monkeypatch):
    from merlin.common.yaml import load_yaml
    from merlin.targetgen import corpus_spec, target_experiment

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit profile requirements must not resolve metadata")

    monkeypatch.setattr(policy, "inferred_oracle_tiers", forbidden)
    contract = {"compute_units": [{"dtypes": ["int8"]}], "capabilities": {"mesh": {"rows": 4}}}
    monkeypatch.setattr(
        target_experiment, "load_capability_manifest", lambda target: SimpleNamespace(contract=contract)
    )
    checked = []
    for declaration in all_declarations():
        datapath = (load_yaml(declaration.recipe) or {}).get("datapath", {})
        if not datapath.get("required_oracle_tiers"):
            continue
        te = SimpleNamespace(target=declaration.target, sim_via="unproved-fixture")
        assert corpus_spec.derive_binding(te, datapath).tiers == datapath["required_oracle_tiers"]
        checked.append(declaration.profile)
    assert "radiance" in checked


def test_missing_declared_plugin_metadata_never_guesses_arc(isolated):
    with pytest.raises(policy.OracleMetadataUnavailable, match="plugin.sim_oracle_metadata"):
        policy.oracle_tier_plan("fixture", "missing-fixture")
    # The metadata boundary is stricter; this migration does not silently alter
    # the legacy evaluator's fallback policy for existing experiment callers.
    assert tuple(evaluator.oracle_adapters("fixture", "missing-fixture")) == ("L3",)


def test_builtin_adapter_binding_never_imports_evaluation(isolated, monkeypatch):
    monkeypatch.delitem(policy._SIM_ADAPTER_FACTORIES, "chipyard", raising=False)
    with pytest.raises(policy.OracleMetadataUnavailable, match="merlin-experiments"):
        policy._chipyard_adapters("fixture")


def test_program_policy_reexports_exact_legacy_identities():
    for name in ("_RTL_ENGINES", "_rtl_engine_dir", "_rtl_engine_probe", "select_rtl_engine", "OracleUnavailable"):
        assert getattr(program_policy, name) is getattr(program_oracle, name)


@pytest.mark.parametrize("engine", ["gsim", "verilator", None])
@pytest.mark.parametrize("also,smoke,skip", list(itertools.product([False, True], repeat=3)))
def test_support_plugin_metadata_matches_all_existing_tier_switches(monkeypatch, engine, also, smoke, skip):
    from merlin.runtime.backends.base import get_backend

    plugin = get_backend("muon").muon_oracles
    monkeypatch.setattr(plugin, "l3_selection", (lambda target: _selected(engine)) if engine else _absent)
    for key, enabled in (
        ("MERLIN_MUON_L3_VERILATOR_ALSO", also),
        ("MERLIN_EXEC_SMOKE", smoke),
        ("MERLIN_MUON_SKIP_RTL_L3", skip),
    ):
        monkeypatch.setenv(key, "1" if enabled else "0")
    expected = {"L2"}
    if not skip:
        if engine:
            expected.add("L3")
        if engine == "gsim" and also:
            expected.add("L3-verilator")
        if smoke:
            expected.add("L3-smoke")
    assert set(plugin.tier_plan("fixture").tiers) == expected
    assert set(plugin.default_adapters("fixture")) == expected


def test_core_only_corpus_binding_refuses_to_import_optional_evaluation():
    code = """
import importlib.abc, sys
from types import SimpleNamespace
class BlockEvaluation(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "aet" or fullname.startswith("aet.") or fullname in {
            "merlin.targetgen.capsule_runner", "merlin.targetgen.capsule_grade",
            "merlin.targetgen.capsule_golden", "merlin.targetgen.program_oracle",
        }:
            raise AssertionError("metadata imported evaluation: " + fullname)
sys.meta_path.insert(0, BlockEvaluation())
from merlin.targetgen import oracle_policy as policy, corpus_spec, target_experiment
from merlin.targetgen import program_engine_policy as program
policy._ensure_sim_metadata_discovered = lambda: None
policy._endpoint_of = lambda target: ("inline_asm_insn", None)
policy.chipyard_l3_selection = lambda target: {"engine": "fixture"}
contract = {"compute_units": [{"dtypes": ["int8"]}], "capabilities": {"mesh": {"rows": 4}}}
target_experiment.load_capability_manifest = lambda target: SimpleNamespace(contract=contract)
target = SimpleNamespace(target="fixture", sim_via="chipyard")
binding = corpus_spec.derive_binding(target, {"accum_dtype": "i32"})
assert binding.tiers == ["L2", "L3"]
assert binding.tile_dim == 4
policy._endpoint_of = lambda target: ("external_backend", "fixture_model")
program.select_rtl_engine = lambda target: {"engine": "fixture"}
assert policy.oracle_tier_plan("fixture", "").tiers == ("L2", "L3")
target.sim_via = "unregistered-fixture"
explicit_profile = {"accum_dtype": "i32", "required_oracle_tiers": ["declared"]}
assert corpus_spec.derive_binding(target, explicit_profile).tiers == ["declared"]
assert "merlin.targetgen.capsule_runner" not in sys.modules
assert "merlin.targetgen.program_oracle" not in sys.modules
"""
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(str(path) for path in python_import_roots()))
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_metadata_discovery_does_not_load_legacy_evaluator_registration(monkeypatch):
    from merlin.runtime.backends import base

    requested = []
    monkeypatch.setattr(policy, "_sim_metadata_env_seen", None)
    monkeypatch.setattr(base, "_oot_plugin_modules", lambda key: requested.append(key) or [])
    policy._ensure_sim_metadata_discovered()
    assert requested == ["sim_oracle_metadata"]


def test_real_support_metadata_discovery_does_not_import_evaluation():
    code = """
import importlib.abc, sys
class BlockEvaluation(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("aet") or fullname == "merlin.targetgen.capsule_runner":
            raise AssertionError("support metadata imported optional evaluation: " + fullname)
sys.meta_path.insert(0, BlockEvaluation())
from merlin.targetgen.oracle_policy import oracle_tier_plan
plan = oracle_tier_plan("phantom5", "phantomsim")
assert plan.tiers == () and plan.route == "exclusive"
assert "merlin.targetgen.capsule_runner" not in sys.modules
"""
    environment = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(str(path) for path in python_import_roots()),
        MERLIN_TARGET_PATH=str(repo_root() / "merlin/tests/fixtures/phantom_target"),
    )
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
