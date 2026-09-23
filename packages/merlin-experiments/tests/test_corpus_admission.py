"""Constructed evaluator inventory stays optional without changing corpus policy."""

from __future__ import annotations

import builtins
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.corpus import admission as workflow

from merlin.common.paths import repo_root
from merlin.targetgen import capsule_runner as evaluator
from merlin.targetgen import conformance, corpora
from merlin.targetgen.contract import materialize
from merlin.targetgen.target_experiment import load_target_experiment


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    root = tmp_path / "corpus" / "isa"
    capsule = root / "example"
    capsule.mkdir(parents=True)
    (capsule / "capsule.yaml").write_text(
        "name: example\nkind: isa\nlabel: public\nrequired_oracle_tiers: [L0, L2, L3]\n",
        encoding="utf-8",
    )
    (capsule / "capsule.interface.mlir").write_text("module {}\n", encoding="utf-8")
    (capsule / "golden.yaml").write_text("outputs: {synthetic: PRIVATE_SYNTHETIC_SENTINEL}\n", encoding="utf-8")
    descriptor = tmp_path / "target_experiment.yaml"
    descriptor.write_text(
        yaml.safe_dump(
            {
                "target": "workflow_fixture",
                "capsule_corpus": str(root),
                "toolchain": {"sim_via": "fixture-engine"},
                "grading": {"expected_cohort": {"source_capsules": 1, "admitted_capsules": 1}},
            }
        ),
        encoding="utf-8",
    )
    return load_target_experiment(descriptor)


def _set_tiers(te, tiers):
    path = te.capsule_corpus / "example/capsule.yaml"
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    doc["required_oracle_tiers"] = tiers
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")


def _read_tiers(root):
    doc = yaml.safe_load((root / "example/capsule.yaml").read_text(encoding="utf-8"))
    return doc["required_oracle_tiers"], doc["oracle_tier_ceiling"]


def test_default_constructs_twice_and_keeps_highest_declared_tier(cohort, monkeypatch, capsys):
    calls = []

    def construct(target, sim_via):
        calls.append((target, sim_via))
        return {"L2": lambda: None, "L3": lambda: None}

    monkeypatch.setattr(evaluator, "oracle_adapters", construct)
    root = workflow.public_capsules_for(cohort)
    assert calls == [(cohort.target, cohort.sim_via)] * 2
    assert root.is_symlink()
    assert _read_tiers(root) == (["L0", "L2", "L3"], "L3")
    assert workflow.validate_materialized_cohort(root, cohort)["n_admitted_capsules"] == 1
    assert workflow.validate_materialized_cohort is materialize.validate_materialized_cohort
    assert (root / "example/golden.yaml").read_bytes() == (cohort.capsule_corpus / "example/golden.yaml").read_bytes()
    assert capsys.readouterr() == ("", "")


def test_explicit_frozen_roots_publish_once_without_shared_cache(cohort, tmp_path, monkeypatch):
    frozen = tmp_path / "staged-source"
    shutil.copytree(cohort.capsule_corpus, frozen)
    (cohort.capsule_corpus / "example/capsule.yaml").unlink()
    monkeypatch.setattr(evaluator, "oracle_adapters", lambda *args: {"L2": lambda: None, "L3": lambda: None})
    destination = tmp_path / "run/public"
    root = workflow.public_capsules_for(cohort, corpus_roots=[frozen], destination=destination)
    assert root == destination
    assert not root.is_symlink()
    assert _read_tiers(root) == (["L0", "L2", "L3"], "L3")
    assert materialize.validate_materialized_cohort(root, cohort, corpus_roots=[frozen])["n_source_capsules"] == 1
    assert not (tmp_path / "out").exists()
    before = (root / ".cohort_admission.json").read_bytes()
    with pytest.raises(FileExistsError):
        workflow.public_capsules_for(cohort, corpus_roots=[frozen], destination=destination)
    assert (root / ".cohort_admission.json").read_bytes() == before


@pytest.mark.parametrize("location", ["source", "ancestor", "descendant"])
def test_explicit_destination_cannot_overlap_source(cohort, monkeypatch, location):
    root = cohort.capsule_corpus
    destination = {"source": root, "ancestor": root.parent, "descendant": root / "derived"}[location]
    before = (root / "example/capsule.yaml").read_bytes()
    monkeypatch.setattr(evaluator, "oracle_adapters", lambda *args: {"L3": lambda: None})
    with pytest.raises(ValueError, match="overlap"):
        workflow.public_capsules_for(cohort, corpus_roots=[root], destination=destination)
    assert (root / "example/capsule.yaml").read_bytes() == before
    assert not (root / "derived").exists()


def test_loaded_loop_override_and_second_construction_keep_original_order(cohort, monkeypatch):
    calls = []

    def loop(target, sim_via, *, declared_tiers):
        calls.append(("loop", declared_tiers))
        return {"L2": lambda: None}

    def full(target, sim_via):
        calls.append(("full", None))
        return {"L3": lambda: None}

    monkeypatch.setattr(evaluator, "qa_loop_adapters", loop)
    monkeypatch.setattr(evaluator, "oracle_adapters", full)
    root = workflow.public_capsules_for(cohort)
    assert calls == [("loop", {"L0", "L2", "L3"}), ("full", None)]
    assert _read_tiers(root)[1] == "L3"


def test_second_observation_not_replaced_by_cached_first_inventory(cohort, monkeypatch):
    observations = iter([{"L2": lambda: None, "L3": lambda: None}, {"L2": lambda: None}])
    monkeypatch.setattr(evaluator, "oracle_adapters", lambda *args: next(observations))
    assert _read_tiers(workflow.public_capsules_for(cohort)) == (["L0", "L2"], "L2")


def test_disjoint_declared_tiers_refuse_without_publishing(cohort, monkeypatch, tmp_path):
    _set_tiers(cohort, ["L0", "L5"])
    monkeypatch.setattr(evaluator, "oracle_adapters", lambda *args: {"L2": lambda: None})
    with pytest.raises(ValueError, match="Refusing to substitute") as error:
        workflow.public_capsules_for(cohort)
    assert "['L0', 'L5']" in str(error.value)
    assert "reaches ['L2']" in str(error.value)
    assert not (tmp_path / "out").exists()


def test_no_constructed_adapters_retains_legacy_floor(cohort, monkeypatch):
    _set_tiers(cohort, ["L0", "L2"])
    monkeypatch.setattr(evaluator, "oracle_adapters", lambda *args: {})
    assert _read_tiers(workflow.public_capsules_for(cohort)) == (["L0", "L2"], "L2")


def test_no_adapters_preserves_unreachable_declared_tiers_for_grader(cohort, monkeypatch):
    _set_tiers(cohort, ["L0", "L3"])
    monkeypatch.setattr(evaluator, "oracle_adapters", lambda *args: {})
    root = workflow.public_capsules_for(cohort)
    doc = yaml.safe_load((root / "example/capsule.yaml").read_text(encoding="utf-8"))
    assert doc["required_oracle_tiers"] == ["L0"]
    assert doc["unreachable_required_oracle_tiers"] == ["L3"]
    assert doc["oracle_tier_ceiling"] == "L2"


def test_empty_declaration_keeps_legacy_fastest_loop_choice(cohort, monkeypatch):
    _set_tiers(cohort, [])
    monkeypatch.setattr(evaluator, "oracle_adapters", lambda *args: {"L3": lambda: None, "L2": lambda: None})
    assert _read_tiers(workflow.public_capsules_for(cohort)) == ([], "L2")


def test_unknown_constructed_tier_is_not_silently_filtered(cohort, monkeypatch):
    _set_tiers(cohort, ["future-tier"])
    monkeypatch.setattr(evaluator, "oracle_adapters", lambda *args: {"future-tier": lambda: None})
    with pytest.raises(ValueError, match="unknown tier ceiling"):
        workflow.public_capsules_for(cohort)


@pytest.mark.parametrize("failure_call", [1, 2])
def test_factory_errors_propagate_from_materialization(cohort, monkeypatch, failure_call):
    calls = 0

    def construct(*args):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise RuntimeError("synthetic construction failure")
        return {"L2": lambda: None}

    monkeypatch.setattr(evaluator, "oracle_adapters", construct)
    with pytest.raises(RuntimeError, match="synthetic construction failure"):
        workflow.public_capsules_for(cohort)
    assert calls == failure_call


@pytest.mark.parametrize("entry", [workflow.public_capsules_for, materialize.materialize_public_cohort])
def test_explicit_ceiling_never_imports_or_constructs_evaluator(cohort, monkeypatch, entry):
    original = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "merlin.targetgen.capsule_runner" or "capsule_runner" in (fromlist or ()):
            raise AssertionError("explicit ceiling imported evaluator")
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(evaluator, "oracle_adapters", lambda *args: pytest.fail("constructed evaluator"))
    assert _read_tiers(entry(cohort, tier_ceiling="L3"))[1] == "L3"


def test_constructed_inventory_uses_standard_descriptor_and_current_callbacks(cohort, monkeypatch):
    seen = []
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", "/must/not/be/read.yaml")
    monkeypatch.setattr(corpora, "standard_descriptor_path", lambda target: cohort.path)
    monkeypatch.setattr(corpora, "descriptor_path", lambda target: pytest.fail("used descriptor override"))

    def construct(target, sim_via):
        seen.append((target, sim_via))
        return {"L3": lambda: None, "L2": lambda: None, "future-tier": lambda: None}

    monkeypatch.setattr(evaluator, "oracle_adapters", construct)
    assert workflow.constructed_oracle_tiers(cohort.target) == ["L2", "L3", "future-tier"]
    assert seen == [(cohort.target, cohort.sim_via)]
    monkeypatch.setattr(evaluator, "oracle_adapters", lambda *args: {})
    assert workflow.constructed_oracle_tiers(cohort.target) == []


@pytest.mark.parametrize("failure", ["missing", "malformed", "factory"])
def test_unresolvable_conformance_inventory_remains_unknown(cohort, monkeypatch, tmp_path, failure):
    descriptor = cohort.path
    if failure == "missing":
        descriptor = tmp_path / "absent.yaml"
    elif failure == "malformed":
        descriptor.write_text("[unfinished", encoding="utf-8")
    monkeypatch.setattr(corpora, "standard_descriptor_path", lambda target: descriptor)

    def construct(*args):
        raise RuntimeError("synthetic construction failure")

    monkeypatch.setattr(evaluator, "oracle_adapters", construct)
    assert workflow.constructed_oracle_tiers(cohort.target) == []


def test_muon_factory_drop_is_observed_not_replaced_by_advertised_tiers(monkeypatch):
    from merlin.runtime.backends.base import get_backend
    from merlin.targetgen import oracle_policy, program_engine_policy

    plugin = get_backend("muon").muon_oracles
    te = load_target_experiment(
        repo_root() / "merlin/experiments/capsule_bench/targets/radiance/target_experiment.yaml"
    )
    for key in ("MERLIN_MUON_L3_VERILATOR_ALSO", "MERLIN_EXEC_SMOKE", "MERLIN_MUON_SKIP_RTL_L3"):
        monkeypatch.setenv(key, "0")
    monkeypatch.setattr(
        plugin,
        "l3_selection",
        lambda target: {
            "engine": "gsim",
            "fidelity": "elaborated_rtl",
            "reason": "synthetic selection",
            "considered": [],
            "passed_over": [],
        },
    )

    def absent(*args, **kwargs):
        raise program_engine_policy.OracleUnavailable("synthetic construction unavailable")

    monkeypatch.setattr(plugin, "gsim_muon_adapter", absent)
    assert oracle_policy.oracle_tier_plan(te.target, te.sim_via).tiers == ("L2", "L3")
    assert workflow.constructed_oracle_tiers(te.target) == ["L2"]


@pytest.fixture
def derivation(monkeypatch):
    from merlin.targetgen import boundary, memory_regime

    monkeypatch.setattr(conformance, "required_cells", lambda *args, **kwargs: ({}, {}))
    monkeypatch.setattr(conformance, "boundaries", lambda *args: conformance.Boundaries())
    monkeypatch.setattr(conformance, "_certified_depth", lambda *args, **kwargs: (None, "fixture"))
    for name in (
        "host_lane_cells",
        "host_only_dtypes",
        "_shape_axis",
        "_epilogue_axis",
        "_group_axis",
        "_carried_state_axis",
        "_application_axis",
        "_cert_affordability",
        "geometry_axis",
        "scope_axis",
        "_conv_geometry_axis",
    ):
        monkeypatch.setattr(conformance, name, lambda *args, **kwargs: {})
    monkeypatch.setattr(
        boundary,
        "required_boundaries",
        lambda *args: {"by_kind": {}, "whole_model_shape": {}, "captures_unreadable": {}},
    )
    for name in ("required_regimes", "reduction_depth_regimes", "required_regime_extents"):
        monkeypatch.setattr(memory_regime, name, lambda *args, **kwargs: {})


def test_non_evaluating_derivation_requires_explicit_inventory(derivation, monkeypatch):
    monkeypatch.setattr(workflow, "constructed_oracle_tiers", lambda *args: pytest.fail("constructed evaluator"))
    with pytest.raises(TypeError, match="oracle_tiers"):
        conformance.derive_spec("fixture", {})
    doc = conformance.derive_spec("fixture", {}, oracle_tiers=[])
    assert doc["oracle_tiers"] == []
    assert doc["generated_by"] == "merlin.targetgen.conformance.spec"
    assert doc["cells"] == []


def test_evaluated_spec_preserves_core_helper_and_workflow_lookup_overrides(derivation, monkeypatch):
    monkeypatch.setattr(workflow, "constructed_oracle_tiers", lambda target: ["L2"])
    assert workflow.conformance_spec("fixture", {})["oracle_tiers"] == ["L2"]
    monkeypatch.setattr(workflow, "constructed_oracle_tiers", lambda target: ["L3"])
    monkeypatch.setattr(conformance, "_shape_axis", lambda target: {"synthetic_override": True})
    observed = workflow.conformance_spec("fixture", {})
    assert observed["oracle_tiers"] == ["L3"]
    assert observed["shape_generalization"] == {"synthetic_override": True}


def test_earlier_derivation_failure_does_not_construct_unused_adapters(derivation, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("synthetic capture failure")

    monkeypatch.setattr(conformance, "required_cells", fail)
    monkeypatch.setattr(workflow, "constructed_oracle_tiers", lambda *args: pytest.fail("queried too early"))
    with pytest.raises(ValueError, match="synthetic capture failure"):
        workflow.conformance_spec("fixture", {})


def test_observer_is_resolved_late_once_after_prior_helper_override(derivation, monkeypatch):
    calls = []

    def observe(target):
        calls.append(("query", target))
        return ["L3"]

    def preceding_helper(target):
        calls.append(("shape", target))
        monkeypatch.setattr(workflow, "constructed_oracle_tiers", observe)
        return {}

    monkeypatch.setattr(workflow, "constructed_oracle_tiers", lambda *args: pytest.fail("queried before helper"))
    monkeypatch.setattr(conformance, "_shape_axis", preceding_helper)
    doc = workflow.conformance_spec("fixture", {})
    assert calls == [("shape", "fixture"), ("query", "fixture")]
    assert doc["oracle_tiers"] == ["L3"]


def test_explicit_and_observed_specs_serialize_identically(derivation, monkeypatch):
    cell = conformance.Cell("contraction", "i8", "partial")
    origin = conformance.CellOrigin(observed_in=("synthetic_capture",), n_regions=1)
    monkeypatch.setattr(conformance, "required_cells", lambda *args, **kwargs: ({cell: origin}, {}))
    monkeypatch.setattr(conformance, "boundaries", lambda *args: conformance.Boundaries(tile_edge=8))
    monkeypatch.setattr(conformance, "_cs", lambda: SimpleNamespace(shape_quantum=lambda *args, **kwargs: 1))
    monkeypatch.setattr(workflow, "constructed_oracle_tiers", lambda target: ["L2", "L3"])
    arguments = {"personas": {"fixture": {"description": "synthetic"}}, "cert_budget_s": 3.0}
    explicit = conformance.derive_spec("fixture", {}, oracle_tiers=["L2", "L3"], **arguments)
    observed = workflow.conformance_spec("fixture", {}, **arguments)
    # Do not sort keys: both values and recorded field order remain identical.
    assert json.dumps(explicit) == json.dumps(observed)
    assert explicit["generated_by"] == "merlin.targetgen.conformance.spec"
    assert explicit["cells"][0]["cell"] == "contraction/i8/partial"


@pytest.mark.parametrize(
    "module,attribute",
    [(conformance, "spec"), (conformance, "_declared_oracle_tiers"), (materialize, "public_capsules_for")],
)
def test_removed_evaluated_attributes_cannot_import_optional_owners(monkeypatch, module, attribute):
    original = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("merlin_experiments"):
            pytest.fail("core attribute lookup attempted an optional workflow import")
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(AttributeError):
        getattr(module, attribute)


def test_core_only_explicit_materialization_validates_real_temp_cohort(cohort, tmp_path):
    script = """
import importlib.abc
import os
import sys
sys.path[:0] = sys.argv[1:3]
os.environ['MERLIN_OUT_ROOT'] = sys.argv[4]
class NoResearch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('merlin_experiments') or fullname == 'merlin.targetgen.capsule_runner':
            raise AssertionError('research imported by explicit core materialization')
sys.meta_path.insert(0, NoResearch())
from merlin.targetgen import conformance
from merlin.targetgen.contract import materialize
from merlin.targetgen.target_experiment import load_target_experiment
te = load_target_experiment(sys.argv[3])
root = materialize.materialize_public_cohort(te, tier_ceiling='L3')
assert root.is_symlink()
assert materialize.validate_materialized_cohort(root, te)['n_admitted_capsules'] == 1
assert conformance.Cell('contraction', 'i8', None).key() == 'contraction/i8'
assert not hasattr(conformance, 'spec')
assert not hasattr(conformance, '_declared_oracle_tiers')
assert not hasattr(materialize, 'public_capsules_for')
assert not any(name.startswith('merlin_experiments') for name in sys.modules)
print('core-only materialization verified')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            script,
            str(repo_root() / "src"),
            str(Path(yaml.__file__).parents[1]),
            str(cohort.path),
            str(tmp_path / "isolated-out"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "core-only materialization verified\n"


@pytest.mark.parametrize("owner", ["admission", "preparation", "release"])
def test_workflow_grader_mask_is_nonvacuous_and_source_pin_is_present(tmp_path, monkeypatch, owner):
    from merlin.common import access
    from merlin.targetgen import tier_cache
    from merlin.targetgen.sandbox import bwrap

    surfaces_module = importlib.import_module("merlin.targetgen.sandbox.answer_surfaces")
    module = f"merlin_experiments.corpus.{owner}"
    source = tmp_path / f"packages/merlin-experiments/src/merlin_experiments/corpus/{owner}.py"
    source.parent.mkdir(parents=True)
    source.write_text("# synthetic host workflow\n", encoding="utf-8")
    monkeypatch.setattr(access, "sys", SimpleNamespace(path=[], prefix=str(tmp_path / "python"), modules={}))
    monkeypatch.setattr(surfaces_module, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(surfaces_module, "artifacts_dir", lambda: tmp_path / "out/artifacts")
    monkeypatch.setattr(surfaces_module, "_evicted_oracle_modules", lambda: [])
    monkeypatch.setattr(surfaces_module, "experimenter_memory_dir", lambda: tmp_path / "absent-memory")
    te = SimpleNamespace(
        target="fixture",
        capsule_corpus=None,
        corpus_siblings=lambda: (),
        hidden_corpus=lambda: None,
        prior_backends=(),
        backend_package=None,
    )
    assert module in access.declared_modules("grader")
    assert {
        "merlin_experiments.corpus.admission",
        "merlin.targetgen.conformance",
        "merlin.targetgen.contract.materialize",
    } <= set(tier_cache._GRADING_MODULES)
    surfaces = surfaces_module.answer_surfaces(te)
    assert any(surface.path == source and surface.origin == "grader" for surface in surfaces)
    unmasked = ["--ro-bind", str(tmp_path), str(tmp_path)]
    assert {surface.path for surface in bwrap.coverage_gap(unmasked, surfaces)} == {source}
    assert bwrap.coverage_gap(bwrap.apply_answer_masks(unmasked, surfaces), surfaces) == []
