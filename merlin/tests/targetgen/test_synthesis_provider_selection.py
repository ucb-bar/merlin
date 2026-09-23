"""Actual synthesis/pipeline ownership follows selected OOT support, not directory guesses."""

from __future__ import annotations

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen import families, pipeline, target_registry
from merlin.targetgen.evidence.store import Evidence
from merlin.targetgen.providers import ProviderError
from merlin.targetgen.synthesize.dialect_plan import synthesize_dialect_plan
from merlin.targetgen.synthesize.target_contract import synthesize_target_contract


def _provider(root, name="synthetic", marker="external"):
    root.mkdir(parents=True)
    (root / "provider.yaml").write_text(
        yaml.safe_dump({"schema": "merlin.provider.v1", "id": marker, "target": name, "role": "support"})
    )
    contracts = root / "contracts"
    contracts.mkdir()
    contract = {
        "name": name,
        "version": "0.1",
        "capabilities": {"ops": []},
        "memory_model": {},
        "compiler_obligations": [],
        "hardware_promises": [],
        "runtime_promises": [],
        "legality": [],
        "notes": marker,
        "requires_human_review": True,
    }
    plan = {"target": name, "dialect_name": marker, "ops": [], "types": [], "lowering": [], "tests": []}
    (contracts / "target_contract.yaml").write_text(yaml.safe_dump(contract))
    (contracts / "dialect_plan.yaml").write_text(yaml.safe_dump(plan))
    return contract, plan


@pytest.fixture
def providers(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    monkeypatch.setenv("MERLIN_SCHEMAS_DIR", str(repo_root() / "merlin/schemas"))
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(repo))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "output"))
    monkeypatch.delenv("MERLIN_TARGET_EXPERIMENT", raising=False)
    _provider(repo / "merlin/targets/synthetic", marker="in_tree")
    root = tmp_path / "external"
    contract, plan = _provider(root)
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root))
    return root, contract, plan


def test_pipeline_uses_one_selected_external_support_owner(providers, tmp_path):
    root, contract, plan = providers
    assert target_registry.resolve("synthetic").base == root
    result = pipeline.build("synthetic", out=tmp_path / "generated", emit=["contract-only"])
    assert result.schema_problems == []
    assert result.plans["target_contract"] == contract
    assert result.plans["dialect_plan"] == plan
    assert yaml.safe_load((result.out / "contracts/target_contract.yaml").read_text()) == contract
    assert yaml.safe_load((result.out / "contracts/dialect_plan.yaml").read_text()) == plan


def test_selected_provider_missing_plan_synthesizes_without_in_tree_fallback(providers):
    root, contract, _ = providers
    (root / "contracts/dialect_plan.yaml").unlink()
    plan = synthesize_dialect_plan(Evidence(target="synthetic", sources={}), contract)
    assert plan["dialect_name"] == "synthetic"
    assert plan["requires_human_review"] is True


@pytest.mark.parametrize("member", ["target_contract", "dialect_plan"])
@pytest.mark.parametrize("bad", ["yaml", "nonmapping", "directory", "dangling", "escape"])
def test_invalid_selected_resources_refuse_before_pipeline_output(providers, tmp_path, member, bad):
    root, _, _ = providers
    path = root / "contracts" / (member + ".yaml")
    if bad == "yaml":
        path.write_text("broken: [")
    elif bad == "nonmapping":
        path.write_text("[]\n")
    else:
        path.unlink()
        if bad == "directory":
            path.mkdir()
        else:
            outside = tmp_path / "outside.yaml"
            if bad == "escape":
                outside.write_text("target: synthetic\n")
            path.symlink_to(outside)
    output = tmp_path / "must-not-exist"
    with pytest.raises((ValueError, OSError, yaml.YAMLError)):
        pipeline.build("synthetic", out=output, emit=["contract-only"])
    assert not output.exists()


def test_explicit_support_missing_contract_does_not_fall_back(providers):
    root, _, _ = providers
    (root / "contracts/target_contract.yaml").unlink()
    with pytest.raises(ProviderError, match="resource is not a file"):
        synthesize_target_contract(Evidence(target="synthetic", sources={}), "synthetic")


def test_selected_plan_cannot_name_a_different_target(providers):
    root, contract, plan = providers
    plan["target"] = "another"
    (root / "contracts/dialect_plan.yaml").write_text(yaml.safe_dump(plan))
    with pytest.raises(ValueError, match="differs from selected provider"):
        synthesize_dialect_plan(Evidence(target="synthetic", sources={}), contract)


def test_declared_contract_location_and_ordered_provider_precedence(providers, tmp_path, monkeypatch):
    root, contract, plan = providers
    alternate = tmp_path / "alternate"
    _provider(alternate, marker="wrong-provider")
    (root / "contracts/target_contract.yaml").rename(root / "definition.yaml")
    metadata = yaml.safe_load((root / "provider.yaml").read_text())
    metadata["contract"] = "definition.yaml"
    (root / "provider.yaml").write_text(yaml.safe_dump(metadata))
    import os

    monkeypatch.setenv("MERLIN_TARGET_PATH", os.pathsep.join(map(str, (root, alternate))))
    result = pipeline.build("synthetic", emit=["contract-only"])
    assert result.plans["target_contract"] == contract
    assert result.plans["dialect_plan"] == plan


def test_unprovided_target_retains_conservative_synthesis(providers):
    result = pipeline.build("unprovided", emit=["contract-only"])
    assert result.plans["target_contract"]["name"] == "unprovided"
    assert result.plans["target_contract"]["requires_human_review"] is True
    assert result.plans["dialect_plan"]["target"] == "unprovided"
    assert result.plans["dialect_plan"]["requires_human_review"] is True


@pytest.mark.parametrize("role", ["candidate_compiler", "host_schedule"])
def test_other_provider_roles_are_not_support_definitions(providers, role):
    root, _, _ = providers
    (root / "provider.yaml").write_text(
        yaml.safe_dump({"schema": "merlin.provider.v1", "id": "not-support", "target": "synthetic", "role": role})
    )
    result = pipeline.build("synthetic", emit=["contract-only"])
    assert result.plans["target_contract"]["notes"] == "in_tree"
    assert result.plans["dialect_plan"]["dialect_name"] == "in_tree"


def test_explicit_example_provider_overrides_family_seed(tmp_path, monkeypatch):
    name = families.DEFAULT_EXAMPLE_TARGET
    root = tmp_path / "provider"
    contract, plan = _provider(root, name=name)
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root))
    result = pipeline.build(name, emit=["contract-only"])
    assert result.plans["target_contract"] == contract
    assert result.plans["dialect_plan"] == plan
