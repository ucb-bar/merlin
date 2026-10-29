"""Static requirements reach only the selected treatment, without new brokers."""

from copy import deepcopy

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen import tool_registry as TR
from merlin.targetgen.contract.pass_catalog import load_catalog
from merlin.targetgen.contract.schemas import ContractViolation
from merlin.targetgen.generate_bundles import _ALL_ARMS, generate_bundles
from merlin.targetgen.generate_prompt import _enforced_workflow
from merlin.targetgen.target_experiment import load_target_experiment

pytestmark = pytest.mark.target("gemmini")


def test_shipped_catalog_is_requirements_not_a_proof_result():
    catalog = load_catalog()
    assert [role["id"] for role in catalog["roles"]] == ["register_allocation"]
    checks = catalog["roles"][0]["verification"]
    assert checks["storage_safety"]["availability"] == "deferred"
    assert checks["resource_geometry"]["availability"] == "deferred"
    assert checks["numerical_semantics"]["availability"] == "existing_grading"
    text = yaml.safe_dump(catalog)
    for forbidden in ("gemmini", "atlas", "VirtualAllocationPlan", "~/Projects"):
        assert forbidden not in text


@pytest.mark.parametrize("mutation", ["duplicate_id", "missing_obligation", "proof_claim", "extra_field"])
def test_malformed_catalog_refuses(tmp_path, mutation):
    catalog = deepcopy(load_catalog())
    role = catalog["roles"][0]
    if mutation == "duplicate_id":
        catalog["roles"].append(deepcopy(role))
    elif mutation == "missing_obligation":
        del role["obligations"]
    elif mutation == "proof_claim":
        role["verification"]["storage_safety"]["availability"] = "verified"
    else:
        role["required_pass_name"] = "hardcoded"
    source = tmp_path / "catalog.yaml"
    source.write_text(yaml.safe_dump(catalog))
    with pytest.raises(ContractViolation):
        load_catalog(source)


def test_duplicate_yaml_fields_refuse(tmp_path):
    path = tmp_path / "catalog.yaml"
    path.write_text("schema: merlin.pass_catalog.v1\nschema: overridden\nroles: []\n")
    with pytest.raises(ContractViolation, match="duplicate key"):
        load_catalog(path)


def test_catalog_is_opt_in_and_denied_in_all_other_arms():
    te = load_target_experiment(repo_root() / "examples/gemmini/target/descriptor.yaml")
    bundles = generate_bundles(te, arms=tuple(_ALL_ARMS))
    paths = set(TR.spec("pass_catalog").bundle_paths)
    for manifest in bundles.values():
        allowed = {row["path"] for row in manifest["allowed"]}
        denied = {row["path"] for row in manifest["denied"]}
        if manifest["arm"] == "merlin_pass_catalog":
            assert paths <= allowed
            assert not paths & denied
            assert set(manifest["tools"]) == set(TR.arm_tools("merlin_rtlchecks")) | {"pass_catalog"}
            assert set(TR.spec("verify_seam").bundle_paths) <= denied
        else:
            assert not paths & allowed
            assert paths <= denied  # includes schema beneath the common schemas/ grant
    assert all(manifest["arm"] != "merlin_pass_catalog" for manifest in generate_bundles(te).values())
    assert TR.spec("pass_catalog").broker is None


@pytest.mark.parametrize(
    "arm,add,drop,visible",
    [
        ("merlin_pass_catalog", (), ("pass_catalog",), False),
        ("merlin_rtlchecks", ("pass_catalog",), (), True),
    ],
)
def test_catalog_ablation_changes_effective_grants_and_prompt(arm, add, drop, visible):
    te = load_target_experiment(repo_root() / "examples/gemmini/target/descriptor.yaml")
    [manifest] = generate_bundles(te, arms=(arm,), add_tools=add, drop_tools=drop).values()
    allowed = {row["path"] for row in manifest["allowed"]}
    denied = {row["path"] for row in manifest["denied"]}
    paths = set(TR.spec("pass_catalog").bundle_paths)
    assert paths <= (allowed if visible else denied)
    assert not paths & (denied if visible else allowed)
    granted = TR.selected_bundle_tool_paths(allowed, manifest["tools"])
    prompt = _enforced_workflow(arm, "command_buffer", granted, te.target)
    assert ("pass_catalog.yaml" in prompt) is visible


def test_malformed_catalog_blocks_treatment_bundle(tmp_path, monkeypatch):
    te = load_target_experiment(repo_root() / "examples/gemmini/target/descriptor.yaml")
    monkeypatch.setenv("MERLIN_CONTRACT_DIR", str(tmp_path))
    with pytest.raises(ContractViolation, match="cannot read pass catalog"):
        generate_bundles(te, arms=("merlin_pass_catalog",))
