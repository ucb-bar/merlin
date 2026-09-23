"""Core lowering must use one selected support contract, including the neutral example."""

import copy
import json
import subprocess
import sys

import pytest
import yaml

from merlin.common.paths import checkout_root, repo_root
from merlin.xdsl_dialects.lowering import pipeline

pytestmark = pytest.mark.target("toy_npu")


@pytest.fixture
def selected(tmp_path, monkeypatch):
    plan = (repo_root() / "examples/toy_npu/target/contracts/dialect_plan.yaml").read_text()
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path / "repo"))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.delenv("MERLIN_TARGET_CONTRACT", raising=False)
    provider = tmp_path / "selected"
    contracts = provider / "contracts"
    contracts.mkdir(parents=True)
    (provider / "provider.yaml").write_text(
        "schema: merlin.provider.v1\nid: selected\ntarget: toy_npu\nrole: support\n"
    )
    source = checkout_root() / "examples/toy_npu/target/contracts/target_contract.yaml"
    contract = copy.deepcopy(yaml.safe_load(source.read_text()))
    contract["notes"] = "selected external support"
    (contracts / "target_contract.yaml").write_text(yaml.safe_dump(contract))
    (contracts / "dialect_plan.yaml").write_text(plan)
    native = tmp_path / "repo/merlin/targets/toy_npu/contracts"
    native.mkdir(parents=True)
    (native / "target_contract.yaml").write_text("name: toy_npu\nnotes: unrelated native contract\n")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(provider))
    return provider, contract, native


def test_selected_contract_wins_native_same_name(selected):
    _, contract, _ = selected
    assert pipeline.load_curated_contract("toy_npu") == contract


@pytest.mark.parametrize("invalid", ["absent", "malformed", "nonmapping"])
def test_invalid_selected_contract_never_falls_back(selected, invalid):
    provider, _, _ = selected
    path = provider / "contracts/target_contract.yaml"
    if invalid == "absent":
        path.unlink()
    else:
        path.write_text("broken: [" if invalid == "malformed" else "[]")
    with pytest.raises(pipeline.LoweringError, match="MERLIN_TARGET_PATH"):
        pipeline.load_curated_contract("toy_npu")


def test_authored_example_is_read_without_selected_support(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path / "repo"))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.delenv("MERLIN_TARGET_PATH", raising=False)
    monkeypatch.delenv("MERLIN_TARGET_CONTRACT", raising=False)
    authored = checkout_root() / "examples/toy_npu/target/contracts/target_contract.yaml"
    assert pipeline.load_curated_contract("toy_npu") == yaml.safe_load(authored.read_text())


def test_authored_example_contract_remains_selected(monkeypatch):
    monkeypatch.delenv("MERLIN_TARGET_PATH", raising=False)
    monkeypatch.delenv("MERLIN_TARGET_CONTRACT", raising=False)
    path = repo_root() / "examples/toy_npu/target/contracts/target_contract.yaml"
    assert pipeline.load_curated_contract("toy_npu") == yaml.safe_load(path.read_text())


@pytest.mark.parametrize("native_present", [False, True])
def test_tiny_lowering_uses_selected_contract(selected, native_present):
    _, contract, native = selected
    if not native_present:
        (native / "target_contract.yaml").unlink()
    # Other tests may already have loaded native provider objects. Selecting a
    # different support tree is a process boundary, not permission to clear the
    # loader's ownership records or replace live plugin objects.
    code = """
import json, sys
from merlin.xdsl_dialects.lowering import pipeline
from merlin.xdsl_dialects.lowering.input_workload import build_matmul_chain
seen = []
original = pipeline.lower_to_contract
def lower(module, selected_contract):
    seen.append(selected_contract)
    return original(module, selected_contract)
pipeline.lower_to_contract = lower
result = pipeline.lower_module(build_matmul_chain(dims=(2, 2, 2)))
assert seen == [json.loads(sys.argv[1])]
for module in result.modules():
    module.verify()
assert pipeline.execute(result)['correct'] is True
"""
    proc = subprocess.run(
        [sys.executable, "-c", code, json.dumps(contract)], capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_missing_explicit_contract_cannot_use_neutral_default(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path / "repo"))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.delenv("MERLIN_TARGET_PATH", raising=False)
    monkeypatch.setenv("MERLIN_TARGET_CONTRACT", str(tmp_path / "missing.yaml"))
    with pytest.raises(pipeline.LoweringError, match="missing.yaml"):
        pipeline.load_curated_contract("toy_npu")
