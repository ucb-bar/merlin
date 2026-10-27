"""The support provider's semantic probes read the elaboration its contract selects."""

from __future__ import annotations

import hashlib

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen import target_registry
from merlin.targetgen.rtl import semantic_facts


def _load(*parts):
    return yaml.safe_load((repo_root().joinpath("examples", "gemmini", *parts)).read_text(encoding="utf-8"))


def test_the_probed_elaboration_is_the_contracts_pinned_firrtl():
    probes = _load("support", "contracts", "semantic_probes.yaml")
    source = _load("target", "contracts", "target_contract.yaml")["rtl_source"]
    elaboration = probes["sources"]["elaboration"]
    assert elaboration["ext_root"] == source["ext_root"]
    assert elaboration["sha256"] == source["artifact_sha256"]["fir"]
    config = source["config"]
    assert elaboration["path"].endswith(f"chipyard.harness.TestHarness.{config}.fir")
    assert not elaboration["path"].startswith("/")


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("relocated", [False, True])
def test_semantic_probes_belong_to_selected_provider(tmp_path, monkeypatch, external, relocated):
    target = "probe_ownership_test"
    provider = tmp_path / "references" / target
    contracts = provider / "contracts"
    contracts.mkdir(parents=True)
    # Explicit providers may name a contract outside their contracts directory.
    contract = provider / ("spec/execution.yaml" if external else "contracts/target_contract.yaml")
    contract.parent.mkdir(exist_ok=True)
    contract.write_text(yaml.safe_dump({"name": target}), encoding="utf-8")
    if external:
        (provider / "provider.yaml").write_text(
            yaml.safe_dump({
                "schema": "merlin.provider.v1", "id": "probe-owner", "target": target,
                "role": "support", "contract": "spec/execution.yaml",
            }),
            encoding="utf-8",
        )
    probes = {"schema": "semantic_probes_v1", "target": target, "sources": {}}
    probe_file = contracts / "semantic_probes.yaml"
    probe_file.write_text(yaml.safe_dump(probes), encoding="utf-8")
    selected = tmp_path / "execution-contract" / "selected.yaml"
    selected.parent.mkdir()
    selected.write_bytes(contract.read_bytes())
    # A file beside the selected execution contract must not supply probe data.
    (selected.parent / "semantic_probes.yaml").write_text("schema: wrong_owner\n", encoding="utf-8")
    monkeypatch.setenv("MERLIN_TARGETS_DIR", str(provider.parent))
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(provider) if external else "")
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.delenv("MERLIN_TARGET_CONTRACT", raising=False)
    if relocated:
        monkeypatch.setenv("MERLIN_TARGET_CONTRACT", str(selected))

    resolved = target_registry.resolve(target)
    assert resolved.base == provider
    assert resolved.contract_path == (selected if relocated and not external else contract)
    assert semantic_facts.probes_path(target) == probe_file
    assert semantic_facts.load_probes(target) == probes
    # Exercise the default derivation lookup and provenance without parsing RTL.
    derived = semantic_facts.derive(target, only=())
    assert derived["status"] == semantic_facts.DERIVED
    assert derived["probes"]["sha256"] == hashlib.sha256(probe_file.read_bytes()).hexdigest()
