"""The support provider's semantic probes read the elaboration its contract selects."""

from __future__ import annotations

import yaml

from merlin.common.paths import repo_root


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
