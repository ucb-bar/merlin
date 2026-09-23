"""Authored numerical inputs retain bytes and ownership across freezing."""

from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.corpus.numeric_policy import load_declared_numeric_policy, numeric_profile_path

from merlin.targetgen.sandbox import bwrap as BW


def test_complete_declared_policy_without_sibling_discovery(tmp_path):
    profile = tmp_path / "profile.yaml"
    policy = {"operand_dtype": "fp32", "accum_dtype": "fp32", "subnormal_operand_flush": True, "atol": 0.001}
    profile.write_text(yaml.safe_dump({"datapath": policy}))
    (tmp_path / "profile.hidden.yaml").write_text("this must never be parsed: [")
    experiment = SimpleNamespace(numeric_profile="profile.yaml")
    observed, identity = load_declared_numeric_policy(experiment, repo=tmp_path)
    assert observed == policy
    assert identity["scope"] == "declared-numerical-assumptions"
    assert identity["hardware_verified"] is False
    assert identity["size_bytes"] == profile.stat().st_size
    profile.write_text(yaml.safe_dump({"datapath": {**policy, "subnormal_operand_flush": False}}))
    changed, changed_identity = load_declared_numeric_policy(experiment, repo=tmp_path)
    assert changed["subnormal_operand_flush"] is False
    assert changed_identity["sha256"] != identity["sha256"]


@pytest.mark.parametrize("external", [False, True])
def test_frozen_policy_uses_declared_snapshot_even_after_live_deletion(tmp_path, monkeypatch, external):
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", "")
    repo = tmp_path / "repo"
    repo.mkdir()
    profile = (tmp_path if external else repo) / "profile.yaml"
    profile.write_text("datapath:\n  subnormal_operand_flush: true\n  operand_dtype: fp32\n")
    declaration = str(profile) if external else "profile.yaml"
    experiment = SimpleNamespace(numeric_profile=declaration)
    bundle = {"allowed": [], "host_inputs": [{"path": declaration}]}
    ws = tmp_path / "run" / "workspace"
    ws.mkdir(parents=True)
    try:
        BW.materialize_bundle_inputs(ws, bundle, repo=repo)
        expected = BW.snapshot_record(ws)
        profile.unlink()
        BW.verify_snapshot_binding(ws, bundle, expected, repo=repo)
        [frozen] = BW.snapshot_input_paths(ws, bundle, [numeric_profile_path(declaration, repo=repo)], repo=repo)
        policy, identity = load_declared_numeric_policy(experiment, repo=repo, frozen_profile=frozen)
        assert policy["subnormal_operand_flush"] is True
        assert identity["source"] == "frozen-input"
        with pytest.raises(ValueError, match="missing"):
            load_declared_numeric_policy(experiment, repo=repo)
        frozen.chmod(0o600)
        frozen.write_text("datapath:\n  subnormal_operand_flush: false\n")
        with pytest.raises(RuntimeError, match="content verification"):
            BW.snapshot_input_paths(ws, bundle, [profile], repo=repo)
    finally:
        BW.remove_bundle_snapshot(ws)


@pytest.mark.parametrize("value", ["", " ", "../escape.yaml", 2, False, [], {}])
def test_invalid_declaration_refuses(value, tmp_path):
    with pytest.raises(ValueError, match="numeric_profile"):
        load_declared_numeric_policy(SimpleNamespace(numeric_profile=value), repo=tmp_path)


@pytest.mark.parametrize("text", ["[]", "datapath: []", "datapath: {}", "other: value"])
def test_malformed_declared_profile_refuses(text, tmp_path):
    (tmp_path / "profile.yaml").write_text(text)
    with pytest.raises(ValueError, match="numeric profile"):
        load_declared_numeric_policy(SimpleNamespace(numeric_profile="profile.yaml"), repo=tmp_path)


def test_absent_declaration_never_discovers_profile(tmp_path):
    (tmp_path / "profile.yaml").write_text("datapath: {operand_dtype: fp32}")
    assert load_declared_numeric_policy(SimpleNamespace(), repo=tmp_path) == (None, None)
    with pytest.raises(ValueError, match="explicit declaration"):
        load_declared_numeric_policy(SimpleNamespace(), repo=tmp_path, frozen_profile=tmp_path / "profile.yaml")


@pytest.mark.parametrize("target", ["gemmini", "atlas", "mx_gemmini", "radiance", "saturn_opu", "saturn_opu_rvv"])
def test_existing_experiments_load_their_complete_authored_policy(target):
    from merlin.common.paths import repo_root
    from merlin.targetgen.corpus_spec import profile_datapath
    from merlin.targetgen.target_experiment import load_target_experiment

    repo = repo_root()
    descriptor = repo / "merlin/experiments/capsule_bench/targets" / target / "target_experiment.yaml"
    experiment = load_target_experiment(descriptor)
    policy, identity = load_declared_numeric_policy(experiment, repo=repo)
    authored = yaml.safe_load((repo / experiment.numeric_profile).read_bytes())
    assert policy == profile_datapath(authored, numeric_only=True)
    assert identity["hardware_verified"] is False
    if target == "atlas":
        assert policy["subnormal_operand_flush"] is True


@pytest.mark.parametrize("value", ["", False, [], "../recipe.yaml"])
def test_descriptor_rejects_invalid_numeric_reference(tmp_path, value):
    from merlin.targetgen.target_experiment import load_target_experiment

    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text(yaml.safe_dump({"target": "synthetic", "numeric_profile": value}))
    with pytest.raises(ValueError, match="numeric_profile"):
        load_target_experiment(descriptor)
