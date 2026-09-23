"""Model policy transport uses bound inputs; no compiler or process is launched."""

import json
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

from merlin.targetgen import capsule_runner as CR
from merlin.targetgen.sandbox import bwrap as BW


@pytest.fixture
def frozen_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", "")
    repo = tmp_path / "repo"
    repo.mkdir()
    profile = tmp_path / "external-policy.yaml"
    profile.write_text("datapath: {operand_dtype: fp32, subnormal_operand_flush: true}\n")
    descriptor = repo / "target_experiment.yaml"
    descriptor.write_text(yaml.safe_dump({"target": "fixture", "numeric_profile": str(profile)}))
    alternate = repo / "alternate.yaml"
    alternate.write_text("datapath: {operand_dtype: fp32, subnormal_operand_flush: false}\n")
    bundle = {"allowed": [], "host_inputs": [{"path": str(p)} for p in (descriptor, profile, alternate)]}
    ws = tmp_path / "run" / "workspace"
    ws.mkdir(parents=True)
    BW.materialize_bundle_inputs(ws, bundle, repo=repo)
    expected = BW.snapshot_record(ws)
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(descriptor))
    monkeypatch.setenv(CR._MODEL_HOST_SNAPSHOT_ROOT_ENV, str(BW.bundle_snapshot_root(ws)))
    monkeypatch.setenv(CR._MODEL_HOST_SNAPSHOT_REQUIRED_ENV, "1")
    monkeypatch.setenv(CR._MODEL_HOST_SNAPSHOT_RECORD_ENV, json.dumps(expected))
    yield ws, bundle, repo, descriptor, profile, alternate
    BW.remove_bundle_snapshot(ws)


def test_model_policy_uses_frozen_declaration_after_originals_removed(frozen_policy):
    ws, bundle, repo, descriptor, profile, alternate = frozen_policy
    descriptor.write_text(yaml.safe_dump({"target": "fixture", "numeric_profile": str(alternate)}))
    policy, identity = CR._resolve_model_numeric_policy("fixture")
    assert policy["subnormal_operand_flush"] is True
    assert identity["source"] == "frozen-input"
    descriptor.unlink()
    profile.unlink()
    assert CR._resolve_model_numeric_policy("fixture") == (policy, identity)


def test_model_policy_rejects_marker_tamper(frozen_policy):
    ws, *_ = frozen_policy
    marker = BW.bundle_snapshot_root(ws) / "snapshot.json"
    marker.chmod(0o600)
    marker.write_text(marker.read_text() + "\n")
    with pytest.raises(ValueError, match="provenance"):
        CR._resolve_model_numeric_policy("fixture")


def test_public_policy_resolver_returns_frozen_descriptor_not_changed_live_target(frozen_policy, monkeypatch):
    _, _, _, descriptor, profile, _ = frozen_policy
    descriptor.write_text("target: changed-live-target\n")
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", "/missing/ambient-descriptor.yaml")
    policy, identity, experiment = CR.resolve_numeric_policy("fixture", descriptor=descriptor)
    assert experiment.target == "fixture"
    assert experiment.numeric_profile == str(profile)
    assert policy["subnormal_operand_flush"] is True
    assert identity["source"] == "frozen-input"


def test_model_policy_rejects_payload_tamper(frozen_policy):
    ws, bundle, repo, _, profile, _ = frozen_policy
    [frozen] = BW.snapshot_input_paths(ws, bundle, [profile], repo=repo)
    frozen.chmod(0o600)
    frozen.write_text("datapath: {operand_dtype: fp32}\n")
    with pytest.raises(RuntimeError, match="verification"):
        CR._resolve_model_numeric_policy("fixture")


def test_model_policy_requires_bound_context(frozen_policy, monkeypatch):
    monkeypatch.delenv(CR._MODEL_HOST_SNAPSHOT_RECORD_ENV)
    with pytest.raises(ValueError, match="provenance"):
        CR._resolve_model_numeric_policy("fixture")


@pytest.mark.parametrize("policy", [None, {"operand_dtype": "fp32", "subnormal_operand_flush": True}])
def test_parent_transports_explicit_policy_without_rediscovery(monkeypatch, policy):
    def unexpected(*args, **kwargs):
        pytest.fail("explicit policy must not be rediscovered")

    observed = {}
    monkeypatch.setattr(CR, "_resolve_model_numeric_policy", unexpected)
    monkeypatch.setattr(
        CR, "_grade_model_capsule_inline", lambda cap, **kwargs: observed.update(kwargs) or {"status": "pass"}
    )
    result = CR._grade_model_capsule_unlocked(
        {"name": "M"},
        target="fixture",
        timeout=1,
        budget_s=0,
        numeric_policy=policy,
        numeric_policy_identity={"sha256": "test"},
    )
    assert result["status"] == "pass"
    assert observed["numeric_policy"] == policy
    assert observed["numeric_policy_identity"] == {"sha256": "test"}


def test_missing_frozen_descriptor_refuses_before_child(frozen_policy, monkeypatch):
    _, _, repo, _, _, _ = frozen_policy
    other = repo / "not-captured.yaml"
    other.write_text("target: fixture\n")
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(other))
    result = CR._grade_model_capsule_unlocked({"name": "M"}, target="fixture", timeout=1, budget_s=5)
    assert result["status"] == "incomplete"
    assert "not declared" in result["failure"]["detail"]


@pytest.mark.parametrize("policy", [None, {"operand_dtype": "fp32", "subnormal_operand_flush": True}])
def test_budgeted_child_json_roundtrip_preserves_policy(monkeypatch, policy):
    import subprocess

    seen = {}

    class Child:
        returncode = 0

        def __init__(self, argv, **kwargs):
            spec = Path(argv[argv.index("--model-grade") + 1])
            output = Path(argv[argv.index("--model-grade-out") + 1])
            seen.update(json.loads(spec.read_text()))
            assert CR.main(["--model-grade", str(spec), "--model-grade-out", str(output)]) == 0

        def communicate(self, **kwargs):
            return "", None

    def inline(capsule, **kwargs):
        assert kwargs["numeric_policy"] == policy
        assert kwargs["numeric_policy_identity"] == {"sha256": "bound"}
        return {"status": "pass"}

    monkeypatch.setattr(subprocess, "Popen", Child)
    monkeypatch.setattr(CR, "_grade_model_capsule_inline", inline)
    result = CR._grade_model_capsule_unlocked(
        {"name": "M"},
        target="fixture",
        timeout=1,
        budget_s=5,
        numeric_policy=policy,
        numeric_policy_identity={"sha256": "bound"},
    )
    assert result["status"] == "pass"
    assert seen["numeric_policy"] == policy


def test_inline_passes_complete_policy_to_real_compile_boundary(monkeypatch, tmp_path):
    from merlin import compile_cli

    policy = {"operand_dtype": "fp32", "subnormal_operand_flush": True, "atol": 0.001}
    seen = {}

    @contextmanager
    def runtime(*args, **kwargs):
        yield object(), {"test": "synthetic already captured input"}, lambda: None

    def compile_model(*args, **kwargs):
        seen.update(kwargs)
        raise SystemExit("synthetic stop after recording actual compile boundary")

    monkeypatch.setattr(CR, "_model_runtime_bundle", runtime)
    monkeypatch.setattr(CR, "_resolve_model_host_lane", lambda *args: (None, tmp_path, {"dtype_strategy": "fp32"}))
    monkeypatch.setattr(compile_cli, "compile_model", compile_model)
    result = CR._grade_model_capsule_inline(
        {"name": "M", "operation": {"attributes": {"model": "synthetic", "compile_dtype": "fp32"}}},
        target="fixture",
        timeout=1,
        numeric_policy=policy,
    )
    assert seen["numeric_policy"] == policy
    assert seen["package"] == str(tmp_path)
    assert seen["auto_capture"] is False
    assert result["status"] == "incomplete"
