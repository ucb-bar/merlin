"""Focused tamper tests for the CPU-host frozen execution environment."""
from __future__ import annotations

import subprocess
from pathlib import Path

from merlin.compare import frozen_environment as environment
from merlin.common.paths import repo_root


def _git_repo(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='1'\n")
    (path / "uv.lock").write_text("version = 1\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=path, check=True)
    return path


def test_cpu_host_generated_workspaces_are_excluded_from_source_identity():
    candidate = (repo_root() / "out/build/cpu-host-compiler/probe/workspace/README.md")
    check = subprocess.run(
        ["git", "check-ignore", "-q", str(candidate)], cwd=repo_root(), check=False)
    assert check.returncode == 0


def _capture(tmp_path: Path, monkeypatch):
    aet = _git_repo(tmp_path / "aet")
    chia = _git_repo(tmp_path / "chia")
    source = tmp_path / "controller.py"
    source.write_text("frozen = True\n")
    probe = tmp_path / "probe.c"
    probe.write_text("int main(void) { return 0; }\n")
    telemetry = {"aet_source": str(aet), "chia_source": str(chia),
                 "chia_python": "/fixture/python"}
    agent = {"driver": "codex", "model": "fixed-model", "reasoning_effort": "high",
             "orchestrator": "chia", "billing": "subscription_notional"}
    monkeypatch.setattr(environment, "capture_local_identity",
                        lambda **_kwargs: {"tools": {"clang": "frozen"}})
    monkeypatch.setattr(environment, "_local_completeness_errors", lambda _identity: [])
    monkeypatch.setattr(environment, "capture_k1_identity",
                        lambda _probe: {"device": "k1-fixture", "vlenb": 32, "harts": 8})
    captured = environment.capture_frozen_environment(
        tmp_path / "frozen_environment.json", source_paths={"controller": source},
        agent=agent, telemetry=telemetry, probe_source=probe, include_live_board=True)
    return captured, source, probe, agent, telemetry


def test_frozen_environment_round_trips_exact_local_board_and_source(tmp_path, monkeypatch):
    captured, source, probe, agent, telemetry = _capture(tmp_path, monkeypatch)
    second = environment.capture_frozen_environment(
        tmp_path / "second" / "frozen_environment.json",
        source_paths={"controller": source}, agent=agent, telemetry=telemetry,
        probe_source=probe, include_live_board=True)
    assert second["source_bundle_sha256"] == captured["source_bundle_sha256"]
    assert second["sha256"] == captured["sha256"]
    result = environment.validate_frozen_environment(
        Path(captured["path"]), expected_sha256=captured["sha256"],
        source_paths={"controller": source}, agent=agent, telemetry=telemetry,
        probe_source=probe, check_local=True, check_board=True)
    assert result["ready"] is True
    assert result["errors"] == []
    assert result["evidence"]["local_identity_matches"] is True
    assert result["evidence"]["k1_identity_matches"] is True


def test_frozen_environment_rejects_protocol_source_or_bundle_tampering(tmp_path, monkeypatch):
    captured, source, probe, agent, telemetry = _capture(tmp_path, monkeypatch)
    source.write_text("frozen = False\n")
    changed_source = environment.validate_frozen_environment(
        Path(captured["path"]), expected_sha256=captured["sha256"],
        source_paths={"controller": source}, agent=agent, telemetry=telemetry,
        probe_source=probe, check_local=False, check_board=False)
    assert not changed_source["ready"]
    assert any("protocol source differs" in error for error in changed_source["errors"])

    source.write_text("frozen = True\n")
    bundle = Path(captured["source_bundle_path"])
    with bundle.open("ab") as stream:
        stream.write(b"tampered")
    changed_bundle = environment.validate_frozen_environment(
        Path(captured["path"]), expected_sha256=captured["sha256"],
        source_paths={"controller": source}, agent=agent, telemetry=telemetry,
        probe_source=probe, check_local=False, check_board=False)
    assert not changed_bundle["ready"]
    assert any("source bundle digest" in error for error in changed_bundle["errors"])


def test_frozen_environment_rejects_local_or_k1_identity_drift(tmp_path, monkeypatch):
    captured, source, probe, agent, telemetry = _capture(tmp_path, monkeypatch)
    monkeypatch.setattr(environment, "capture_local_identity",
                        lambda **_kwargs: {"tools": {"clang": "changed"}})
    monkeypatch.setattr(environment, "capture_k1_identity",
                        lambda _probe: {"device": "different-board", "vlenb": 32, "harts": 8})
    result = environment.validate_frozen_environment(
        Path(captured["path"]), expected_sha256=captured["sha256"],
        source_paths={"controller": source}, agent=agent, telemetry=telemetry,
        probe_source=probe, check_local=True, check_board=True)
    assert not result["ready"]
    assert result["evidence"]["local_identity_matches"] is False
    assert result["evidence"]["k1_identity_matches"] is False
    assert any("local tool/config/dependency/source identity" in error
               for error in result["errors"])
    assert any("K1 device/kernel/OS/ISA identity" in error for error in result["errors"])


def test_frozen_environment_rejects_manifest_byte_tampering(tmp_path, monkeypatch):
    captured, source, probe, agent, telemetry = _capture(tmp_path, monkeypatch)
    manifest = Path(captured["path"])
    manifest.chmod(0o644)
    with manifest.open("ab") as stream:
        stream.write(b" ")
    result = environment.validate_frozen_environment(
        manifest, expected_sha256=captured["sha256"], source_paths={"controller": source},
        agent=agent, telemetry=telemetry, probe_source=probe,
        check_local=False, check_board=False)
    assert not result["ready"]
    assert any("manifest digest" in error for error in result["errors"])


def test_live_environment_cannot_omit_a_required_tool_identity():
    errors = environment._local_completeness_errors({
        "tools": {"mlir_opt": {"present": False}},
        "dependencies": {},
        "chia_installed_distributions": {"returncode": 0},
        "source_revisions": {
            name: {"git_commit": "a" * 40} for name in ("merlin", "aet", "chia")},
        "transport": {"k1_ssh_key_identity": {
            "present": True, "public_key_sha256": "b" * 64}},
    })
    assert "required tool is not fully identified: mlir_opt" in errors
