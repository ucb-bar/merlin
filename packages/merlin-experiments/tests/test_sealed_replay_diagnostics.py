"""Exact replay refusal details through the ordinary verifier; no sandbox/model execution."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from merlin_experiments.capture_execution import sealed_m2m as M


def _replay_fixture(tmp_path, monkeypatch, change):
    run = tmp_path / "run"
    source, runtime, output = run / "snapshots/source", run / "snapshots/guest-root", run / "capture"
    for path in (source, runtime, output):
        path.mkdir(parents=True)
    (output / "result.npy").write_bytes(b"same opaque payload")
    (output / "result.npy").chmod(0o664)
    bwrap = tmp_path / "bwrap"
    bwrap.write_bytes(b"fixture binary")
    command = M._command(output)
    materialized, process = {"status": "verified_materialized"}, {"returncode": 0}
    document = {
        "schema": M.SCHEMA_V1,
        "status": "pending_replay",
        "issuer_sha256": M._V1_ISSUER_SHA256,
        "nonce": "0" * 32,
        "plan": {
            "schema": M.SCHEMA_V1,
            "status": "plan_only",
            "command_template_sha256": hashlib.sha256((M._LAUNCH_PREFIX + M._LAUNCH_SUFFIX).encode()).hexdigest(),
        },
        "command": list(command),
        "policy_sha256": M._policy(command, output),
        "scope": M._V1_SCOPE,
        "source": M._snapshot_tree(source),
        "guest_root": M._snapshot_tree(runtime),
        "output": M._snapshot_tree(output),
        "process": process,
        "materialized": materialized,
        "bwrap_sha256": M._file_digest(bwrap),
    }
    (run / "sealed_m2m_pending.json").write_text(json.dumps(document))
    monkeypatch.setattr(M, "_validate_snapshots", lambda *a, **kw: None)
    monkeypatch.setattr(M, "_materialized", lambda *a, **kw: materialized)
    monkeypatch.setattr(M, "_bwrap_binary", lambda *a: bwrap)

    def execute(binary, guest_root, selected_source, replay, selected_command, selected_output, **kwargs):
        assert (binary, guest_root, selected_source, selected_command, selected_output) == (
            bwrap,
            runtime,
            source,
            command,
            output,
        )
        shutil.copy2(output / "result.npy", replay / "result.npy")
        change(replay)
        return process

    monkeypatch.setattr(M, "_execute", execute)
    return run


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda root: (root / "result.npy").chmod(0o600), "mode-only difference.*expected 0o664, actual 0o600"),
        (lambda root: (root / "result.npy").write_bytes(b"changed opaque payload"), "content hash or size differs"),
        (lambda root: (root / "result.npy").unlink(), "missing member 'result.npy'"),
        (lambda root: (root / "extra").write_bytes(b"unexpected"), "unexpected member 'extra'"),
    ],
)
def test_real_replay_comparison_refuses_with_relative_failure_detail(tmp_path, monkeypatch, change, message):
    run = _replay_fixture(tmp_path, monkeypatch, change)
    with pytest.raises(M.SealedM2MError, match=message) as caught:
        M.replay_verify(run)
    assert str(tmp_path) not in str(caught.value)
    assert "opaque payload" not in str(caught.value)


def test_identical_replay_still_checks_complete_real_inventory(tmp_path, monkeypatch):
    run = _replay_fixture(tmp_path, monkeypatch, lambda _: None)
    result = M.replay_verify(run)
    assert result["status"] == "verified_sandbox_replay"
    assert result["phase0_admission"] == "not_granted"


def test_mode_does_not_hide_simultaneous_payload_change(tmp_path, monkeypatch):
    def change(root: Path):
        (root / "result.npy").write_bytes(b"changed opaque payload")
        (root / "result.npy").chmod(0o600)

    run = _replay_fixture(tmp_path, monkeypatch, change)
    with pytest.raises(M.SealedM2MError, match="content hash or size differs") as caught:
        M.replay_verify(run)
    assert "mode-only" not in str(caught.value)
