"""Derive's issue-time attestation mode re-reads every bound byte and refuses any mismatch."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from merlin_experiments.capture_execution import sealed_m2m
from merlin_experiments.capture_execution.sealed_static import _digest, _file_digest
from merlin_experiments.phase0 import attested_evidence, parallel_replay
from merlin_experiments.phase0 import capture_execution_attestation as attestation_module
from merlin_experiments.phase0 import capture_selection as selected
from merlin_experiments.phase0.capture_execution_attestation import AttestationNotVerified, attest_sealed_m2m
from merlin_experiments.phase0.parallel_replay import WORKERS_ENV, Replays

from test_phase0_capture_selection import _fixture


@pytest.fixture
def issued(tmp_path, monkeypatch):
    plan, library, bwrap, run, _, arguments = _fixture(tmp_path, monkeypatch)
    loader = b"def load():\n    pass\n"
    plan["loader_sha256"] = _digest(loader)
    identity = selected.select(**arguments)
    path, sha = Path(identity["path"]), identity["sha256"]
    manifest = selected.load(path, expected_sha256=sha)
    capture = run / "capture"
    capture.mkdir(parents=True)
    run.chmod(0o700)
    model = capture / "model.mlir"
    model.write_bytes(b"module {}\n")
    (capture / "capture_receipt.json").write_bytes(b"{}\n")
    guest_library = run / "snapshots/guest-root" / library.relative_to("/")
    guest_library.parent.mkdir(parents=True)
    shutil.copy2(library, guest_library)
    (run / "snapshots/source/workload").mkdir(parents=True)
    (run / "snapshots/source/workload/loader.py").write_bytes(loader)
    pending = run / "sealed_m2m_pending.json"
    receipt = {
        "schema": sealed_m2m.SCHEMA,
        "status": "pending_replay",
        "capture_selection_sha256": sha,
        "plan": plan,
        "policy_sha256": manifest["sandbox_policy_sha256"],
        "bwrap_sha256": manifest["bwrap"]["sha256"],
        "issuer_sha256": manifest["issuer_source_sha256"],
        "source": {"s": 1},
        "guest_root": {"g": 1},
        "output": {"o": 1},
        "materialized": {"m": 1},
    }
    pending.write_text(json.dumps(receipt))
    # The snapshot machinery is exercised by its own tests; here it only has to agree with the receipt.
    monkeypatch.setattr(sealed_m2m, "_validate_snapshots", lambda *a, **k: None)
    monkeypatch.setattr(
        sealed_m2m,
        "_snapshot_tree",
        lambda root, **_: {"snapshots/source": {"s": 1}, "snapshots/guest-root": {"g": 1}}.get(
            Path(root).relative_to(run).as_posix(), {"o": 1}
        ),
    )
    monkeypatch.setattr(sealed_m2m, "_materialized_v2", lambda *a, **k: {"m": 1})
    monkeypatch.setattr(attestation_module, "verify_capture_receipt", lambda _model: {"status": "verified_materialized"})
    monkeypatch.setattr(
        sealed_m2m,
        "replay_verify",
        lambda *a, **k: {
            "status": "verified_sandbox_replay",
            "sealed_source_closure_replayed": True,
            "receipt_sha256": _file_digest(pending),
        },
    )
    replay = selected.verify(path, expected_sha256=sha, model_path=model)
    document = attest_sealed_m2m(replay, selection_path=path, model_path=model)
    raw = selected._json(document) + b"\n"
    attestation = tmp_path / "attestation.json"
    attestation.write_bytes(raw)
    return {
        "selection": path,
        "sha": sha,
        "model": model,
        "attestation": attestation,
        "attestation_sha": _digest(raw),
        "document": document,
        "replay": replay,
        "pending": pending,
        "receipt": receipt,
        "run": run,
        "loader": run / "snapshots/source/workload/loader.py",
    }


def _check(case, **override):
    values = {**case, **override}
    return attested_evidence.attested_one(
        values["selection"], values["sha"], values["model"], values["attestation"], values["attestation_sha"]
    )


def test_attested_evidence_matches_a_replay_record_plus_the_mode(issued):
    evidence, document, loaded = _check(issued)
    assert document == issued["document"]
    assert loaded == selected.load(issued["selection"], expected_sha256=issued["sha"])
    expected = {
        **issued["replay"],
        "status": "verified_issue_time_attestation",
        "evidence_mode": "issue_time_attestation",
        "attestation_sha256": issued["attestation_sha"],
    }
    assert evidence == expected


def test_tampered_model_byte_is_refused(issued):
    issued["model"].write_bytes(b"module { }\n")
    with pytest.raises(AttestationNotVerified):
        _check(issued)


def test_wrong_attestation_digest_is_refused(issued):
    with pytest.raises(AttestationNotVerified, match="recorded issue-time identity"):
        _check(issued, attestation_sha="0" * 64)
    issued["attestation"].write_bytes(issued["attestation"].read_bytes() + b" ")
    with pytest.raises(AttestationNotVerified, match="recorded issue-time identity"):
        _check(issued)


def test_attestation_for_a_different_selection_is_refused(issued, tmp_path):
    document = json.loads(issued["attestation"].read_bytes())
    document["selection"]["sha256"] = "1" * 64
    other = tmp_path / "other.json"
    raw = json.dumps(document).encode()
    other.write_bytes(raw)
    with pytest.raises(AttestationNotVerified, match="different selection"):
        _check(issued, attestation=other, attestation_sha=_digest(raw))


def test_changed_pending_receipt_is_refused(issued):
    issued["pending"].write_text(json.dumps({**issued["receipt"], "nonce": "changed"}))
    with pytest.raises(AttestationNotVerified):
        _check(issued)
    issued["pending"].write_text(json.dumps({**issued["receipt"], "capture_selection_sha256": "2" * 64}))
    with pytest.raises(ValueError, match="does not bind"):
        _check(issued)


def test_changed_snapshot_loader_or_issuer_is_refused(issued, monkeypatch):
    issued["loader"].write_bytes(b"def load():\n    return 1\n")
    with pytest.raises(ValueError, match="loader"):
        _check(issued)
    issued["loader"].write_bytes(b"def load():\n    pass\n")
    _check(issued)
    monkeypatch.setattr(sealed_m2m, "_snapshot_tree", lambda root, **_: {"changed": 1})
    with pytest.raises(ValueError, match="snapshot or output bytes"):
        _check(issued)


def test_replays_with_an_attestation_do_not_run_the_sandbox(issued, monkeypatch):
    def forbidden(*_a, **_k):
        raise AssertionError("the capture was replayed")

    monkeypatch.setattr(sealed_m2m, "replay_verify", forbidden)
    monkeypatch.setenv(WORKERS_ENV, "0")
    job = (issued["selection"], issued["sha"], issued["model"], issued["attestation"], issued["attestation_sha"])
    with Replays([job]) as replays:
        assert replays[0][0]["evidence_mode"] == "issue_time_attestation"
    # Without an attestation the default is unchanged: the replay runs.
    with Replays([job[:3]]) as replays, pytest.raises(AssertionError, match="replayed"):
        replays[0]
    assert parallel_replay.replay_one is not None
