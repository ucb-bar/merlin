"""Driver-private completion custody for replay through the existing closed recorder.

Public channel responses are never recovery authority. This journal lives beside
private source snapshots and retains the verified launch identity, not a digest
recomputed from the author's subsequently edited source.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from merlin_experiments.phase1.feedback import promotion, snapshots


def _store(ws: Path) -> Path:
    anchor = snapshots.promotion_snapshot_path(ws, "0" * 32, create_store=True)
    assert anchor is not None
    root = anchor.parent / ".completed"
    if root.is_symlink() or root.resolve().is_relative_to(ws.resolve()):
        raise ValueError("completion store is not outside the candidate workspace")
    root.mkdir(mode=0o700, exist_ok=True)
    return root


def _encoded(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def persist(ws: Path, job_id: str, verdict: dict, *, identity: str, source_digest: str,
            source_identity_verified: bool, cert_tier: str, instrument: str | None) -> Path:
    """Persist only an actually completed, source-verified promoted job."""
    if not source_identity_verified or not isinstance(source_digest, str) or not source_digest:
        raise ValueError("completion has no verified launch source")
    if identity != f"submission:{source_digest}" and not promotion._valid_execution(identity):
        raise ValueError("completion identity does not bind its verified source or execution")
    record = {
        "schema": "private_promotion_completion_v1", "workspace": str(ws.resolve()),
        "job_id": job_id, "identity": identity, "source_digest": source_digest,
        "source_identity_verified": True, "cert_tier": cert_tier, "instrument": instrument,
        "verdict": verdict, "verdict_sha256": hashlib.sha256(_encoded(verdict)).hexdigest(),
    }
    path = _store(ws) / (hashlib.sha256(job_id.encode()).hexdigest() + ".json")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(_encoded(record))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def recover(ws: Path, *, cert_tier: str, instrument: str | None, log=None,
            recovered_results: dict | None = None) -> list[str]:
    """Replay protected completions; duplicate/unknown identities remain recorder no-ops."""
    recorded = []
    for path in sorted(_store(ws).glob("*.json")):
        try:
            if path.is_symlink() or not path.is_file():
                continue
            record = json.loads(path.read_bytes())
            if (record.get("schema") != "private_promotion_completion_v1"
                    or record.get("workspace") != str(ws.resolve())
                    or record.get("cert_tier") != cert_tier or record.get("instrument") != instrument
                    or record.get("source_identity_verified") is not True):
                continue
            identity, digest = record.get("identity"), record.get("source_digest")
            job_id = record.get("job_id")
            if (not isinstance(job_id, str) or not job_id
                    or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for c in job_id)
                    or path.name != hashlib.sha256(job_id.encode()).hexdigest() + ".json"):
                continue
            if not isinstance(digest, str) or not digest:
                continue
            if identity != f"submission:{digest}" and not promotion._valid_execution(identity):
                continue
            verdict = record["verdict"]
            if (not isinstance(verdict, dict)
                    or hashlib.sha256(_encoded(verdict)).hexdigest() != record.get("verdict_sha256")):
                continue
            recorded.extend(promotion.record_cert(ws, verdict, cert_tier, log, identity=identity,
                                                 source_identity_verified=True, request=job_id))
            if recovered_results is not None:
                recovered_results[job_id] = verdict
        except (OSError, ValueError, KeyError, TypeError):
            # A corrupt/unbound journal member authorizes nothing and cannot block ordinary work.
            continue
    return recorded
