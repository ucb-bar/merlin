"""Unverified execution diagnostics, deliberately separate from capture admission."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

# Replay-only predecessor: successful guest policy and byte comparisons are unchanged.
PRE_RETENTION_ISSUER_SHA256 = "1c0b4e2bf38893f63c08f7a39e308037e7e80f01615b89d19a4e504a75dc5a20"


def _write(path: Path, raw: bytes) -> None:
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        stream.write(raw)


def process(failure_dir: Path, returncode: int, stdout: bytes, stderr: bytes) -> None:
    """Retain actual streams before any materialization or publication checks."""
    if returncode:
        directory = failure_dir
        directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    else:
        directory = failure_dir.with_name(failure_dir.name.removesuffix("-failure") + "-execution")
        directory = Path(tempfile.mkdtemp(prefix=directory.name + "-", dir=directory.parent))
    _write(directory / "stdout.bin", stdout)
    _write(directory / "stderr.bin", stderr)
    record = {
        "schema": "merlin.sealed_capture_process_diagnostic.v1",
        "status": "unverified",
        "phase0_admission": "not_granted",
        "returncode": returncode,
        "stdout_bytes": len(stdout),
        "stderr_bytes": len(stderr),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
    }
    _write(directory / "process.json", (json.dumps(record, sort_keys=True) + "\n").encode())


@contextmanager
def publication(run: Path, *, replay: Path | None, plan: dict, selection_sha256: str | None):
    """Delete successful replay scratch; keep failed output without minting a pending receipt.

    This record is not consumable by attest/replay_verify and supplies no recovery API.
    It only preserves observations that would otherwise disappear on a publication error.
    """
    try:
        yield
    except BaseException as exc:
        record = {
            "schema": "merlin.sealed_capture_failure_diagnostic.v1",
            "status": "unverified",
            "phase0_admission": "not_granted",
            "source_closure_verified": False,
            "recovery_supported": False,
            "selection_sha256": selection_sha256,
            "plan_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
            "capture": "capture",
            "replay": None if replay is None else replay.relative_to(run).as_posix(),
            "error": {"type": type(exc).__name__, "detail": str(exc)[:4096]},
        }
        try:
            _write(run / "sealed_capture_unverified_failure.json", (json.dumps(record, sort_keys=True) + "\n").encode())
        except OSError as diagnostic_error:
            exc.add_note(f"unverified failure record could not be retained: {type(diagnostic_error).__name__}")
        raise
    else:
        if replay is not None:
            shutil.rmtree(replay, ignore_errors=True)
