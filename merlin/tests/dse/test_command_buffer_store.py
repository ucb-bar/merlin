"""The expanded command buffer is stored once beside a result, referenced, and digest-checked on read."""

from __future__ import annotations

import hashlib
import json
import os
import shutil

import pytest

from merlin.perf import command_buffer_store as STORE
from merlin.perf.work_volume import command_buffer_evidence


def _buffer(values: int = 4096) -> dict:
    return {
        "abi_version": "0.1",
        "tensors": {
            "X": {"shape": [1, 64], "dtype": "i8", "role": "input"},
            "W": {"shape": [64, values // 64], "dtype": "i8", "role": "weight"},
            "Y": {"shape": [1, values // 64], "dtype": "i32", "role": "output"},
        },
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "X", "rhs": "W_res", "dst": "acc"}},
            {"opcode": "COMMIT", "operands": {"src": "acc", "dst": "Y"}},
        ],
        "canonical_inputs": {"W": {"shape": [64, values // 64], "values": list(range(values))}},
    }


def test_stored_evidence_references_one_canonical_file_and_counts_the_same_work(tmp_path):
    cb = _buffer()
    inline_work, inline = command_buffer_evidence(cb, compiler_provenance="p")
    work, artifact = command_buffer_evidence(cb, compiler_provenance="p", store=str(tmp_path))

    assert work == inline_work
    assert "command_buffer" not in artifact
    assert artifact["artifact_sha256"] == inline["artifact_sha256"]
    reference = artifact[STORE.REFERENCE_KEY]
    stored = list(tmp_path.iterdir())
    assert [path.name for path in stored] == [reference["name"]] == [STORE.stored_name(artifact["artifact_sha256"])]
    payload = stored[0].read_bytes()
    # The file IS the canonical encoding the digest was taken over: one digest, no second identity.
    assert hashlib.sha256(payload).hexdigest() == reference["sha256"] == artifact["artifact_sha256"]
    assert reference["bytes"] == len(payload) and reference["commands"] == 3
    assert STORE.load_command_buffer(artifact) == inline["command_buffer"] == cb
    # The inline summary answers shape questions without the tensor values.
    assert STORE.command_buffer_tensors(artifact) == {
        name: {"dtype": spec["dtype"], "shape": spec["shape"], "role": spec["role"]}
        for name, spec in cb["tensors"].items()
    }
    assert len(json.dumps(artifact)) < 2048 < len(payload)


def test_a_changed_or_missing_stored_buffer_is_refused(tmp_path):
    _, artifact = command_buffer_evidence(_buffer(), compiler_provenance="p", store=str(tmp_path))
    path = tmp_path / artifact[STORE.REFERENCE_KEY]["name"]
    original = path.read_bytes()

    path.write_bytes(original.replace(b'"values":[0,', b'"values":[9,', 1))
    with pytest.raises(STORE.CommandBufferIntegrityError, match="changed"):
        STORE.load_command_buffer(artifact)

    path.write_bytes(original + b" ")
    with pytest.raises(STORE.CommandBufferIntegrityError):
        STORE.load_command_buffer(artifact)

    path.unlink()
    with pytest.raises(STORE.CommandBufferIntegrityError, match="missing"):
        STORE.load_command_buffer(artifact)


def test_a_reference_that_names_another_digest_is_refused(tmp_path):
    _, artifact = command_buffer_evidence(_buffer(), compiler_provenance="p", store=str(tmp_path))
    forged = {**artifact, "artifact_sha256": "0" * 64}
    with pytest.raises(STORE.CommandBufferIntegrityError, match="another digest"):
        STORE.load_command_buffer(forged)
    renamed = {**artifact, STORE.REFERENCE_KEY: {**artifact[STORE.REFERENCE_KEY], "name": "elsewhere.json"}}
    with pytest.raises(STORE.CommandBufferIntegrityError, match="content address"):
        STORE.load_command_buffer(renamed)


def test_a_moved_result_directory_still_resolves_through_its_base(tmp_path):
    run = tmp_path / "run"
    _, artifact = command_buffer_evidence(_buffer(), compiler_provenance="p", store=str(run))
    moved = tmp_path / "moved"
    shutil.move(str(run), str(moved))
    with pytest.raises(STORE.CommandBufferIntegrityError, match="missing"):
        STORE.load_command_buffer(artifact)
    assert STORE.load_command_buffer(artifact, base=moved) == _buffer()


def test_storing_the_same_bytes_again_does_not_rewrite_the_file(tmp_path):
    _, artifact = command_buffer_evidence(_buffer(), compiler_provenance="p", store=str(tmp_path))
    path = tmp_path / artifact[STORE.REFERENCE_KEY]["name"]
    os.utime(path, ns=(1, 1))
    command_buffer_evidence(_buffer(), compiler_provenance="q", store=str(tmp_path))
    assert path.stat().st_mtime_ns == 1


def test_legacy_embedded_buffers_and_absent_buffers_still_read():
    cb = _buffer(128)
    _, inline = command_buffer_evidence(cb, compiler_provenance="p")
    assert STORE.has_command_buffer(inline)
    assert STORE.load_command_buffer(inline) == cb
    assert STORE.command_buffer_tensors(inline) == cb["tensors"]
    _, absent = command_buffer_evidence(None, compiler_provenance="p")
    assert not STORE.has_command_buffer(absent)
    assert STORE.load_command_buffer(absent) is None
    assert STORE.command_buffer_tensors(absent) == {}


def test_stored_names_are_recognised_only_as_content_addresses():
    sha = "a" * 64
    assert STORE.is_stored_name(STORE.stored_name(sha))
    assert not STORE.is_stored_name("expanded_command_buffer.json")
    assert not STORE.is_stored_name(STORE.stored_name("g" * 64))
    assert not STORE.is_stored_name("command_buffer.json")


def test_receipt_bridge_reads_a_stored_buffer_and_refuses_a_changed_one(tmp_path):
    from merlin.perf import receipt_bridge as RB

    cb = _buffer()
    _, artifact = command_buffer_evidence(cb, compiler_provenance="compiler", store=str(tmp_path))
    identity = {"program": {"sha256": artifact["artifact_sha256"]}}
    issues: list = []
    command, provenance = RB._command_buffer({"command_buffer_artifact": artifact}, {}, identity, "row", issues)
    assert (command, provenance, issues) == (cb, "compiler", [])

    path = tmp_path / artifact[STORE.REFERENCE_KEY]["name"]
    path.write_bytes(path.read_bytes().replace(b'"values":[0,', b'"values":[7,', 1))
    command, provenance = RB._command_buffer({"command_buffer_artifact": artifact}, {}, identity, "row", issues)
    assert (command, provenance) == ({}, "")
    assert [issue["code"] for issue in issues] == ["COMMAND_BUFFER_RECEIPT_MISMATCH"]
