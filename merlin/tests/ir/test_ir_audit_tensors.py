"""Inspection tensor sidecars preserve opaque dense bytes, not numeric conversions."""

import hashlib
import json

import pytest

from merlin.common.ir_audit import IrAudit, read_tensor_payload


@pytest.fixture(autouse=True)
def no_processes_or_listeners(monkeypatch):
    import socket
    import subprocess

    def refused(*args, **kwargs):
        pytest.fail("tensor audit tests must not launch processes or listeners")

    monkeypatch.setattr(subprocess, "Popen", refused)
    monkeypatch.setattr(socket.socket, "bind", refused)


@pytest.mark.parametrize(
    "damage", ["missing", "bytes", "file-link", "directory-link", "root-link", "traversal", "size", "shape", "format"]
)
def test_reopened_tensor_reader_refuses_invalid_payload(tmp_path, damage):
    with IrAudit(tmp_path, enabled="both", producer="fixture", source=__file__) as audit:
        descriptor = audit.tensor(b"1234", element_type="i8", shape=(4,))
    assert read_tensor_payload(audit.directory, descriptor) == b"1234"
    blob = audit.directory / descriptor["file"]
    if damage == "missing":
        blob.unlink()
    elif damage == "bytes":
        blob.write_bytes(b"4321")
    elif damage == "file-link":
        replacement = tmp_path / "replacement"
        blob.rename(replacement)
        blob.symlink_to(replacement)
    elif damage == "directory-link":
        replacement = tmp_path / "replacement"
        blob.parent.rename(replacement)
        blob.parent.symlink_to(replacement, target_is_directory=True)
    elif damage == "root-link":
        replacement = tmp_path / "replacement"
        audit.directory.rename(replacement)
        audit.directory.symlink_to(replacement, target_is_directory=True)
    elif damage == "traversal":
        descriptor["file"] = "../" + descriptor["file"]
    elif damage == "size":
        descriptor["bytes"] += 1
    elif damage == "shape":
        descriptor["shape"] = [True]
    else:
        descriptor["format"] = "unknown"
    with pytest.raises(ValueError):
        read_tensor_payload(audit.directory, descriptor)


@pytest.mark.parametrize("mode", ["compact", "both"])
def test_tensor_preserves_nan_payloads_signed_zero_and_infinity_bits(tmp_path, mode):
    # Distinct IEEE-754 NaN payloads, negative zero, and infinity: never decoded/repacked.
    payload = bytes.fromhex("0100c07f 0200c07f 00000080 0000807f")
    digest = hashlib.sha256(payload).hexdigest()
    with IrAudit(tmp_path, enabled=mode, producer="fixture", source=__file__) as audit:
        tensor = audit.tensor(payload, element_type="f32", shape=(4,))
        assert tensor == {
            "file": f"tensors/{digest}.bin",
            "sha256": digest,
            "bytes": len(payload),
            "element_type": "f32",
            "shape": [4],
            "format": "xdsl-dense-bytes",
        }
        assert (audit.directory / tensor["file"]).read_bytes() == payload
        audit.stage("input", "exact IR", inspection="elided IR", inspection_tensors=[tensor])
    record = json.loads((audit.directory / "index.json").read_text())
    assert record["outcome"] == "completed"
    assert record["stages"][0]["inspection"]["tensors"] == [tensor]


def test_repeated_payload_deduplicates_without_collapsing_type_or_shape(tmp_path):
    payload = bytes(range(16))
    with IrAudit(tmp_path, enabled="both", producer="fixture", source=__file__) as audit:
        first = audit.tensor(payload, element_type="i32", shape=(4,))
        blob = audit.directory / first["file"]
        original = blob.stat()
        repeated = audit.tensor(payload, element_type="i32", shape=(4,))
        matrix = audit.tensor(payload, element_type="i32", shape=(2, 2))
        floating = audit.tensor(payload, element_type="f32", shape=(4,))
        assert repeated == first
        assert matrix["file"] == floating["file"] == first["file"]
        assert matrix["shape"] == [2, 2] and first["shape"] == [4]
        assert floating["element_type"] == "f32" and first["element_type"] == "i32"
        assert list((audit.directory / "tensors").iterdir()) == [blob]
        assert blob.stat().st_ino == original.st_ino
        assert blob.stat().st_mtime_ns == original.st_mtime_ns
        audit.stage("input", "exact", inspection="compact", inspection_tensors=[first, matrix, floating])


def test_mutated_existing_blob_is_refused_not_overwritten(tmp_path):
    payload = b"\x01\x02\x03\x04"
    with pytest.raises(ValueError):
        with IrAudit(tmp_path, enabled="both", producer="fixture", source=__file__) as audit:
            tensor = audit.tensor(payload, element_type="i8", shape=(4,))
            blob = audit.directory / tensor["file"]
            blob.write_bytes(b"changed")
            audit.tensor(payload, element_type="i8", shape=(4,))
    assert blob.read_bytes() == b"changed"


@pytest.mark.parametrize("mode", [False, True, "exact"])
def test_tensor_refused_without_compact_inspection_mode(tmp_path, mode):
    with IrAudit(tmp_path, enabled=mode, producer="fixture", source=__file__) as audit:
        with pytest.raises(ValueError):
            audit.tensor(b"\x00", element_type="i8", shape=(1,))
    assert not list(tmp_path.rglob("*.bin"))


def test_tensor_refused_when_no_audit_workdir():
    with IrAudit(None, enabled="both", producer="fixture", source=__file__) as audit:
        with pytest.raises(ValueError):
            audit.tensor(b"\x00", element_type="i8", shape=(1,))


def test_stage_tensor_metadata_requires_an_inspection_view(tmp_path):
    with IrAudit(tmp_path, enabled="both", producer="fixture", source=__file__) as audit:
        tensor = audit.tensor(b"\x00", element_type="i8", shape=(1,))
        with pytest.raises(ValueError, match="inspection view"):
            audit.stage("no-view", "exact", inspection_tensors=[tensor])
        assert audit.record["stages"] == []
        audit.stage("with-view", "exact", inspection="compact", inspection_tensors=[tensor])
    record = json.loads((audit.directory / "index.json").read_text())
    (second,) = record["stages"]
    assert second["inspection"]["tensors"] == [tensor]
    assert second["inspection"]["parent_sha256"] == hashlib.sha256(b"exact").hexdigest()
    assert second["inspection"]["executable"] is False


def test_exact_mode_refuses_inspection_tensor_metadata(tmp_path):
    with IrAudit(tmp_path, enabled="exact", producer="fixture", source=__file__) as audit:
        with pytest.raises(ValueError, match="inspection view"):
            audit.stage("input", "exact", inspection="ignored", inspection_tensors=[{"ignored": True}])
    record = json.loads((audit.directory / "index.json").read_text())
    assert record["stages"] == []
    assert not (audit.directory / "tensors").exists()


def test_stage_rejects_forged_tensor_descriptor(tmp_path):
    with IrAudit(tmp_path, enabled="both", producer="fixture", source=__file__) as audit:
        tensor = audit.tensor(b"\x00", element_type="i8", shape=(1,))
        forged = {**tensor, "shape": [999]}
        with pytest.raises(ValueError):
            audit.stage("input", "exact", inspection="compact", inspection_tensors=[forged])
        assert audit.record["stages"] == []


def test_payload_mutation_after_stage_refuses_completion(tmp_path):
    with pytest.raises(ValueError):
        with IrAudit(tmp_path, enabled="both", producer="fixture", source=__file__) as audit:
            tensor = audit.tensor(b"\x00", element_type="i8", shape=(1,))
            audit.stage("input", "exact", inspection="compact", inspection_tensors=[tensor])
            (audit.directory / tensor["file"]).write_bytes(b"changed")
    record = json.loads((audit.directory / "index.json").read_text())
    assert record["outcome"] == "failed"
    assert record["stages"][0]["inspection"]["tensors"] == [tensor]


def test_original_exception_survives_tensor_mutation_on_exit(tmp_path):
    original = RuntimeError("original lowering failure")
    with pytest.raises(RuntimeError) as caught:
        with IrAudit(tmp_path, enabled="both", producer="fixture", source=__file__) as audit:
            tensor = audit.tensor(b"\x00", element_type="i8", shape=(1,))
            audit.stage("input", "exact", inspection="compact", inspection_tensors=[tensor])
            (audit.directory / tensor["file"]).write_bytes(b"changed")
            raise original
    assert caught.value is original


@pytest.mark.parametrize("mode,view", [("exact", False), ("both", False), ("both", True), ("compact", True)])
@pytest.mark.parametrize("damage", ["mutate", "delete", "symlink"])
def test_stage_payload_damage_refuses_completion(tmp_path, mode, view, damage):
    with pytest.raises(ValueError, match="stage payload changed"):
        with IrAudit(tmp_path, enabled=mode, producer="fixture", source=__file__) as audit:
            audit.stage("input", "exact IR", inspection="compact IR")
            stage = audit.record["stages"][0]
            entry = stage["inspection"] if view else stage
            path = audit.directory / entry["file"]
            if damage == "mutate":
                path.write_bytes(b"changed")
            elif damage == "delete":
                path.unlink()
            else:
                original = path.read_bytes()
                replacement = tmp_path / "replacement"
                replacement.write_bytes(original)
                path.unlink()
                path.symlink_to(replacement)
    record = json.loads((audit.directory / "index.json").read_text())
    assert record["outcome"] == "failed"
    assert record["failure_type"] == "StagePayloadChanged"


def test_original_exception_survives_stage_deletion_on_exit(tmp_path):
    original = RuntimeError("original lowering failure")
    with pytest.raises(RuntimeError) as caught:
        with IrAudit(tmp_path, enabled="exact", producer="fixture", source=__file__) as audit:
            audit.stage("input", "exact IR")
            (audit.directory / audit.record["stages"][0]["file"]).unlink()
            raise original
    assert caught.value is original
    record = json.loads((audit.directory / "index.json").read_text())
    assert record["outcome"] == "failed"
    assert record["failure_type"] == "RuntimeError"
