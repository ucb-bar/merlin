"""Whole-source pins stream; parsed sidecars retain the existing JSON bound."""

import hashlib

import pytest

from merlin.common import strict_json
from merlin.llvmlower import device_preparation as preparation


def test_large_original_source_is_pinned_and_suffix_change_is_refused(tmp_path):
    path = tmp_path / "original.mlir"
    payload = b"a" * ((16 << 20) + 17)
    path.write_bytes(payload)
    frozen = preparation._DeviceOffloadInvocation(None, None)
    preparation.retain_device_offload_source(frozen, path)
    assert frozen.sources == ((path, hashlib.sha256(payload).hexdigest()),)
    preparation.require_device_offload_artifacts(frozen)
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"b")
    with pytest.raises(ValueError, match="realized source or sidecar changed"):
        preparation.require_device_offload_artifacts(frozen)


def test_sidecar_payload_uses_existing_strict_json_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(strict_json, "MAX_RECORD_BYTES", 1024)
    path = tmp_path / "device_signatures.json"
    payload = b'{"source": "' + b"a" * 1010 + b'"}'
    assert len(payload) == 1024
    path.write_bytes(payload)
    assert strict_json.loads(preparation._artifact_payload(path))["source"] == "a" * 1010
    path.write_bytes(payload + b" ")
    with pytest.raises(ValueError, match="compilation byte budget"):
        preparation._artifact_payload(path)
    # Hash-only source custody does not allocate a bounded JSON payload.
    assert preparation._artifact_digest(path) == hashlib.sha256(payload + b" ").hexdigest()


def test_streamed_artifact_pins_preserve_direct_file_custody(tmp_path):
    source = tmp_path / "source.mlir"
    source.write_bytes(b"original source")
    alias = tmp_path / "alias.mlir"
    alias.symlink_to(source)
    frozen = preparation._DeviceOffloadInvocation(None, None)
    with pytest.raises(ValueError, match="original direct regular file"):
        preparation.retain_device_offload_source(frozen, alias)
    preparation.retain_device_offload_source(frozen, source)
    replacement = tmp_path / "replacement.mlir"
    replacement.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(replacement)
    with pytest.raises(ValueError, match="original direct regular file"):
        preparation.require_device_offload_artifacts(frozen)
