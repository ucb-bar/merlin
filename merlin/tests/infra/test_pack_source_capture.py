"""A saved TorchAO source archive contains only published, receipted bytes."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path
from types import ModuleType

import pytest


def _packer() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "examples/kodiak/pack_source_capture.py"
    spec = importlib.util.spec_from_file_location("pack_source_capture_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "saved_capture"
    source.mkdir()
    artifacts = {}
    for name, data in (
        ("model.mlir", b"module {}\n"),
        ("weights.safetensors", b"signed i8 weights"),
        ("weights.safetensors.manifest.json", b"{}\n"),
    ):
        (source / name).write_bytes(data)
        artifacts[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (source / "ambient-debug.json").write_text("do not ship me")
    receipt = {"schema": "m2m.capture-receipt.v1", "artifacts": artifacts}
    raw = (json.dumps(receipt, sort_keys=True) + "\n").encode()
    (source / "capture_receipt.json").write_bytes(raw)
    return source, hashlib.sha256(raw).hexdigest()


def test_archive_has_exact_receipt_members_and_deterministic_bytes(tmp_path, monkeypatch):
    packer = _packer()
    source, receipt_sha = _fixture(tmp_path)
    monkeypatch.setattr(packer, "verify_capture_receipt", lambda _path: {"status": "verified_materialized"})
    first, second = tmp_path / "first.tar.gz", tmp_path / "second.tar.gz"
    reports = [packer.pack_capture(source, output, receipt_sha, min_free_gib=0) for output in (first, second)]
    assert first.read_bytes() == second.read_bytes()
    assert reports[0]["archive_sha256"] == reports[1]["archive_sha256"]
    with tarfile.open(first, "r:gz") as archive:
        members = archive.getmembers()
        assert [m.name for m in members] == [f"saved_capture/{n}" for n in reports[0]["members"]]
        assert all(m.isfile() and m.uid == 0 and m.gid == 0 and m.mtime == 0 for m in members)
        assert not any("ambient-debug" in m.name for m in members)


def test_published_receipt_hash_and_space_guard_prevent_output(tmp_path, monkeypatch):
    packer = _packer()
    source, receipt_sha = _fixture(tmp_path)
    output = tmp_path / "refused.tar.gz"
    with pytest.raises(ValueError, match="published SHA256"):
        packer.pack_capture(source, output, "0" * 64, min_free_gib=0)
    assert not output.exists()
    monkeypatch.setattr(packer, "verify_capture_receipt", lambda _path: {"status": "verified_materialized"})
    with pytest.raises(ValueError, match="scratch margin"):
        packer.pack_capture(source, output, receipt_sha, min_free_gib=10**12)
    assert not output.exists()


def test_archive_verifier_rejects_unreceipted_member(tmp_path, monkeypatch):
    packer = _packer()
    source, receipt_sha = _fixture(tmp_path)
    monkeypatch.setattr(packer, "verify_capture_receipt", lambda _path: {"status": "verified_materialized"})
    output = tmp_path / "forged.tar.gz"
    with tarfile.open(output, "w:gz") as archive:
        payload = b"extra"
        info = tarfile.TarInfo("saved_capture/extra.bin")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    receipt = json.loads((source / "capture_receipt.json").read_text())
    with pytest.raises(ValueError, match="unreceipted"):
        packer.verify_archive(output, source.name, receipt, receipt_sha)
