"""Large input hashing keeps the original digest without materializing files in RAM."""

import hashlib
from pathlib import Path

import pytest
from merlin_experiments.phase2.contracts import StageGateError, exact_tree_record


def test_tree_digest_streams_with_original_identity(tmp_path, monkeypatch):
    root = tmp_path / "sources"
    root.mkdir()
    payload = b"large-data" * 250_000
    (root / "weights").write_bytes(payload)
    expected = hashlib.sha256(b"weights\0" + payload + b"\0").hexdigest()

    def refuse_full_read(self):
        raise AssertionError("tree hashing must not allocate a whole file")

    monkeypatch.setattr(Path, "read_bytes", refuse_full_read)
    assert exact_tree_record(root) == {"sha256": expected, "n_files": 1, "n_bytes": len(payload)}


def test_tree_digest_still_rejects_linked_sources(tmp_path):
    (tmp_path / "source").write_text("value")
    (tmp_path / "alias").symlink_to(tmp_path / "source")
    with pytest.raises(StageGateError, match="symlink"):
        exact_tree_record(tmp_path)
