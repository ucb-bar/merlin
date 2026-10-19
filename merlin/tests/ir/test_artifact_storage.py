"""Fresh producer bytes remain the authority for optional artifact sharing."""

import os

import pytest

from merlin.llvmlower.artifact_storage import share_identical_artifact


def test_complete_equal_artifacts_share_inode_and_preserve_every_byte(tmp_path):
    previous, fresh = tmp_path / "previous", tmp_path / "fresh"
    raw = bytes(range(256)) * 257
    previous.write_bytes(raw)
    fresh.write_bytes(raw)
    report = share_identical_artifact(previous, fresh)
    assert previous.read_bytes() == fresh.read_bytes() == raw and os.path.samefile(previous, fresh)
    assert report["bytes"] == len(raw) and report["shared_inode"] and not report["already_shared"]
    assert share_identical_artifact(previous, fresh)["already_shared"]
    assert not list(tmp_path.glob(".artifact-share-*"))


@pytest.mark.parametrize("prior", [b"same extXnt", b"short"])
def test_changed_full_content_or_extent_refuses_without_replacing_fresh(tmp_path, prior):
    previous, fresh = tmp_path / "previous", tmp_path / "fresh"
    previous.write_bytes(prior)
    fresh.write_bytes(b"same extent")
    identity = fresh.stat().st_ino
    with pytest.raises(ValueError, match="complete fresh bytes"):
        share_identical_artifact(previous, fresh)
    assert fresh.read_bytes() == b"same extent" and fresh.stat().st_ino == identity


@pytest.mark.parametrize("side", ["previous", "fresh"])
def test_symbolic_paths_are_not_immutable_artifact_inputs(tmp_path, side):
    real = tmp_path / "real"
    real.write_bytes(b"bytes")
    previous, fresh = tmp_path / "previous", tmp_path / "fresh"
    previous.write_bytes(b"bytes")
    fresh.write_bytes(b"bytes")
    selected = previous if side == "previous" else fresh
    selected.unlink()
    selected.symlink_to(real)
    with pytest.raises(ValueError, match="regular immutable"):
        share_identical_artifact(previous, fresh)
    assert real.read_bytes() == b"bytes"
