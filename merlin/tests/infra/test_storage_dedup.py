"""Collapsing duplicate bytes must never change what a name reads, and must skip mode-verified trees.

The content store gets the saving when a tree is frozen, which does nothing for the trees written
before it existed -- and those are most of the root: 65.9 GiB of out/ was the same bytes under
several names when this was written. Re-pointing each name at one store object reclaims that without
removing anything, but it also makes the file read-only and makes its MODE shared, which is exactly
the property a mode-verified freeze depends on. Both halves are pinned here.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from merlin.common import content_store as CS
from merlin.common import storage_cli as SC


@pytest.fixture
def rooted(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(tmp_path / "out" / "artifacts" / "cache" / CS.NAMESPACE))
    (tmp_path / "out" / "artifacts" / "cache").mkdir(parents=True)
    declared = dict(SC.contract(), scan_roots=[])
    monkeypatch.setattr(SC, "contract", lambda: declared)
    return tmp_path / "out"


def _write(path: Path, body: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


def _inode(path: Path) -> int:
    return path.stat(follow_symlinks=False).st_ino


def test_two_names_for_one_content_end_on_one_inode(rooted):
    body = b"w" * 4096
    first = _write(rooted / "artifacts" / "delivery" / "a" / "bundle.zip", body)
    second = _write(rooted / "artifacts" / "delivery" / "b" / "bundle.zip", body)
    assert _inode(first) != _inode(second)

    assert SC.main(["dedup", "--min-bytes", "1024", "--apply"]) == 0

    assert first.read_bytes() == body, "a name stopped resolving to its own bytes"
    assert second.read_bytes() == body
    assert _inode(first) == _inode(second), "the duplicate was not collapsed"


def test_same_size_different_content_is_left_alone(rooted):
    """The size prefilter is an optimisation, not the test. Acting on it would corrupt a result."""
    first = _write(rooted / "artifacts" / "delivery" / "a" / "x.bin", b"a" * 4096)
    second = _write(rooted / "artifacts" / "delivery" / "b" / "x.bin", b"b" * 4096)

    SC.main(["dedup", "--min-bytes", "1024", "--apply"])

    assert first.read_bytes() == b"a" * 4096
    assert second.read_bytes() == b"b" * 4096
    assert _inode(first) != _inode(second)


def test_a_mode_verified_tree_is_never_deduplicated(rooted):
    """perf-bench's source snapshot proves its integrity by the file being unwritable. A store
    object's mode belongs to the inode every holder shares, so one holder's chmod -- including the
    one needed to delete its own tree -- would break OTHER snapshots' verification."""
    body = b"s" * 4096
    sealed = _write(rooted / "artifacts" / "perf-bench" / "t" / "snap" / "src.py", body)
    seal_name = (SC.contract()["mode_verified_seals"][0]).replace("*", "deadbeef")
    (sealed.parent / seal_name).write_text("{}")
    loose = _write(rooted / "artifacts" / "delivery" / "b" / "src.py", body)

    assert SC.main(["dedup", "--min-bytes", "1024", "--apply"]) == 0

    assert _inode(sealed) != _inode(loose), "a mode-verified tree was pulled into the shared store"
    assert sealed.read_bytes() == body


def test_a_frozen_directory_gets_its_mode_back(rooted):
    """Replacing a directory entry needs write on the DIRECTORY, and a frozen tree has none. The
    permission has to be borrowed and returned, or the freeze is quietly undone."""
    body = b"f" * 4096
    first = _write(rooted / "artifacts" / "delivery" / "a" / "w.bin", body)
    second = _write(rooted / "artifacts" / "delivery" / "b" / "w.bin", body)
    for parent in (first.parent, second.parent):
        parent.chmod(0o500)
    try:
        assert SC.main(["dedup", "--min-bytes", "1024", "--apply"]) == 0
        assert _inode(first) == _inode(second)
        for parent in (first.parent, second.parent):
            assert parent.stat().st_mode & 0o777 == 0o500, "the freeze was left open"
    finally:
        for parent in (first.parent, second.parent):
            parent.chmod(0o700)


def test_a_dry_run_changes_nothing(rooted):
    body = b"d" * 4096
    first = _write(rooted / "artifacts" / "delivery" / "a" / "w.bin", body)
    second = _write(rooted / "artifacts" / "delivery" / "b" / "w.bin", body)

    assert SC.main(["dedup", "--min-bytes", "1024"]) == 0

    assert _inode(first) != _inode(second)


def test_small_files_are_below_the_threshold(rooted):
    """A 200-byte duplicate costs an inode to save 200 bytes; the walk should not spend on it."""
    first = _write(rooted / "artifacts" / "delivery" / "a" / "tiny", b"t" * 200)
    second = _write(rooted / "artifacts" / "delivery" / "b" / "tiny", b"t" * 200)

    SC.main(["dedup", "--min-bytes", "1024", "--apply"])

    assert _inode(first) != _inode(second)


def test_adopt_reports_only_the_bytes_it_actually_freed(rooted):
    """Two names for one inode free nothing when the first is adopted -- the bytes go when the last
    name does. Counting the size at each name would report the saving once per name."""
    body = b"n" * 4096
    first = _write(rooted / "artifacts" / "delivery" / "a" / "w.bin", body)
    second = first.parent / "hardlink.bin"
    os.link(first, second)
    store = CS.store_root()

    assert CS.adopt(first, store) == 0, "freed bytes that another name still holds"
    assert CS.adopt(second, store) == 4096
    assert first.read_bytes() == body and second.read_bytes() == body


def test_adopt_is_idempotent(rooted):
    """dedup runs repeatedly over a growing root; a second pass over adopted files must do nothing."""
    body = b"i" * 4096
    path = _write(rooted / "artifacts" / "delivery" / "a" / "w.bin", body)
    store = CS.store_root()

    CS.adopt(path, store)
    inode = _inode(path)
    assert CS.adopt(path, store) == 0
    assert _inode(path) == inode
    assert path.read_bytes() == body


def test_an_adopted_file_is_read_only(rooted):
    """The store's guarantee against silent corruption: an in-place write to one holder would reach
    every other holder, so the shared inode must refuse the write rather than take it."""
    body = b"r" * 4096
    path = _write(rooted / "artifacts" / "delivery" / "a" / "w.bin", body)
    CS.adopt(path, CS.store_root())

    with pytest.raises(PermissionError):
        with path.open("ab") as stream:
            stream.write(b"x")


def test_a_build_tree_is_never_walked(rooted):
    """A build tree is rewritten in place by construction: a compiler opening its output with
    O_TRUNC gets EACCES from a read-only file rather than a rebuild. The exclusion is declared in
    the contract as a property of that root, not decided here."""
    body = b"o" * 4096
    built = _write(rooted / "build" / "obj" / "const_blob.o", body)
    product = _write(rooted / "artifacts" / "delivery" / "a" / "const_blob.o", body)

    assert SC.main(["dedup", "--min-bytes", "1024", "--apply"]) == 0

    assert _inode(built) != _inode(product), "a live build tree was made read-only"
    assert built.stat().st_mode & 0o200, "a build output lost its write bit"


def test_explicit_paths_override_the_default_roots(rooted):
    """An operator who knows a tree is quiet must be able to say so, and one who knows a tree is
    live must be able to leave it out -- neither is a property the tool can read off the disk."""
    body = b"p" * 4096
    first = _write(rooted / "artifacts" / "delivery" / "a" / "w.bin", body)
    second = _write(rooted / "artifacts" / "delivery" / "b" / "w.bin", body)
    elsewhere = _write(rooted / "artifacts" / "probes" / "c" / "w.bin", body)

    assert SC.main(["dedup", str(rooted / "artifacts" / "delivery"), "--min-bytes", "1024", "--apply"]) == 0

    assert _inode(first) == _inode(second)
    assert _inode(elsewhere) != _inode(first), "a path outside the named roots was touched"


def test_the_reclaim_excludes_the_copy_the_store_keeps(rooted):
    """Counting only the inodes the names gave up over-reports by one file per new store object --
    for a group of seven 4.1 GB weight files, a 4.1 GB error in the number an operator acts on."""
    body = b"c" * 8192
    paths = [_write(rooted / "artifacts" / "delivery" / d / "w.bin", body) for d in "abcd"]

    found = SC.dedup_candidates([rooted], min_bytes=1024)
    result = SC.dedup(found["groups"])

    assert result["names"] == 4
    assert result["released_bytes"] == 4 * 8192
    assert result["stored_bytes"] == 8192, "the store's own copy was not counted"
    assert result["reclaimed_bytes"] == 3 * 8192 == found["reclaimable_bytes"]
    assert all(p.read_bytes() == body for p in paths)
