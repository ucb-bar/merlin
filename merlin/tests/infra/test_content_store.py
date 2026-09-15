"""The store itself: an ignore rule is honored exactly, and two frozen trees share their bytes.

``merlin.common.content_store`` replaced a deep copy in two places -- an agent run's declared input
closure and a performance suite's source snapshot. Both had a reason to copy (the frozen bytes must
not follow a later in-place edit of the source) and no reason to copy *per consumer*.
"""
from __future__ import annotations

import importlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from merlin.common import content_store as CS
from merlin.common.paths import merlin_dir


def _tree(root: Path) -> Path:
    (root / "keep").mkdir(parents=True)
    (root / "keep" / "wanted.txt").write_text("wanted\n", encoding="utf-8")
    (root / "top.txt").write_text("top\n", encoding="utf-8")
    return root


def test_two_frozen_trees_of_the_same_source_share_their_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv(CS.LOCATION_ENV, str(tmp_path / "store"))
    source = _tree(tmp_path / "src")

    CS.place_tree(source, tmp_path / "first", CS.store_root())
    CS.place_tree(source, tmp_path / "second", CS.store_root())

    for relative in ("keep/wanted.txt", "top.txt"):
        first, second = tmp_path / "first" / relative, tmp_path / "second" / relative
        assert first.read_bytes() == second.read_bytes()
        assert first.stat().st_ino == second.stat().st_ino, f"{relative} was stored twice"


def test_disabling_the_store_still_freezes_the_same_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv(CS.LOCATION_ENV, "")
    source = _tree(tmp_path / "src")
    assert CS.store_root() is None

    CS.place_tree(source, tmp_path / "copied", CS.store_root())
    landed = tmp_path / "copied" / "top.txt"
    assert landed.read_text() == "top\n"
    assert landed.stat().st_nlink == 1


def test_a_freeze_that_verifies_by_FILE_MODE_must_not_use_the_store(tmp_path, monkeypatch):
    """The boundary of this mechanism, kept as a test because it is not obvious.

    A store object's mode belongs to its inode, so it is shared by every consumer linking it and any
    one of them can change it for all the others -- including the chmod a caller has to perform to
    delete its own frozen tree. That is harmless for a freeze verified by CONTENT (the sandbox's
    bundle closure re-digests the bytes), and unsound for one that treats "this file is not writable"
    as evidence of its own integrity: the perf-bench source snapshot does exactly that, and adopting
    the store there made a second snapshot fail verification because an unrelated first one had been
    chmodded during its teardown. Reproduced here so the saving does not tempt someone to weaken the
    check instead.
    """
    monkeypatch.setenv(CS.LOCATION_ENV, str(tmp_path / "store"))
    source = _tree(tmp_path / "src")
    CS.place_tree(source, tmp_path / "first", CS.store_root())
    CS.place_tree(source, tmp_path / "second", CS.store_root())

    first = tmp_path / "first" / "top.txt"
    second = tmp_path / "second" / "top.txt"
    assert first.stat().st_ino == second.stat().st_ino
    first.chmod(0o600)                     # what a caller does to remove its own frozen tree
    assert second.stat().st_mode & 0o222, "a mode change reached the other consumer -- as documented"
    assert second.read_text() == "top\n", "the BYTES, which is what the store actually promises"


def test_the_perf_bench_snapshot_still_owns_its_own_bytes(tmp_path, monkeypatch):
    """Consequence of the above: that snapshot deliberately still deep-copies. Pinned so the
    mechanism is not quietly reintroduced there."""
    monkeypatch.setenv(CS.LOCATION_ENV, str(tmp_path / "store"))
    scripts = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    snapshot = importlib.import_module("perf_snapshot")

    source = tmp_path / "repo"
    _tree(source / "merlin" / "python")
    output_root = tmp_path / "out"
    output_root.mkdir()

    try:
        for name in ("run1", "run2"):
            snapshot.create(source, tmp_path / name, output_root=output_root,
                            source_roots=("merlin/python",))
            assert snapshot.verify(tmp_path / name)["schema"] == snapshot.SCHEMA

        first = tmp_path / "run1" / "merlin/python/keep/wanted.txt"
        second = tmp_path / "run2" / "merlin/python/keep/wanted.txt"
        assert first.read_text() == "wanted\n"
        assert first.stat().st_ino != second.stat().st_ino
        assert first.stat().st_nlink == 1, "the snapshot must own its mode, so it owns its inode"
    finally:
        # A finished snapshot is mode 0500 all the way down, so nothing -- including pytest's own
        # temp reclaim -- can remove it until the write bits come back. Leaving it is how a suite
        # accumulates unreclaimable trees, which is the thing this whole change is about.
        for name in ("run1", "run2"):
            root = tmp_path / name
            if not root.exists():
                continue
            root.chmod(0o700)
            for path in sorted(root.rglob("*"), key=lambda p: len(p.parts)):
                if not path.is_symlink():
                    path.chmod(0o700 if path.is_dir() else 0o600)


def test_a_source_edit_after_the_freeze_does_not_reach_either_snapshot(tmp_path, monkeypatch):
    """The property both callers copy for. A link to the SOURCE would lose it; a link to a store
    object does not, because the store holds its own copy."""
    monkeypatch.setenv(CS.LOCATION_ENV, str(tmp_path / "store"))
    source = _tree(tmp_path / "src")
    CS.place_tree(source, tmp_path / "frozen", CS.store_root())

    (source / "top.txt").write_bytes(b"edited in place")

    assert (tmp_path / "frozen" / "top.txt").read_text() == "top\n"


def test_a_non_regular_file_is_refused_rather_than_silently_dropped(tmp_path):
    source = _tree(tmp_path / "src")
    os.mkfifo(source / "pipe")
    with pytest.raises(RuntimeError, match="not a regular file or directory"):
        CS.place_tree(source, tmp_path / "out", CS.store_root())


def test_two_links_to_one_shared_directory_are_not_a_cycle(tmp_path, monkeypatch):
    """A tree may legitimately declare the same corpus twice. Only a directory that contains
    ITSELF makes the walk unbounded, and that is what must be refused."""
    monkeypatch.setenv(CS.LOCATION_ENV, str(tmp_path / "store"))
    source = _tree(tmp_path / "src")
    (source / "shared").mkdir()
    (source / "shared" / "corpus.txt").write_text("corpus\n", encoding="utf-8")
    (source / "keep" / "as_a").symlink_to(source / "shared")
    (source / "keep" / "as_b").symlink_to(source / "shared")

    CS.place_tree(source, tmp_path / "out", CS.store_root())
    out = tmp_path / "out" / "keep"
    assert (out / "as_a" / "corpus.txt").read_text() == "corpus\n"
    assert (out / "as_b" / "corpus.txt").read_text() == "corpus\n"
    assert (out / "as_a" / "corpus.txt").stat().st_ino == (out / "as_b" / "corpus.txt").stat().st_ino

    (source / "shared" / "back").symlink_to(source / "keep")
    with pytest.raises(RuntimeError, match="symlink cycle"):
        CS.place_tree(source, tmp_path / "looped", CS.store_root())


def test_a_dangling_symlink_is_named_rather_than_dropped(tmp_path, monkeypatch):
    monkeypatch.setenv(CS.LOCATION_ENV, str(tmp_path / "store"))
    source = _tree(tmp_path / "src")
    (source / "keep" / "gone").symlink_to(source / "never_existed")

    with pytest.raises(RuntimeError, match="not a regular file or directory"):
        CS.place_tree(source, tmp_path / "out", CS.store_root())


def _place(job: tuple[str, str, str]) -> tuple[str, int, int]:
    """One worker's freeze, run in its own process (module level so it is picklable)."""
    source, dst, store = (Path(p) for p in job)
    CS.place_file(source, dst, store)
    stat = dst.stat()
    return dst.read_text(), stat.st_size, stat.st_ino


def test_runs_racing_on_the_same_content_all_get_the_same_correct_bytes(tmp_path, monkeypatch):
    """~13 sessions share this host, so two campaigns materializing the same grant at the same
    moment is the normal case, not the edge one. A half-written object must never be linked: the
    store stages under a name no other writer can hold and renames it into place, which is atomic.
    """
    store = tmp_path / "store"
    monkeypatch.setenv(CS.LOCATION_ENV, str(store))
    source = tmp_path / "grant.bin"
    source.write_text("x" * 200_000, encoding="utf-8")

    jobs = []
    for n in range(12):
        run = tmp_path / f"run{n}"
        run.mkdir()
        jobs.append((str(source), str(run / "grant.bin"), str(store)))

    with ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(_place, jobs))

    assert {text for text, _, _ in results} == {"x" * 200_000}, "a worker saw partial bytes"
    assert {size for _, size, _ in results} == {200_000}
    # Whichever writer won the rename, every run ends up on ONE object -- that is the saving.
    assert len({ino for _, _, ino in results}) == 1
    assert not any(p.name.endswith(".pending") for p in store.rglob("*")), "staging file left behind"
