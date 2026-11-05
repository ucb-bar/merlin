"""Values derived from files are reused only while every file keeps its stat identity."""

from __future__ import annotations

import os
import time

import pytest

from merlin.common import stat_memo as M
from merlin.common.tree_hash import _hash_tree, hash_tree


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    M.clear()
    monkeypatch.setattr(M, "SETTLED_NS", -(10**12))  # file times cannot be aged in a test
    yield
    M.clear()


def _tree(root):
    (root / "pkg" / "__pycache__").mkdir(parents=True)
    (root / "pkg" / "a.py").write_text("a = 1\n")
    (root / "pkg" / "__pycache__" / "a.pyc").write_bytes(b"cache")
    (root / "build").mkdir()
    (root / "build" / "out.o").write_bytes(b"object")
    return root


def test_hash_tree_is_reused_until_any_hashed_file_changes(tmp_path):
    root = _tree(tmp_path / "src")
    first = hash_tree(root)
    assert first == _hash_tree(root)
    calls = []
    original = M.cached

    def spy(kind, key, signature, compute):
        return original(kind, key, signature, lambda: calls.append(kind) or compute())

    M.cached, saved = spy, M.cached
    try:
        assert hash_tree(root) == first and calls == []
        (root / "pkg" / "__pycache__" / "a.pyc").write_bytes(b"other cache")  # never hashed: no recompute
        assert hash_tree(root) == first and calls == []
        time.sleep(0.05)
        (root / "pkg" / "a.py").write_text("a = 2\n")
        changed = hash_tree(root)
        assert calls == ["hash_tree"] and changed == _hash_tree(root) and changed != first
    finally:
        M.cached = saved


def test_same_size_rewrite_with_restored_mtime_is_still_seen(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"aaaa")
    key = str(path)
    assert M.cached("bytes", key, lambda: M.file_signature(path), path.read_bytes) == b"aaaa"
    info = path.stat()
    time.sleep(0.05)
    path.write_bytes(b"bbbb")
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))  # only ctime still moves
    assert M.cached("bytes", key, lambda: M.file_signature(path), path.read_bytes) == b"bbbb"


def test_links_recent_files_and_old_values_are_never_reused(tmp_path, monkeypatch):
    root = _tree(tmp_path / "src")
    (root / "link").symlink_to(root / "pkg" / "a.py")
    assert M.tree_signature(root) is None  # a link's target is not covered by its stat
    calls = []
    compute = lambda: calls.append(1) or len(calls)
    plain = tmp_path / "plain.txt"
    plain.write_text("x")
    M.cached("k", "plain", lambda: M.file_signature(plain), compute)
    M.cached("k", "plain", lambda: M.file_signature(plain), compute)
    assert len(calls) == 1
    monkeypatch.setattr(M, "MAX_AGE_S", 0.0)
    M.cached("k", "plain", lambda: M.file_signature(plain), compute)
    assert len(calls) == 2
    monkeypatch.setattr(M, "MAX_AGE_S", 300.0)
    monkeypatch.setattr(M, "SETTLED_NS", 10**12)  # everything looks freshly written
    M.clear()
    M.cached("k", "plain", lambda: M.file_signature(plain), compute)
    M.cached("k", "plain", lambda: M.file_signature(plain), compute)
    assert len(calls) == 4


def test_failures_are_not_cached_and_values_are_copies(tmp_path):
    path = tmp_path / "f.txt"
    path.write_text("x")
    calls = []

    def failing():
        calls.append(1)
        raise ValueError("refused")

    for _ in range(2):
        with pytest.raises(ValueError):
            M.cached("k", "f", lambda: M.file_signature(path), failing)
    assert len(calls) == 2
    value = M.cached("d", "f", lambda: M.file_signature(path), lambda: {"n": [1]})
    value["n"].append(2)
    assert M.cached("d", "f", lambda: M.file_signature(path), lambda: {"n": [9]}) == {"n": [1]}


def test_disabled_memo_always_computes(tmp_path, monkeypatch):
    path = tmp_path / "f.txt"
    path.write_text("x")
    monkeypatch.setattr(M, "_DISABLED", True)
    calls = []
    for _ in range(2):
        M.cached("k", "f", lambda: M.file_signature(path), lambda: calls.append(1))
    assert len(calls) == 2
