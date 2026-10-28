"""Offloaded inventories return the same rows and errors as in-process ones, and change nothing else."""

import os
from pathlib import Path

import pytest

from merlin_experiments.capture_execution import inventory_processes as IP
from merlin_experiments.capture_execution import readonly_runtime, sealed_m2m, sealed_static
from merlin_experiments.capture_execution.sealed_static import SealedCaptureError


def _sample(root: Path) -> Path:
    (root / "pkg/sub").mkdir(parents=True)
    (root / "pkg/a.py").write_text("print(1)\n")
    (root / "pkg/sub/big.bin").write_bytes(os.urandom(3 * 1024 * 1024))
    (root / "pkg/sub/x").write_bytes(b"x")
    os.chmod(root / "pkg/sub/x", 0o751)
    (root / "alias").symlink_to("pkg/sub", target_is_directory=True)
    return root


def _inventories(root: Path):
    return (
        sealed_static._tree(root, allow_directory_symlinks=True),
        readonly_runtime._tree(root, allow_directory_symlinks=True),
        sealed_m2m._tree(root),
        sealed_m2m._snapshot_tree(root),
        sealed_m2m._source_tree(root),
    )


def test_offloaded_inventories_are_identical_and_scoped(tmp_path):
    root = _sample(tmp_path / "tree")
    originals = (sealed_static._tree, readonly_runtime._tree, sealed_m2m._strict_tree, sealed_m2m._source_tree)
    local = _inventories(root)
    with IP.offloaded_inventories(2) as active:
        assert active == 2
        assert sealed_static._tree is not originals[0] and sealed_m2m._source_tree is not originals[3]
        offloaded = _inventories(root)
        calls = []
        # The calling process does no inventory work itself while routed.
        sealed_static._file_digest, saved = (lambda path: calls.append(path)), sealed_static._file_digest
        try:
            assert _inventories(root) == local
        finally:
            sealed_static._file_digest = saved
        assert calls == []
    assert offloaded == local
    assert (sealed_static._tree, readonly_runtime._tree, sealed_m2m._strict_tree, sealed_m2m._source_tree) == originals


def test_offloaded_inventory_sees_changes_made_between_calls(tmp_path):
    root = _sample(tmp_path / "tree")
    with IP.offloaded_inventories(2):
        before = sealed_m2m._snapshot_tree(root)
        (root / "pkg/a.py").write_text("print(2)\n")
        after = sealed_m2m._snapshot_tree(root)
    assert before != after == sealed_m2m._snapshot_tree(root)


def test_offloaded_errors_keep_type_message_and_cause_chain(tmp_path):
    root = _sample(tmp_path / "tree")
    (root / "pkg/link").symlink_to("/etc/hostname")
    with pytest.raises(SealedCaptureError) as local:
        sealed_static._tree(root)
    with pytest.raises(sealed_m2m.SealedM2MError) as local_m2m:
        sealed_m2m._source_tree(root)
    with IP.offloaded_inventories(2):
        with pytest.raises(SealedCaptureError) as routed:
            sealed_static._tree(root)
        with pytest.raises(sealed_m2m.SealedM2MError) as routed_m2m:
            sealed_m2m._source_tree(root)
    assert str(routed.value) == str(local.value)
    assert str(routed_m2m.value) == str(local_m2m.value)
    assert type(routed_m2m.value.__cause__) is type(local_m2m.value.__cause__)
    assert str(routed_m2m.value.__cause__) == str(local_m2m.value.__cause__)


def test_single_worker_and_disabled_env_change_nothing(tmp_path, monkeypatch):
    original = sealed_static._tree
    with IP.offloaded_inventories(1) as active:
        assert active == 0 and sealed_static._tree is original
    monkeypatch.setenv(IP.PROCESSES_ENV, "0")
    with IP.offloaded_inventories(8) as active:
        assert active == 0 and sealed_static._tree is original


def test_a_replaced_inventory_is_never_rerouted(tmp_path, monkeypatch):
    def fake(root, **kwargs):
        return {"fake": {}}

    monkeypatch.setattr(sealed_static, "_tree", fake)
    with IP.offloaded_inventories(2):
        assert sealed_static._tree is fake
        assert sealed_static._tree(tmp_path) == {"fake": {}}
    assert sealed_static._tree is fake


def test_relative_roots_stay_in_process(tmp_path, monkeypatch):
    root = _sample(tmp_path / "tree")
    monkeypatch.chdir(tmp_path)
    with IP.offloaded_inventories(2):
        assert sealed_static._tree(Path("tree"), allow_directory_symlinks=True) == sealed_static._tree.__wrapped__(
            root, allow_directory_symlinks=True
        )


def test_unusable_process_pool_leaves_inventories_in_process(monkeypatch):
    original = sealed_static._tree

    def refuse(*args, **kwargs):
        raise OSError("no shared memory")

    monkeypatch.setattr(IP, "ProcessPoolExecutor", refuse)
    with IP.offloaded_inventories(4) as active:
        assert active == 0 and sealed_static._tree is original


def test_worker_refuses_a_definition_from_other_source_bytes(tmp_path):
    root = _sample(tmp_path / "tree")
    assert isinstance(IP._inventory("sealed_static", "_tree", "0" * 64, root, {}), IP._Rerun)
    digest = IP._source_digest("sealed_static")
    assert IP._inventory("sealed_static", "_tree", digest, root, {"allow_directory_symlinks": True}) == (
        "rows",
        sealed_static._tree(root, allow_directory_symlinks=True),
    )
