"""Guarded facts never freeze mutable privacy or mount-policy decisions."""
import importlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from merlin.common import bundle_snapshot as B
from merlin.common import read_lease_digest
from merlin.targetgen.sandbox.guarded_inventory import GuardedInventory
from merlin.targetgen.sandbox import host_surfaces as H

A = importlib.import_module('merlin.targetgen.sandbox.answer_surfaces')


@pytest.fixture
def inventory(monkeypatch):
    monkeypatch.setattr(read_lease_digest, 'enabled', lambda: True)
    value = GuardedInventory()
    yield value
    value.close()


def guarded(value, root):
    facts = value.tree(root)
    if facts is None:
        pytest.skip('kernel inotify guard unavailable')
    return facts


def test_singleflight_and_external_hardlink_mutation(tmp_path, inventory, monkeypatch):
    root = tmp_path / 'private'
    root.mkdir()
    leaf = root / 'secret'
    leaf.write_text('private')
    original = B._scanned_members
    calls = []

    def scan(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(B, '_scanned_members', scan)
    first = guarded(inventory, root)
    count = len(calls)
    with ThreadPoolExecutor(max_workers=4) as workers:
        assert all(x is first for x in workers.map(inventory.tree, [root] * 8))
    assert len(calls) == count
    assert first.linked_inodes == frozenset()
    os.link(leaf, tmp_path / 'outside')
    records = [('private', root, root)]
    observed = H._private_hardlink_inodes(records, inventory=inventory)
    assert observed == H._private_hardlink_inodes(records)
    assert (leaf.stat().st_dev, leaf.stat().st_ino) in observed
    (root / 'new').write_text('new')
    assert any(path.name == 'new' for _, path, _, _ in guarded(inventory, root).members)


def test_mutated_root_symlink_and_unreadable_scan_keep_original_refusal(tmp_path, inventory, monkeypatch):
    root = tmp_path / 'private'
    root.mkdir()
    (root / 'secret').write_text('private')
    guarded(inventory, root)
    moved = tmp_path / 'old'
    root.rename(moved)
    root.mkdir()
    (root / 'replacement').write_text('new')
    assert {p.name for _, p, _, _ in guarded(inventory, root).members} == {'replacement'}
    (root / 'broken').symlink_to(tmp_path / 'missing')
    with pytest.raises(OSError):
        H._private_hardlink_inodes([('private', root, root)], inventory=inventory)
    (root / 'broken').unlink()
    original = os.scandir

    def denied(path):
        if Path(path) == root:
            raise PermissionError('private tree unreadable')
        return original(path)

    monkeypatch.setattr(os, 'scandir', denied)
    with pytest.raises(PermissionError):
        H._private_hardlink_inodes([('private', root, root)], inventory=inventory)


def test_changed_mount_topology_and_owner_close_disable_reuse(tmp_path, inventory, monkeypatch):
    root = tmp_path / 'private'
    root.mkdir()
    (root / 'secret').write_text('private')
    facts = guarded(inventory, root)
    previous_guard = inventory._entries[root][0]
    original = Path.read_bytes

    def changed(path):
        return original(path) + b'\n' if str(path) == '/proc/self/mountinfo' else original(path)

    monkeypatch.setattr(Path, 'read_bytes', changed)
    assert inventory.tree(root) is not facts
    assert previous_guard.fd is None
    inventory.close()
    assert inventory.tree(root) is None


def test_answer_discovery_observes_new_file_and_metadata_bytes(tmp_path, inventory):
    root = tmp_path / 'corpus'
    root.mkdir()
    (root / 'golden.yaml').write_text('withheld')
    (root / 'capsule.yaml').write_text('name: original\n')
    metadata = A.CapsuleMetadata()
    metadata.inventory.close()
    metadata.inventory = inventory
    assert set(A._discovered(root, 'golden.*', metadata)) == {root / 'golden.yaml'}
    assert metadata.load(root)['name'] == 'original'
    (root / 'golden.extra').write_text('withheld')
    (root / 'capsule.yaml').write_text('name: changed\n')
    assert set(A._discovered(root, 'golden.*', metadata)) == set(root.rglob('golden.*'))
    assert metadata.load(root)['name'] == 'changed'
    (root / 'alias').symlink_to(tmp_path)
    assert set(A._discovered(root, 'golden.*', metadata)) == set(root.rglob('golden.*'))


def test_bound_or_unsupported_guard_falls_back_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(read_lease_digest, 'enabled', lambda: True)
    root = tmp_path / 'private'
    root.mkdir()
    (root / 'a').write_text('a')
    value = GuardedInventory(max_members=0)
    assert value.tree(root) is None
    os.link(root / 'a', tmp_path / 'alias')
    records = [('private', root, root)]
    assert H._private_hardlink_inodes(records, inventory=value) == H._private_hardlink_inodes(records)
    value.close()


def test_runtime_alias_changes_and_fresh_mount_authority(tmp_path, inventory):
    private = tmp_path / "private"
    private.mkdir()
    leaf = private / "secret"
    leaf.write_text("private")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "ordinary").write_text("ordinary")
    argv = ["--ro-bind", str(runtime), "/runtime"]
    records = [("private", private, private)]
    assert H._host_hardlink_surfaces(argv, records, set(), inventory=inventory) == []
    os.link(leaf, runtime / "new_alias")
    expected = H._host_hardlink_surfaces(argv, records, set())
    assert expected
    assert H._host_hardlink_surfaces(argv, records, set(), inventory=inventory) == expected
    # Approved exact mounts are a current policy decision, never part of the cache.
    assert H._host_hardlink_surfaces(argv, records, {(runtime, Path("/runtime"))}, inventory=inventory) == []
    (runtime / "new_alias").unlink()
    (runtime / "indirect").symlink_to(leaf)
    assert H._host_hardlink_surfaces(argv, records, set(), inventory=inventory) == H._host_hardlink_surfaces(argv, records, set())
