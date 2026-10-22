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


def test_ancestor_views_share_watch_coverage_and_keep_relative_members(tmp_path, inventory):
    root = tmp_path / 'runtime'
    child = root / 'private'
    child.mkdir(parents=True)
    leaf = child / 'secret'
    leaf.write_text('private')
    os.link(leaf, tmp_path / 'outside')
    parent = guarded(inventory, root)
    guard = inventory._entries[root][0]
    child_facts = inventory.tree_nofollow(child)
    assert child_facts.members == ((('secret',), leaf, 'file', 7),)
    assert child_facts.linked_inodes == parent.linked_inodes
    assert len(inventory._entries) == 1
    assert inventory.tree(child) == child_facts
    assert inventory._entries[root][0] is guard
    (child / 'new').write_text('changed')
    assert any(path.name == 'new' for _, path, _, _ in guarded(inventory, child).members)


def test_parent_guard_replaces_children_only_after_complete_admission(tmp_path, monkeypatch):
    monkeypatch.setattr(read_lease_digest, 'enabled', lambda: True)
    root = tmp_path / 'runtime'
    child = root / 'private'
    child.mkdir(parents=True)
    (child / 'secret').write_text('private')
    with_inventory = GuardedInventory(max_members=2)
    try:
        guarded(with_inventory, child)
        old_guard = with_inventory._entries[child][0]
        assert with_inventory.tree_nofollow(root) is not None
        assert old_guard.fd is None
        assert len(with_inventory._entries) == 1
        assert with_inventory._members == 2
        assert with_inventory.tree(child) is not None
        assert len(with_inventory._entries) == 1
    finally:
        with_inventory.close()


def test_nofollow_ancestor_cannot_grant_symlink_private_semantics(tmp_path, inventory):
    root = tmp_path / 'runtime'
    child = root / 'private'
    child.mkdir(parents=True)
    secret = tmp_path / 'secret'
    secret.write_text('private')
    (child / 'alias').symlink_to(secret)
    assert inventory.tree_nofollow(root) is not None
    assert inventory.tree(child) is None
    records = [('private', child, child)]
    assert H._private_hardlink_inodes(records, inventory=inventory) == H._private_hardlink_inodes(records)
    # An indirect requested root must not inherit the lexical parent guard.
    (root / 'directory_alias').symlink_to(child, target_is_directory=True)
    assert inventory.tree_nofollow(root / 'directory_alias') is None


def test_ancestor_view_observes_external_hardlink_and_root_replacement(tmp_path, inventory):
    root = tmp_path / 'runtime'
    child = root / 'private'
    child.mkdir(parents=True)
    leaf = child / 'secret'
    leaf.write_text('private')
    guarded(inventory, root)
    assert not inventory.tree(child).linked_inodes
    os.link(leaf, tmp_path / 'outside')
    assert inventory.tree(child).linked_inodes == frozenset({(leaf.stat().st_dev, leaf.stat().st_ino)})
    child.rename(root / 'old')
    child.mkdir()
    (child / 'replacement').write_text('new')
    assert {p.name for _, p, _, _ in guarded(inventory, child).members} == {'replacement'}


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


def test_nofollow_alias_inventory_reuses_symlink_tree_without_private_authority(tmp_path, inventory, monkeypatch):
    from merlin.targetgen.sandbox import guarded_inventory as G

    private = tmp_path / 'private'
    private.mkdir()
    secret = private / 'secret'
    secret.write_text('withheld')
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    (runtime / 'indirect').symlink_to(private, target_is_directory=True)
    os.link(secret, runtime / 'direct')
    argv = ['--ro-bind', str(runtime), '/runtime']
    records = [('private', private, private)]
    expected = H._host_hardlink_surfaces(argv, records, set())
    original = G._alias_members
    calls = []

    def scan(root):
        calls.append(root)
        return original(root)

    monkeypatch.setattr(G, '_alias_members', scan)
    first = inventory.tree_nofollow(runtime)
    if first is None:
        pytest.skip('kernel inotify guard unavailable')
    assert not any('secret' in parts for parts, _, _, _ in first.members)
    assert inventory.tree(runtime) is None  # private discovery stays strict
    count = len(calls)
    with ThreadPoolExecutor(max_workers=4) as workers:
        assert all(x is first for x in workers.map(inventory.tree_nofollow, [runtime] * 8))
    assert len(calls) == count
    assert H._host_hardlink_surfaces(argv, records, set(), inventory=inventory) == expected
    assert len(expected) == 1  # direct alias only, never the symlink referent

    # A skipped symlink becomes a real hardlink: no stale alias set is reused.
    (runtime / 'indirect').unlink()
    os.link(secret, runtime / 'indirect')
    assert inventory.tree_nofollow(runtime) is not first
    assert H._host_hardlink_surfaces(argv, records, set(), inventory=inventory) == H._host_hardlink_surfaces(argv, records, set())
    assert len(H._host_hardlink_surfaces(argv, records, set(), inventory=inventory)) == 2


def test_nofollow_guard_external_link_and_directory_replacement(tmp_path, inventory):
    root = tmp_path / 'runtime'
    root.mkdir()
    leaf = root / 'ordinary'
    leaf.write_text('bytes')
    (root / 'link').symlink_to(tmp_path / 'missing')
    first = inventory.tree_nofollow(root)
    if first is None:
        pytest.skip('kernel inotify guard unavailable')
    os.link(leaf, tmp_path / 'outside')
    second = inventory.tree_nofollow(root)
    assert second is not first
    assert (leaf.stat().st_dev, leaf.stat().st_ino) in second.linked_inodes
    (root / 'link').unlink()
    (root / 'link').mkdir()
    os.link(leaf, root / 'link' / 'new_alias')
    third = inventory.tree_nofollow(root)
    assert third is not second
    assert any(parts == ('link', 'new_alias') for parts, _, _, _ in third.members)
    leaf.write_text('changed')
    assert inventory.tree_nofollow(root) is not third


def test_nofollow_guard_setup_rescan_closes_on_mutation(tmp_path, inventory, monkeypatch):
    from merlin.targetgen.sandbox import guarded_inventory as G

    root = tmp_path / 'runtime'
    root.mkdir()
    (root / 'link').symlink_to(tmp_path / 'missing')
    original = G._alias_members
    calls = 0

    def changing(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            (root / 'late').write_text('new')
        return original(path)

    monkeypatch.setattr(G, '_alias_members', changing)
    assert inventory.tree_nofollow(root) is None
    assert not inventory._entries


def test_disjoint_alias_device_keeps_nested_mount_fallback(tmp_path, inventory, monkeypatch):
    root = tmp_path / 'system'
    root.mkdir()
    # No relevant nested mount: do not install a potentially huge irrelevant guard.
    monkeypatch.setattr(H, '_private_hardlink_inodes', lambda *a, **kw: {(root.stat().st_dev + 1, 123)})
    monkeypatch.setattr(H, '_host_mount_devices', lambda: [])
    monkeypatch.setattr(inventory, 'tree_nofollow', lambda *a: pytest.fail('disjoint tree scanned'))
    assert H._host_hardlink_surfaces(['--ro-bind', str(root), '/system'], [], set(), inventory=inventory) == []


@pytest.mark.parametrize('nofollow', [False, True])
def test_resource_exhaustion_retries_fresh_facts_once_per_interval(tmp_path, inventory, monkeypatch, nofollow):
    import errno
    from merlin.targetgen.sandbox import guarded_inventory as G

    root = tmp_path / 'runtime'
    root.mkdir()
    leaf = root / 'leaf'
    leaf.write_text('payload')
    native, calls, clock = B._RosterGuard, [], [100.0]
    def exhausted_once(*args, **kwargs):
        guard = native(*args, **kwargs)
        calls.append(guard)
        if len(calls) == 1:
            guard.close()
            guard._record_resource_failure(errno.ENOSPC)
        return guard
    monkeypatch.setattr(B, '_RosterGuard', exhausted_once)
    monkeypatch.setattr(G.time, 'monotonic', lambda: clock[0])
    tree = inventory.tree_nofollow if nofollow else inventory.tree
    assert tree(root) is None
    assert not inventory._unsupported and not inventory._entries
    # An outside hardlink created while unobserved MUST be in newly captured facts.
    os.link(leaf, tmp_path / 'new_alias')
    for _ in range(20):
        assert tree(root) is None
    assert len(calls) == 1
    clock[0] = 130.0
    with ThreadPoolExecutor(max_workers=8) as workers:
        facts = list(workers.map(tree, [root] * 16))
    assert len(calls) == 2 and facts[0] is not None
    assert all(fact is facts[0] for fact in facts)
    assert (leaf.stat().st_dev, leaf.stat().st_ino) in facts[0].linked_inodes
    assert not inventory._resource_retry_after
    inventory.close()
    assert calls[-1].fd is None


def test_resource_retry_does_not_convert_structural_refusal_into_cache(tmp_path, inventory, monkeypatch):
    import errno
    from merlin.targetgen.sandbox import guarded_inventory as G

    root = tmp_path / 'private'
    root.mkdir()
    (root / 'leaf').write_text('payload')
    native, clock = B._RosterGuard, [100.0]
    def exhausted(*args, **kwargs):
        guard = native(*args, **kwargs)
        guard.close()
        guard._record_resource_failure(errno.ENOSPC)
        return guard
    monkeypatch.setattr(B, '_RosterGuard', exhausted)
    monkeypatch.setattr(G.time, 'monotonic', lambda: clock[0])
    assert inventory.tree(root) is None
    (root / 'broken').symlink_to(tmp_path / 'missing')
    clock[0] = 130.0
    monkeypatch.setattr(B, '_RosterGuard', native)
    with pytest.raises(OSError):
        H._private_hardlink_inodes([('private', root, root)], inventory=inventory)
    assert root in inventory._unsupported and not inventory._entries
