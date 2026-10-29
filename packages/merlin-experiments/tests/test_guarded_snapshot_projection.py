"""Frozen metadata reuse retains live policy checks and watched request custody."""
import contextvars
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from merlin.common import bundle_snapshot as B
from merlin.targetgen.sandbox import bwrap as W
from test_compiler_profile_execution import execution as execution


def test_projection_copy_detaches_paths_and_preserves_container_cycles():
    path = Path('/one/long/shared/prefix/input')
    value = {'rows': [(path, path)]}
    value['self'] = value
    copied = B._copy_projection(value)
    assert copied is not value and copied['self'] is copied
    left, right = copied['rows'][0]
    assert left == path and left is right and left is not path
    copied['rows'].clear()
    assert value['rows'] == [(path, path)]


@pytest.fixture
def guarded(execution, monkeypatch):
    monkeypatch.setenv('MERLIN_SNAPSHOT_LEASE_CACHE', '1')
    owner, _, _ = execution
    entry = B.shared_entry_verification(owner.workspace, owner.bundle, repo=owner.repo)
    assert entry is not None
    with B.shared_entry_scope(entry):
        yield owner, entry
    entry.close()


def test_scoped_hit_is_single_flight_and_returns_independent_values(guarded):
    owner, entry = guarded
    calls = []
    def compute():
        calls.append(1)
        return {'paths': [owner.repo / 'input']}
    def read():
        with B.shared_entry_scope(entry):
            return B.guarded_projection(owner.workspace, owner.bundle, owner.repo, ['mapping'], compute)
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(lambda _: read(), range(8)))
    assert len(calls) == 1
    values[0]['paths'].clear()
    assert read()['paths'] == [owner.repo / 'input']


def test_unscoped_or_foreign_thread_context_cannot_reuse(guarded):
    owner, entry = guarded
    calls = []
    def read():
        return B.guarded_projection(owner.workspace, owner.bundle, owner.repo, ['mapping'], lambda: calls.append(1) or [])
    read(); read()
    assert len(calls) == 1
    copied = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(copied.run, read).result()
    with B.independent_snapshot_reads():
        read(); read()
    assert len(calls) == 4


def test_snapshot_rename_cannot_return_cached_mapping(guarded):
    owner, entry = guarded
    first = W._snapshot_grants(owner.workspace, owner.bundle, owner.repo)
    frozen = first[1][0][2]
    frozen.parent.chmod(0o755)
    frozen.rename(frozen.with_name('moved'))
    with pytest.raises(RuntimeError, match='incomplete'):
        W._snapshot_grants(owner.workspace, owner.bundle, owner.repo)


def test_marker_and_policy_changes_invalidate_projection(guarded):
    owner, entry = guarded
    calls = []
    def read(bundle):
        return B.guarded_projection(owner.workspace, bundle, owner.repo, ['mapping'], lambda: calls.append(1) or [])
    read(owner.bundle); read(owner.bundle)
    read({**owner.bundle, 'denied': [{'path': 'different'}]})
    marker = entry.root / B.MARKER
    marker.chmod(0o644)
    marker.write_bytes(marker.read_bytes() + b'\n')
    read(owner.bundle)
    assert len(calls) == 3


def test_warm_grants_reuse_frozen_records_but_keep_live_location_checks(guarded, monkeypatch):
    owner, entry = guarded
    reads, checks = [], []
    original = W._snapshot_records
    original_locations = W._validate_host_locations
    def records(*args, **kwargs):
        reads.append(1)
        return original(*args, **kwargs)
    def locations(*args, **kwargs):
        checks.append(1)
        return original_locations(*args, **kwargs)
    monkeypatch.setattr(W, '_snapshot_records', records)
    monkeypatch.setattr(W, '_validate_host_locations', locations)
    expected = W._snapshot_grants(owner.workspace, owner.bundle, owner.repo)
    for _ in range(3):
        assert W._snapshot_grants(owner.workspace, owner.bundle, owner.repo) == expected
    assert len(reads) == 2  # Public and host record classes, once each.
    assert len(checks) == 4  # Live original-path safety remains per call.
    with B.independent_snapshot_reads():
        W._snapshot_grants(owner.workspace, owner.bundle, owner.repo)
    assert len(reads) == 4


def test_declared_input_keys_do_not_alias(guarded):
    owner, entry = guarded
    def read(key):
        return B.guarded_projection(owner.workspace, owner.bundle, owner.repo, ['source', key], lambda: key)
    assert read('original') == 'original'
    assert read('changed-profile') == 'changed-profile'


def test_large_exact_declarations_do_not_exhaust_value_budget(guarded):
    owner, entry = guarded
    key = ['host-roster', 'a' * (17 << 20)]
    calls = []
    def read(value):
        return B.guarded_projection(owner.workspace, owner.bundle, owner.repo, value,
                                    lambda: calls.append(1) or ['small-projection'])
    assert read(key) == read(key) == ['small-projection']
    assert len(calls) == 1 and entry._projection_bytes < 1024
    read(['host-roster', 'b' * (17 << 20)])
    assert len(calls) == 2


def test_frozen_classification_reuses_only_current_validated_grants(guarded, monkeypatch):
    from types import SimpleNamespace
    from merlin.targetgen.sandbox.answer_surfaces import CapsuleMetadata

    owner, entry = guarded
    metadata = CapsuleMetadata()
    calls = []
    original = W._snapshot_grants
    def grants(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(W, '_snapshot_grants', grants)
    target = SimpleNamespace(capsule_corpus=None, corpus_siblings=lambda: ())
    try:
        W._frozen_answer_surfaces(owner.workspace, owner.bundle, target, [], owner.repo, metadata)
        assert len(calls) == 1  # No second host-roster resolution inside derived().
        W._frozen_answer_surfaces(owner.workspace, owner.bundle, target, [], owner.repo, metadata)
        assert len(calls) == 2  # Still validate each invocation before any mask reuse.
        marker = entry.root / B.MARKER
        marker.chmod(0o644)
        import json
        changed = json.loads(marker.read_bytes())
        changed['host_inputs'] = ['not-declared']
        marker.write_text(json.dumps(changed))
        with pytest.raises(RuntimeError, match='host-only input set'):
            W._frozen_answer_surfaces(owner.workspace, owner.bundle, target, [], owner.repo, metadata)
        assert len(calls) == 3
    finally:
        metadata.inventory.close()


def test_frozen_classification_unguarded_paths_keep_second_fresh_reader(guarded, monkeypatch):
    from types import SimpleNamespace
    from merlin.targetgen.sandbox.answer_surfaces import CapsuleMetadata

    owner, _ = guarded
    metadata = CapsuleMetadata()
    monkeypatch.setattr(metadata.inventory, 'tree', lambda root: None)
    calls = []
    original = W._snapshot_grants
    def grants(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(W, '_snapshot_grants', grants)
    target = SimpleNamespace(capsule_corpus=None, corpus_siblings=lambda: ())
    W._frozen_answer_surfaces(owner.workspace, owner.bundle, target, [], owner.repo, metadata)
    assert len(calls) == 2 and not metadata._derived
    metadata.inventory.close()


def test_marker_mutation_between_resolution_and_classification_is_reopened(guarded, monkeypatch):
    import json
    from types import SimpleNamespace
    from merlin.targetgen.sandbox.answer_surfaces import CapsuleMetadata

    owner, entry = guarded
    metadata = CapsuleMetadata()
    calls = []
    original = W._snapshot_grants
    def grants(*args):
        calls.append(1)
        result = original(*args)
        if len(calls) == 1:
            marker = entry.root / B.MARKER
            marker.chmod(0o644)
            changed = json.loads(marker.read_bytes())
            changed['host_inputs'] = ['changed-after-resolution']
            marker.write_text(json.dumps(changed))
        return result
    monkeypatch.setattr(W, '_snapshot_grants', grants)
    target = SimpleNamespace(capsule_corpus=None, corpus_siblings=lambda: ())
    try:
        with pytest.raises(RuntimeError, match='host-only input set'):
            W._frozen_answer_surfaces(owner.workspace, owner.bundle, target, [], owner.repo, metadata)
        assert len(calls) == 2 and not metadata._derived
    finally:
        metadata.inventory.close()
