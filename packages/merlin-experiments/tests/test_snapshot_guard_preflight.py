"""Fail early on selected guard exhaustion; never replace custody verification."""
import errno
from types import SimpleNamespace

import pytest

from merlin.common import bundle_snapshot as B
from merlin.targetgen.sandbox import preflight as P


def test_disabled_reuse_does_not_allocate_guard_or_census(monkeypatch):
    monkeypatch.delenv('MERLIN_SNAPSHOT_LEASE_CACHE', raising=False)
    monkeypatch.setattr(P, '_inotify_pressure', lambda: pytest.fail('unselected census'))
    monkeypatch.setattr(B, '_RosterGuard', lambda *args: pytest.fail('unselected guard'))
    result = P.require_snapshot_guards()
    assert result['status'] == 'disabled' and not result['selected']
    assert 'Full independent' in result['fallback']


@pytest.mark.parametrize(('number', 'reason'), [
    (errno.EMFILE, 'inotify_instances_or_process_fd_limit'),
    (errno.ENOSPC, 'inotify_watch_limit'),
    (errno.ENFILE, 'system_open_file_limit'),
    (errno.ENOMEM, 'kernel_memory_pressure'),
    (None, 'guard_could_not_be_armed'),
])
def test_resource_failures_are_explicit_refusals_and_close_probe(monkeypatch, tmp_path, number, reason):
    monkeypatch.setenv('MERLIN_SNAPSHOT_LEASE_CACHE', '1')
    closed = []
    monkeypatch.setattr(B, '_RosterGuard', lambda *args: SimpleNamespace(
        resource_errno=number, unchanged=lambda: False, close=lambda: closed.append(1)))
    monkeypatch.setattr(P, '_inotify_pressure', lambda: {'limits': {'max_user_instances': 128}})
    native = P.probe_snapshot_guards
    monkeypatch.setattr(P, 'probe_snapshot_guards', lambda: native(scratch_parent=tmp_path))
    with pytest.raises(P.SnapshotGuardUnavailable) as caught:
        P.require_snapshot_guards()
    assert caught.value.record['reason'] == reason
    assert caught.value.record['resource_errno'] == number
    assert 'no checks are skipped' in str(caught.value)
    assert closed == [1] and not list(tmp_path.iterdir())


def test_real_guard_probe_observes_mutation_and_retains_no_descriptors(monkeypatch, tmp_path):
    monkeypatch.setenv('MERLIN_SNAPSHOT_LEASE_CACHE', '1')
    guards = []
    original = B._RosterGuard
    def observe(*args):
        guard = original(*args)
        guards.append(guard)
        return guard
    monkeypatch.setattr(B, '_RosterGuard', observe)
    result = P.probe_snapshot_guards(scratch_parent=tmp_path, include_pressure=False)
    if result['resource_errno'] is not None:
        pytest.skip(f'host guard resource exhaustion: {result}')
    assert result['status'] == 'ready'
    assert result['reason'] == 'allocated_and_mutation_invalidated'
    assert len(guards) == 1 and guards[0].fd is None
    assert not list(tmp_path.iterdir())


def test_probe_does_not_cache_exhaustion_after_capacity_recovers(monkeypatch, tmp_path):
    monkeypatch.setenv('MERLIN_SNAPSHOT_LEASE_CACHE', '1')
    observations = []
    def allocate(*args):
        fail = not observations
        observations.append(fail)
        answers = iter((False,) if fail else (True, False))
        return SimpleNamespace(resource_errno=errno.EMFILE if fail else None,
                               unchanged=lambda: next(answers), close=lambda: None)
    monkeypatch.setattr(B, '_RosterGuard', allocate)
    first = P.probe_snapshot_guards(scratch_parent=tmp_path, include_pressure=False)
    second = P.probe_snapshot_guards(scratch_parent=tmp_path, include_pressure=False)
    assert first['status'] == 'unavailable' and second['status'] == 'ready'
