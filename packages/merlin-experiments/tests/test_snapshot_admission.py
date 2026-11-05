"""Snapshot capacity is shared across processes; it grants no execution credit."""
import multiprocessing
from pathlib import Path

import pytest

from merlin_experiments.phase1.feedback import snapshot_admission as A


def _borrow(root, key, connection):
    try:
        with A.admitted(Path(root), key, timeout_s=0.1) as row:
            connection.send(('admitted', row['slot']))
            connection.recv()
    except TimeoutError:
        connection.send(('capacity', None))
    finally:
        connection.close()


def child(root, key):
    context = multiprocessing.get_context('spawn')
    parent, remote = context.Pipe()
    process = context.Process(target=_borrow, args=(str(root), key, remote))
    process.start()
    remote.close()
    assert parent.poll(10), 'child did not report snapshot admission'
    return process, parent, parent.recv()


def close(process, pipe, result):
    if result[0] == 'admitted':
        pipe.send('release')
    process.join(10)
    assert process.exitcode == 0
    pipe.close()


def test_two_distinct_sources_with_cross_process_same_source_borrow(tmp_path):
    root = tmp_path / 'admission'
    with A.admitted(root, 'a' * 64) as first, A.admitted(root, 'b' * 64):
        process, pipe, result = child(root, 'a' * 64)
        assert result == ('admitted', first['slot'])
        close(process, pipe, result)
        process, pipe, result = child(root, 'c' * 64)
        assert result == ('capacity', None)
        close(process, pipe, result)
    with A.admitted(root, 'c' * 64) as row:
        assert row['authority'] == 'scheduling_only'


def test_dead_process_releases_capacity(tmp_path):
    root = tmp_path / 'admission'
    with A.admitted(root, 'a' * 64):
        process, pipe, result = child(root, 'b' * 64)
        assert result[0] == 'admitted'
        process.terminate()
        process.join(10)
        pipe.close()
        with A.admitted(root, 'c' * 64, timeout_s=0.2):
            pass


def test_no_private_slot_alias_or_cpu_lease_wait(tmp_path, monkeypatch):
    from merlin.common import cpu_pool
    root = tmp_path / 'admission'
    root.mkdir(mode=0o700)
    (root / 'slot-0').symlink_to(tmp_path / 'outside')
    with pytest.raises(OSError):
        with A.admitted(root, 'a' * 64):
            pytest.fail('aliased slot admitted')
    monkeypatch.setattr(cpu_pool, 'current_thread_owns_cpu_lease', lambda: True)
    with pytest.raises(RuntimeError, match='CPU lease'):
        with A.admitted(root, 'a' * 64):
            pytest.fail('CPU owner waited for capacity')
