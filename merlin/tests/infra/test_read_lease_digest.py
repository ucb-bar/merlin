"""Real Linux lease custody; no compiler, model or native qualification."""

from __future__ import annotations

import errno
import hashlib
import json
import mmap
import os
import signal
from concurrent.futures import ThreadPoolExecutor

import pytest

from merlin.common import bundle_snapshot as S
from merlin.common import read_lease_digest as L


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    L.close()
    monkeypatch.setenv('MERLIN_SNAPSHOT_LEASE_CACHE', '1')
    monkeypatch.setattr(L, 'MIN_BYTES', 4096)
    yield
    L.close()


@pytest.fixture
def payload(tmp_path):
    path = tmp_path / 'sdk.bin'
    path.write_bytes(b'x' * 8192)
    return path


def require_lease(path):
    value = L.digest(path)
    if value is None:
        pytest.skip('Linux read leases unavailable for the selected temporary filesystem')
    assert value == hashlib.sha256(path.read_bytes()).hexdigest()
    return value


def break_lease(path, *, truncate=False):
    flags = os.O_WRONLY | os.O_NONBLOCK | (os.O_TRUNC if truncate else 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        assert error.errno in (errno.EAGAIN, errno.EWOULDBLOCK)
    else:
        # The dedicated custodian may already have invalidated/released it.
        os.close(descriptor)


def frozen(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    source = repo / 'sdk.bin'
    source.write_bytes(b'x' * 8192)
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    root = S.bundle_snapshot_root(workspace)
    payload = S.snapshot_path(root, source, repo)
    payload.parent.mkdir(parents=True)
    payload.write_bytes(source.read_bytes())
    bundle = {'allowed': [{'path': str(source)}], 'host_inputs': []}
    observed = dict(zip(('content_sha256', 'n_files', 'n_bytes'), S.snapshot_content(root), strict=True))
    marker = {'version': 4, 'repo': str(repo), 'allowed': [str(source)], 'host_inputs': [],
              'grants': [{'path': str(source), 'destination': str(source),
                          'snapshot': str(payload.relative_to(root))}], 'host_records': [], **observed}
    (root / S.MARKER).write_text(json.dumps(marker))
    return workspace, bundle, repo, payload


def test_real_reuse_hashes_once_and_leaves_parent_sigio_handler(payload, monkeypatch):
    original = L.hashlib.file_digest
    reads = []
    handler = signal.getsignal(signal.SIGIO)

    def count(stream, *args):
        reads.append(os.fstat(stream.fileno()).st_size)
        return original(stream, *args)

    monkeypatch.setattr(L.hashlib, 'file_digest', count)
    expected = require_lease(payload)
    with L.verification():
        assert L.digest(payload) == expected
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert list(pool.map(L.digest, [payload, payload])) == [expected, expected]
    assert reads == [8192]
    assert signal.getsignal(signal.SIGIO) == handler


@pytest.mark.parametrize('value', [None, '0', 'true'])
def test_full_hash_is_default_and_opt_in_is_exact(payload, monkeypatch, value):
    if value is None:
        monkeypatch.delenv('MERLIN_SNAPSHOT_LEASE_CACHE', raising=False)
    else:
        monkeypatch.setenv('MERLIN_SNAPSHOT_LEASE_CACHE', value)
    assert L.digest(payload) is None
    assert L._CACHE is None


@pytest.mark.parametrize('mapping', [False, True])
def test_existing_writer_or_writable_mapping_uses_full_hash(tmp_path, mapping):
    workspace, bundle, repo, path = frozen(tmp_path)
    L.close()
    writer = os.open(path, os.O_RDWR)
    view = mmap.mmap(writer, 8192, access=mmap.ACCESS_WRITE) if mapping else None
    if view is not None:
        os.close(writer)
        writer = None
    try:
        assert L.digest(path) is None
        S.verify_bundle_snapshot(workspace, bundle, repo=repo)
        before = path.stat()
        if view is not None:
            view[0] = ord('y')
        else:
            os.pwrite(writer, b'y', 0)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        with pytest.raises(RuntimeError, match='content differs'):
            S.verify_bundle_snapshot(workspace, bundle, repo=repo)
    finally:
        if view is not None:
            view.close()
        if writer is not None:
            os.close(writer)


@pytest.mark.parametrize('truncate', [False, True])
def test_write_or_truncate_attempt_refuses_before_publication(payload, tmp_path, truncate):
    require_lease(payload)
    published = tmp_path / 'accepted'
    with pytest.raises(RuntimeError, match='lease custody changed'):
        with L.verification():
            assert L.digest(payload) is not None
            break_lease(payload, truncate=truncate)
        published.write_bytes(b'wrong')
    assert not published.exists()
    assert L.digest(payload) is None  # Broken epochs do not turn into new cached hits.


def test_first_hash_must_finish_under_same_live_lease(payload, monkeypatch):
    original = L.hashlib.file_digest

    def break_during_hash(stream, *args):
        result = original(stream, *args)
        break_lease(payload)
        return result

    monkeypatch.setattr(L.hashlib, 'file_digest', break_during_hash)
    with pytest.raises(RuntimeError, match='lease custody changed'):
        L.digest(payload)
    assert not L._CACHE.entries


@pytest.mark.parametrize('same_bytes', [False, True])
def test_rename_replacement_invalidates_exact_cached_path(payload, tmp_path, same_bytes):
    require_lease(payload)
    replacement = tmp_path / 'replacement'
    replacement.write_bytes((b'x' if same_bytes else b'y') * 8192)
    with pytest.raises(RuntimeError, match='lease custody changed'):
        with L.verification():
            assert L.digest(payload) is not None
            os.replace(replacement, payload)


def test_symlink_replacement_refuses(payload, tmp_path):
    require_lease(payload)
    replacement = tmp_path / 'replacement'
    replacement.write_bytes(b'x' * 8192)
    payload.unlink()
    payload.symlink_to(replacement)
    with pytest.raises(RuntimeError, match='lease custody changed'):
        L.digest(payload)


def test_custodian_loss_refuses_and_falls_back_later(payload):
    require_lease(payload)
    L._CACHE.process.kill()
    L._CACHE.process.wait(timeout=3)
    with pytest.raises(RuntimeError, match='lease custody changed'):
        L.digest(payload)
    assert L.digest(payload) is None


def test_cache_caps_descriptors_and_does_not_recycle_active_entries(payload, tmp_path, monkeypatch):
    monkeypatch.setattr(L, 'MAX_FILES', 1)
    require_lease(payload)
    other = tmp_path / 'other'
    other.write_bytes(b'z' * 8192)
    assert L.digest(other) is None
    assert len(L._CACHE.entries) == 1
    assert L.digest(payload) is not None


@pytest.mark.parametrize('available', [0, 1])
def test_real_small_fd_budget_falls_back_and_prioritizes_largest(payload, tmp_path, monkeypatch, available):
    import resource

    larger = tmp_path / 'z-larger'
    larger.write_bytes(b'z' * 16384)
    with monkeypatch.context() as selected:
        selected.setenv('MERLIN_SNAPSHOT_LEASE_CACHE', '0')
        expected = S.snapshot_content(tmp_path)
    original_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    soft = len(os.listdir('/proc/self/fd')) + L.FD_HEADROOM + L._FD_OVERHEAD + available
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, original_limit[1]))
        assert S.snapshot_content(tmp_path) == expected
        assert L._CACHE.capacity == available
        assert len(L._CACHE.entries) == available
        if available:
            assert next(iter(L._CACHE.entries.values()))[2][2] == larger.stat().st_size
        else:
            assert L._CACHE.process is None
    finally:
        L.close()
        resource.setrlimit(resource.RLIMIT_NOFILE, original_limit)


def test_unsupported_kernel_keeps_independent_full_reads(tmp_path, monkeypatch):
    workspace, bundle, repo, path = frozen(tmp_path)
    L.close()
    monkeypatch.setattr(L, 'fcntl', None)
    before = path.stat()
    with pytest.raises(RuntimeError, match='content differs'):
        with S.verified_snapshot_scope(workspace, bundle, repo=repo):
            path.write_bytes(b'y' * 8192)
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


def test_final_snapshot_guard_refuses_broken_previously_used_member(tmp_path):
    workspace, bundle, repo, path = frozen(tmp_path)
    require_lease(path)
    before = path.stat()
    accepted = []
    with pytest.raises(RuntimeError, match='lease custody changed'):
        with S.verified_snapshot_scope(workspace, bundle, repo=repo):
            # Write access breaks the lease; restoring mtime never revives it.
            with path.open('r+b') as stream:
                stream.write(b'y')
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            with pytest.raises(RuntimeError, match='lease custody changed'):
                S.verify_bundle_snapshot(workspace, bundle, repo=repo)
            # A caught intermediate failure must not silently become a miss.
            assert L.digest(path) is None
        accepted.append(True)
    assert not accepted and S._ACTIVE_VERIFICATION.get() is None


def test_active_scope_refuses_lost_cache_even_when_bytes_stay_equal(tmp_path):
    workspace, bundle, repo, path = frozen(tmp_path)
    require_lease(path)
    accepted = []
    with pytest.raises(RuntimeError, match='lease owner changed'):
        with S.verified_snapshot_scope(workspace, bundle, repo=repo):
            L.close()
        accepted.append(True)
    assert not accepted and S._ACTIVE_VERIFICATION.get() is None


def test_fork_discards_parent_cache_without_releasing_parent_lease(payload):
    expected = require_lease(payload)
    parent_custodian = L._CACHE.process.pid
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            before = L._CACHE is None
            value = L.digest(payload)
            child_custodian = L._CACHE.process.pid
            os.write(write_fd, json.dumps([before, value, child_custodian]).encode())
            L.close()
        finally:
            os._exit(0)
    os.close(write_fd)
    try:
        before, value, child_custodian = json.loads(os.read(read_fd, 4096))
    finally:
        os.close(read_fd)
        _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert before and value == expected and child_custodian != parent_custodian
    assert L._CACHE.process.pid == parent_custodian
    assert L.digest(payload) == expected
