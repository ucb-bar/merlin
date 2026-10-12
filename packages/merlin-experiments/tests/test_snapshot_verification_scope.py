"""Finite invocation-local payload batching; no model/native execution."""
import os
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path

import pytest

from merlin.common import bundle_snapshot as S
from merlin.targetgen.sandbox import bwrap as B


@pytest.fixture
def frozen(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    source = repo / 'input'
    source.write_bytes(b'exact original payload')
    ws = tmp_path / 'workspace'
    ws.mkdir()
    bundle = {'allowed': [{'path': str(source), 'mode': 'ro'}], 'host_inputs': [], 'denied': []}
    B.materialize_bundle_inputs(ws, bundle, repo=repo)
    return ws, bundle, repo, S.snapshot_path(S.bundle_snapshot_root(ws), source, repo)


def test_scoped_repeated_and_context_copied_checks_hash_only_boundaries(frozen, monkeypatch):
    ws, bundle, repo, payload = frozen
    original = S.hashlib.file_digest
    calls = []
    def digest(stream, *args):
        calls.append(Path(stream.name))
        return original(stream, *args)
    monkeypatch.setattr(S.hashlib, 'file_digest', digest)
    with S.verified_snapshot_scope(ws, bundle, repo=repo):
        for _ in range(3):
            S.verify_bundle_snapshot(ws, bundle, repo=repo)
            B.verify_bundle_snapshot(ws, bundle, repo=repo)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(copy_context().run, S.verify_bundle_snapshot, ws, bundle, repo=repo).result()
    assert calls.count(payload) == 2
    S.verify_bundle_snapshot(ws, bundle, repo=repo)
    assert calls.count(payload) == 3


def test_scoped_final_check_reopens_changed_bytes_even_restored_mtime(frozen):
    ws, bundle, repo, payload = frozen
    before = payload.stat()
    with pytest.raises(RuntimeError, match='content differs'):
        with S.verified_snapshot_scope(ws, bundle, repo=repo):
            payload.chmod(0o600)
            payload.write_bytes(b'changed payload bytes')
            os.utime(payload, ns=(before.st_atime_ns, before.st_mtime_ns))
            S.verify_bundle_snapshot(ws, bundle, repo=repo)
    assert S._ACTIVE_VERIFICATION.get() is None


@pytest.mark.parametrize('which', ['marker', 'bundle'])
def test_scoped_owner_mutation_refuses_and_resets(frozen, which):
    ws, bundle, repo, payload = frozen
    with pytest.raises(RuntimeError, match='policy or marker changed'):
        with S.verified_snapshot_scope(ws, bundle, repo=repo):
            if which == 'marker':
                marker = S.bundle_snapshot_root(ws) / S.MARKER
                marker.chmod(0o600)
                marker.write_bytes(marker.read_bytes() + b'\n')
            else:
                bundle['denied'].append({'path': 'new-private-selection'})
    assert S._ACTIVE_VERIFICATION.get() is None


def test_exception_and_nested_scope_do_not_leave_live_reuse(frozen, monkeypatch):
    ws, bundle, repo, payload = frozen
    original = S.hashlib.file_digest
    calls = []
    def digest(stream, *args):
        calls.append(Path(stream.name))
        return original(stream, *args)
    monkeypatch.setattr(S.hashlib, 'file_digest', digest)
    with pytest.raises(ValueError, match='candidate failed'):
        with S.verified_snapshot_scope(ws, bundle, repo=repo):
            with S.verified_snapshot_scope(ws, bundle, repo=repo):
                S.verify_bundle_snapshot(ws, bundle, repo=repo)
            raise ValueError('candidate failed')
    assert calls.count(payload) == 4
    assert S._ACTIVE_VERIFICATION.get() is None


def test_nested_cross_root_and_unissued_token_cannot_reuse_signature(frozen, tmp_path, monkeypatch):
    ws, bundle, repo, payload = frozen
    other_ws = tmp_path / 'other' / 'workspace'
    other_ws.mkdir(parents=True)
    B.materialize_bundle_inputs(other_ws, bundle, repo=repo)
    other = S.snapshot_path(S.bundle_snapshot_root(other_ws), repo / 'input', repo)
    original = S.hashlib.file_digest
    calls = []
    def digest(stream, *args):
        calls.append(Path(stream.name))
        return original(stream, *args)
    monkeypatch.setattr(S.hashlib, 'file_digest', digest)
    with S.verified_snapshot_scope(ws, bundle, repo=repo):
        S.verify_bundle_snapshot(other_ws, bundle, repo=repo)
        with S.verified_snapshot_scope(other_ws, bundle, repo=repo):
            S.verify_bundle_snapshot(ws, bundle, repo=repo)
    assert calls.count(other) == 3
    assert calls.count(payload) == 3
    forged = S._Verification()
    forged.root = S.bundle_snapshot_root(ws)
    forged.signature = ('invented', 0, 0)
    token = S._ACTIVE_VERIFICATION.set(forged)
    try:
        assert S.verify_bundle_snapshot(ws, bundle, repo=repo)['n_files'] == 1
    finally:
        S._ACTIVE_VERIFICATION.reset(token)
    assert calls.count(payload) == 4


def test_final_custody_failure_prevents_output_publication(frozen, tmp_path):
    from merlin_experiments.phase1.feedback import compiler_profile_runtime as R
    ws, bundle, repo, payload = frozen
    owner = R.CommonTransformService.__new__(R.CommonTransformService)
    owner.workspace, owner.bundle, owner.repo = ws, bundle, repo
    executor = R._Executor.__new__(R._Executor)
    executor.owner = owner
    output = tmp_path / 'published-output.json'

    @R.snapshot_scoped_entrypoint
    def invoke(self):
        payload.chmod(0o600)
        payload.write_bytes(b'changed original bytes')
        R.finish_snapshot_invocation()
        output.write_bytes(b'must not be published')

    with pytest.raises(RuntimeError, match='content differs'):
        invoke(executor)
    assert not output.exists()
    assert S._ACTIVE_VERIFICATION.get() is None
    assert R._INVOCATION_PUBLICATION.get() is None
