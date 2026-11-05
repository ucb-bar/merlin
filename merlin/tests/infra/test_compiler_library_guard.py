"""Exact request-owned library reuse, fresh final reads and mutation refusal."""
from contextvars import copy_context
from concurrent.futures import ThreadPoolExecutor

import pytest

from merlin.targetgen import compiler_library as L


@pytest.fixture
def selected(tmp_path, monkeypatch):
    monkeypatch.setenv('MERLIN_SNAPSHOT_LEASE_CACHE', '1')
    root = tmp_path / 'library'
    (root / 'merlin').mkdir(parents=True)
    (root / 'merlin/__init__.py').write_text('')
    source = root / 'merlin/portable.py'
    source.write_text('def identity(value): return value\n')
    contract = L.freeze_compiler_library(root, review_id='guard-test', public_modules=('merlin.portable',),
        sources=(('merlin/__init__.py', 'merlin'), ('merlin/portable.py', 'merlin.portable')))
    selection = L.CompilerLibrarySelection(contract, root)
    reads = []
    original = L._read
    def counted(*args):
        reads.append(args[1])
        return original(*args)
    monkeypatch.setattr(L, '_read', counted)
    return selection, source, reads


def test_live_request_reuses_across_worker_context_and_finishes_with_full_read(selected):
    selection, _source, reads = selected
    with L.guarded_compiler_library_verification(selection):
        scope = L._LIBRARY_VERIFICATION.get()
        if scope.guard is None or not scope.guard.unchanged():
            pytest.skip('kernel watches unavailable; full verification retained')
        cold = len(reads)
        expected = selection.record()
        expected['contract']['review_id'] = 'caller mutation'
        with ThreadPoolExecutor(max_workers=1) as pool:
            observed = pool.submit(copy_context().run, selection.record).result()
        assert observed['contract']['review_id'] == 'guard-test'
        assert len(reads) == cold
        retained = copy_context()
    assert len(reads) == cold * 2  # Mandatory independent final verification.
    retained.run(selection.record)
    assert len(reads) == cold * 3  # Expired copied context cannot reuse.


def test_mutated_member_refuses_warm_access_and_final(selected):
    selection, source, _reads = selected
    with pytest.raises(L.CompilerLibraryError, match='bytes changed'):
        with L.guarded_compiler_library_verification(selection):
            selection.record()
            source.write_text('def identity(value): return None\n')
            with pytest.raises(L.CompilerLibraryError, match='bytes changed'):
                selection.record()


def test_library_scope_cannot_reuse_another_root_or_selection(selected, tmp_path):
    selection, _source, reads = selected
    other = L.CompilerLibrarySelection(selection.contract, selection.root)
    with L.guarded_compiler_library_verification(selection):
        before = len(reads)
        other.record()
        assert len(reads) > before
        original = selection.root
        object.__setattr__(selection, 'root', tmp_path)
        try:
            with pytest.raises(L.CompilerLibraryError, match='selection|declaration'):
                selection.record()
        finally:
            object.__setattr__(selection, 'root', original)


def test_changed_policy_refuses_and_final_reads_even_after_body_failure(selected, monkeypatch):
    selection, _source, reads = selected
    original = L._FORBIDDEN_MODULES
    with pytest.raises(L.CompilerLibraryError, match='changed'):
        with L.guarded_compiler_library_verification(selection):
            before = len(reads)
            monkeypatch.setattr(L, '_FORBIDDEN_MODULES', (*original, 'additional.private'))
            with pytest.raises(L.CompilerLibraryError, match='verifier changed'):
                selection.record()
            raise OSError('body failed')
    assert len(reads) > before


def test_symlink_member_and_unavailable_guard_do_not_skip_checks(selected, monkeypatch):
    selection, source, reads = selected
    monkeypatch.setenv('MERLIN_SNAPSHOT_LEASE_CACHE', '0')
    with L.guarded_compiler_library_verification(selection):
        before = len(reads)
        selection.record()
        assert len(reads) > before
    with pytest.raises(L.CompilerLibraryError, match='linked'):
        with L.guarded_compiler_library_verification(selection):
            target = source.with_suffix('.backup')
            source.rename(target)
            source.symlink_to(target)
            selection.record()


def test_outside_hardlink_mutation_and_foreign_process_do_not_reuse(selected, tmp_path, monkeypatch):
    import os
    selection, source, reads = selected
    alias = tmp_path / 'external-link'
    os.link(source, alias)
    with pytest.raises(L.CompilerLibraryError, match='bytes changed'):
        with L.guarded_compiler_library_verification(selection):
            scope = L._LIBRARY_VERIFICATION.get()
            before = len(reads)
            with monkeypatch.context() as patch:
                patch.setattr(L.os, 'getpid', lambda: scope.process + 1)
                selection.record()
            assert len(reads) > before
            alias.write_text('def identity(value): return False\n')
            with pytest.raises(L.CompilerLibraryError, match='bytes changed'):
                selection.record()
