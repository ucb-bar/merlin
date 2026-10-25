"""Retained source evidence is shared only under live, watched ownership."""
import concurrent.futures
import os
from pathlib import Path

import pytest

from merlin.targetgen.capsule_source_identity import SharedCompilerSource
from merlin.common import read_lease_digest


def make(tmp_path, monkeypatch, *, guarded=True):
    monkeypatch.setattr(read_lease_digest, 'enabled', lambda: guarded)
    source = tmp_path/'candidate'; source.mkdir()
    (source/'compiler.py').write_text('original\n')
    (source/'data').mkdir(); (source/'data'/'spec.json').write_text('{}')
    return SharedCompilerSource(source, tmp_path/'results')


def test_parallel_captures_copy_once_and_reuse_full_checks(tmp_path, monkeypatch):
    item = make(tmp_path, monkeypatch)
    if not all(g.fd is not None for g in item.guards):
        item.close(); pytest.skip('inotify unavailable')
    monkeypatch.setattr(item, '_verify_full', lambda: pytest.fail('repeated full source scan'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
        receipts = list(pool.map(lambda _: item.capture(), range(40)))
    assert all(row == receipts[0] for row in receipts)
    assert len(list((tmp_path/'results'/'compiler-source').iterdir())) == 1
    item.close()
    with pytest.raises(ValueError, match='not live'):
        item.capture()


@pytest.mark.parametrize('change', ['source', 'copy', 'rename', 'new', 'hardlink'])
def test_mutations_invalidate_for_all_threads(tmp_path, monkeypatch, change):
    item = make(tmp_path, monkeypatch)
    if change == 'source': (item.source/'compiler.py').write_text('modified\n')
    if change == 'copy': (item.destination/'compiler.py').write_text('modified\n')
    if change == 'rename': (item.source/'compiler.py').rename(item.source/'renamed.py')
    if change == 'new': (item.source/'extra.py').write_text('new')
    if change == 'hardlink':
        alias = tmp_path/'alias'; os.link(item.source/'compiler.py', alias)
        alias.write_text('modified\n')
    def check(_):
        with pytest.raises((ValueError, RuntimeError)):
            item.capture()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(check, range(8)))
    item.close()


def test_unavailable_watchers_rehash_and_ignore_only_generated_state(tmp_path, monkeypatch):
    item = make(tmp_path, monkeypatch, guarded=False)
    scans = []; original = item._verify_full
    def verify(): scans.append(True); original()
    monkeypatch.setattr(item, '_verify_full', verify)
    (item.source/'build').mkdir(); (item.source/'build'/'object').write_bytes(b'generated')
    item.capture(); item.capture()
    assert len(scans) == 2
    (item.source/'compiler.py').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        item.capture()
    item.close()


def test_fork_cannot_reuse_parent_owner(tmp_path, monkeypatch):
    item = make(tmp_path, monkeypatch)
    pid = os.fork()
    if pid == 0:
        try: item.capture()
        except ValueError: os._exit(0)
        os._exit(1)
    assert os.waitpid(pid, 0)[1] == 0
    item.close()


def test_archive_independently_rechecks_shared_tree(tmp_path, monkeypatch):
    from merlin_experiments.phase1.capsule_timing_archive import _compiler_candidate
    item = make(tmp_path, monkeypatch)
    receipt = {**item.capture(), 'status':'stable', 'artifacts':{}}
    artifacts = {}
    for name,key in [('package_kernel.elf','elf'),('harness.c','harness'),
                     ('input.interface.mlir','input_interface'),('command_buffer.json','command_buffer')]:
        receipt['artifacts'][name] = 'a'*64
        artifacts[key] = {'sha256':'a'*64}
    document = {'compiler_source':receipt}
    assert _compiler_candidate(document, tmp_path, artifacts) == item.digest['sha256']
    (item.destination/'compiler.py').write_text('changed')
    assert _compiler_candidate(document, tmp_path, artifacts) is None
    item.close()
