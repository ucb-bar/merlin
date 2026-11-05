"""Prepared lock roots satisfy simulator admission under a shared default umask."""
import importlib.util
import os
import stat

import pytest

from merlin.common.paths import repo_root


@pytest.fixture
def prepare():
    spec = importlib.util.spec_from_file_location(
        "prepare_author_launch_slots",
        repo_root() / "examples/gemmini/phase1/launch/prepare_author_launch.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("existing", [False, True])
def test_slot_roots_are_private_before_simulation(prepare, tmp_path, existing):
    roots = {"MERLIN_COMPUTE_CPU_SLOTS": tmp_path / "cpu", "MERLIN_GSIM_SLOT_ROOT": tmp_path / "gsim"}
    if existing:
        for root in roots.values():
            root.mkdir()
            root.chmod(0o775)
    old = os.umask(0o002)
    try:
        prepare.prepare_slot_roots(roots)
    finally:
        os.umask(old)
    for root in roots.values():
        info = root.lstat()
        assert info.st_uid == os.getuid()
        assert stat.S_IMODE(info.st_mode) == 0o700
        assert list(root.iterdir()) == []  # preparation does not acquire any slot


def test_preparation_does_not_chmod_link_target(prepare, tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir()
    target.chmod(0o755)
    link = tmp_path / "gsim"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(OSError):
        prepare.prepare_slot_roots({"MERLIN_COMPUTE_CPU_SLOTS": tmp_path / "cpu", "MERLIN_GSIM_SLOT_ROOT": link})
    assert stat.S_IMODE(target.stat().st_mode) == 0o755


@pytest.mark.parametrize('release', ['/old/repo/release', '/old/repo/new-release'])
def test_relocation_preserves_explicit_release_below_previous_repository(prepare, release):
    original = {
        'MERLIN_TARGET_CONTRACT': '/old/repo/release/payload/contract.yaml',
        'MERLIN_TARGET_EXPERIMENT': '/old/repo/release/payload/experiment.yaml',
        'MERLIN_RTL_FACTS': '/old/repo/release/payload/facts.json',
        'MERLIN_CORPUS_SEAL': '/old/repo/release/private/seal.json',
        'MERLIN_REPO_ROOT': '/old/repo',
        'PYTHONPATH': '/old/repo/src:/old/repo/packages/src',
        'SELECTION': '{"path":"/old/repo/release/selection.json","sha256":"unchanged"}',
        'SLOTS': '/old/.tmp/old-run/slots',
        'EXTERNAL': '/shared/runtime/bin/tool',
    }
    result = prepare.relocate_environment(original, (
        ('/old/repo', '/new/repo'), ('old-run', 'new-run'),
        ('/old/repo/release', release), ('/old/.tmp', '/new/.tmp'),
    ))
    for key in ('MERLIN_TARGET_CONTRACT', 'MERLIN_TARGET_EXPERIMENT', 'MERLIN_RTL_FACTS', 'MERLIN_CORPUS_SEAL'):
        assert result[key] == release + original[key].removeprefix('/old/repo/release')
    assert result['MERLIN_REPO_ROOT'] == '/new/repo'
    assert result['PYTHONPATH'] == '/new/repo/src:/new/repo/packages/src'
    assert result['SELECTION'] == '{"path":"' + release + '/selection.json","sha256":"unchanged"}'
    assert result['SLOTS'] == '/new/.tmp/new-run/slots'
    assert result['EXTERNAL'] == original['EXTERNAL']
    assert original['MERLIN_REPO_ROOT'] == '/old/repo'


def test_environment_relocation_never_rewrites_inserted_text(prepare):
    assert prepare.relocate_environment({'value': 'old-run:/old/repo'}, (
        ('old-run', 'old-run-next'), ('/old/repo', '/new/old-run/repo'),
    )) == {'value': 'old-run-next:/new/old-run/repo'}
