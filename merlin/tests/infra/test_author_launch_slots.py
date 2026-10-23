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
    return module.prepare_slot_roots


@pytest.mark.parametrize("existing", [False, True])
def test_slot_roots_are_private_before_simulation(prepare, tmp_path, existing):
    roots = {"MERLIN_COMPUTE_CPU_SLOTS": tmp_path / "cpu", "MERLIN_GSIM_SLOT_ROOT": tmp_path / "gsim"}
    if existing:
        for root in roots.values():
            root.mkdir()
            root.chmod(0o775)
    old = os.umask(0o002)
    try:
        prepare(roots)
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
        prepare({"MERLIN_COMPUTE_CPU_SLOTS": tmp_path / "cpu", "MERLIN_GSIM_SLOT_ROOT": link})
    assert stat.S_IMODE(target.stat().st_mode) == 0o755
