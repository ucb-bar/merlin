"""Installed source snapshot authority is stdlib-only and requires explicit owners."""

import inspect
import json
import subprocess
import sys

import pytest
from merlin_experiments import source_snapshot as SNAP

from merlin.common.paths import module_source_path


@pytest.fixture
def snapshot(tmp_path):
    source = tmp_path / "source"
    (source / "python/toy").mkdir(parents=True)
    (source / "python/toy/__init__.py").write_text("VALUE = 1\n")
    frozen = tmp_path / "snapshot"
    SNAP.create(
        source,
        frozen,
        output_root=tmp_path / "output",
        source_roots=("python",),
        python_roots=("python",),
        legacy_roots=(),
    )
    yield frozen
    for path in (frozen, *frozen.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o700)


def test_creation_requires_all_three_explicit_ownership_inputs():
    parameters = inspect.signature(SNAP.create).parameters
    for name in ("source_roots", "python_roots", "legacy_roots"):
        assert parameters[name].default is inspect.Parameter.empty


@pytest.mark.parametrize("initial", [None, ()])
def test_selection_presence_and_membership_are_checked_before_copy(tmp_path, initial):
    source = tmp_path / "source"
    owner = source / "python"
    owner.mkdir(parents=True)
    selected = owner / "selected"
    if initial is not None:
        selected.mkdir()
    expected = {"python/selected": initial}
    selected.mkdir(exist_ok=True)
    (selected / "unexpected.py").write_text("CHANGED = True\n")
    destination = tmp_path / "snapshot"
    with pytest.raises(SNAP.SnapshotError, match="source selection membership changed"):
        SNAP.create(
            source,
            destination,
            output_root=tmp_path / "output",
            source_roots=("python",),
            python_roots=("python",),
            legacy_roots=(),
            directory_memberships=expected,
        )
    assert not destination.exists()


def test_fresh_snapshot_refuses_modified_source_bytes(snapshot):
    receipt = SNAP.verify(snapshot)
    assert receipt["schema"] == "merlin.performance-source-snapshot.v4"
    path = snapshot / "python/toy/__init__.py"
    path.chmod(0o644)
    path.write_text("VALUE = 2\n")
    path.chmod(0o444)
    with pytest.raises(SNAP.SnapshotError, match="source changed"):
        SNAP.verify(snapshot)


@pytest.mark.parametrize("version", [1, 2, 3])
def test_legacy_bytes_remain_inspectable_but_cannot_authorize_execution(snapshot, version):
    marker, receipt = SNAP.load_seal(snapshot, "snapshot")
    receipt["schema"] = f"merlin.performance-source-snapshot.v{version}"
    snapshot.chmod(0o700)
    marker.unlink()
    marker = SNAP.seal(snapshot, "snapshot", receipt)
    snapshot.chmod(0o500)
    original = marker.read_bytes()
    for admission in (
        SNAP.verify,
        lambda root: SNAP.import_layout(root, receipt),
        lambda root: SNAP.provider_environment(root, receipt),
    ):
        with pytest.raises(SNAP.SnapshotError, match="newly frozen"):
            admission(snapshot)
    assert marker.read_bytes() == original
    assert SNAP.load_seal(snapshot, "snapshot")[1] == receipt


def test_owner_cold_loads_and_verifies_without_native_or_core_imports(snapshot, tmp_path):
    program = """
import importlib.abc, importlib.util, json, pathlib, subprocess, sys
def forbidden(*a, **kw):
    raise AssertionError('snapshot owner attempted process launch')
subprocess.Popen = forbidden
class NoMerlin(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'merlin', 'merlin_experiments', 'perf_snapshot', '_pbcommon'}:
            raise AssertionError('nonstdlib snapshot dependency: ' + name)
sys.meta_path.insert(0, NoMerlin())
spec = importlib.util.spec_from_file_location('stdlib_snapshot_authority', sys.argv[1])
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)
receipt = owner.verify(pathlib.Path(sys.argv[2]))
assert receipt['schema'] == 'merlin.performance-source-snapshot.v4'
print(json.dumps(receipt['python_roots']))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            program,
            str(module_source_path("merlin_experiments.source_snapshot")),
            str(snapshot),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["python"]
