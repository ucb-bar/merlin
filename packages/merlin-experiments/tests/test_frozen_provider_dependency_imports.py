"""Cold guarded transport imports only sealed support and declared sibling sources."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from merlin_experiments import frozen_python
from merlin_experiments import source_snapshot as SNAP

from merlin.common.paths import module_source_path


@pytest.fixture
def owned(tmp_path):
    source = tmp_path / "source"
    python = source / "python"
    for package in ("merlin", "merlin/targetgen", "merlin_experiments"):
        folder = python / package
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "__init__.py").write_text("")
    (python / "merlin/targetgen/plugins.py").write_bytes(module_source_path("merlin.targetgen.plugins").read_bytes())
    (python / "merlin_experiments/frozen_python.py").write_bytes(Path(frozen_python.__file__).read_bytes())
    support = tmp_path / "companion/support"
    compiler = support.parent / "compiler"
    (support / "contracts").mkdir(parents=True)
    compiler.mkdir()
    (support / "contracts/target_contract.yaml").write_text("name: toy\n")
    (support / "provider.yaml").write_text(
        "schema: merlin.provider.v1\nid: toy-support\ntarget: toy\nrole: support\n"
        "source_dependencies:\n  compiler: ../compiler\n"
    )
    (support / "hook.py").write_text(
        "from pathlib import Path\nfrom merlin.targetgen.plugins import load_module\n"
        "def probe():\n"
        " module = load_module(Path(__file__).parent.parent/'compiler', 'helper', package_name='owned_helper')\n"
        " return {'value':module.VALUE, 'origin':module.__file__}\n"
    )
    (compiler / "helper.py").write_text("from .sibling import VALUE\n")
    (compiler / "sibling.py").write_text("VALUE = 'sealed-owner'\n")
    frozen = tmp_path / "snapshot"
    SNAP.create(
        source,
        frozen,
        output_root=tmp_path / "output",
        source_roots=("python",),
        python_roots=("python",),
        legacy_roots=(),
        runtime_links=False,
        target_name="toy",
        provider={
            "target": "toy",
            "resolved_target": "toy",
            "kind": "external",
            "source": str(support),
            "source_dependencies": {"compiler": str(compiler)},
        },
    )
    yield frozen, support, compiler
    for path in (frozen, *frozen.rglob("*")):
        if not path.is_symlink():
            path.chmod(0o700 if path.is_dir() else 0o600)


PROGRAM = """
import importlib.util,json,os,sys
from pathlib import Path
from merlin.targetgen.plugins import load_module
support=Path(os.environ['MERLIN_TARGET_PATH'])
hook=load_module(support,'hook',package_name='owned_support')
result=hook.probe()
assert str(support.parent/'compiler') not in sys.path
assert importlib.util.find_spec('helper') is None
print(json.dumps(result))
"""


def command(frozen, program=PROGRAM):
    return frozen_python.python_command(frozen, [sys.executable, "-c", program], verifier_source=Path(SNAP.__file__))


def execute(argv):
    return subprocess.run(argv, capture_output=True, text=True, timeout=20)


def test_cold_dependency_import_is_sealed_and_not_a_general_import_root(owned):
    frozen, support, compiler = owned
    (support / "hook.py").write_text("raise AssertionError('live support')\n")
    (compiler / "helper.py").write_text("raise AssertionError('live compiler')\n")
    result = execute(command(frozen))
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["value"] == "sealed-owner"
    assert Path(data["origin"]).is_relative_to(frozen / "_selected_provider/compiler")


def test_cold_container_catalog_import_uses_only_sealed_dependency(owned):
    frozen, _, _ = owned
    program = (
        PROGRAM
        + """
catalog=load_module(support.parent,'compiler.helper',package_name='owned_catalog')
assert catalog.VALUE == 'sealed-owner'
assert Path(catalog.__file__).is_relative_to(support.parent/'compiler')
assert str(support.parent) not in sys.path
assert importlib.util.find_spec('compiler') is None
"""
    )
    result = execute(command(frozen, program))
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("extra", ["file", "directory"])
def test_container_members_cannot_expand_beyond_declared_owners(owned, extra):
    frozen, _, _ = owned
    argv = command(frozen)
    container = frozen / "_selected_provider"
    container.chmod(0o700)
    if extra == "file":
        (container / "unowned.py").write_text("raise AssertionError('extra owner executed')\n")
        (container / "unowned.py").chmod(0o400)
    else:
        (container / "unowned").mkdir()
        (container / "unowned/__init__.py").write_text("raise AssertionError('extra owner executed')\n")
        (container / "unowned/__init__.py").chmod(0o400)
        (container / "unowned").chmod(0o500)
    container.chmod(0o500)
    result = execute(argv)
    assert result.returncode != 0
    assert "SnapshotError" in result.stderr
    assert "extra owner executed" not in result.stderr


@pytest.mark.parametrize("change", ["changed", "missing", "added", "symlink", "writable"])
def test_dependency_drift_refuses_at_cold_boot_before_import(owned, change):
    frozen, _, _ = owned
    argv = command(frozen)
    folder = frozen / "_selected_provider/compiler"
    folder.chmod(0o700)
    helper = folder / "helper.py"
    if change == "changed":
        helper.chmod(0o600)
        helper.write_text("raise AssertionError('unsealed code executed')\n")
        helper.chmod(0o400)
    elif change == "missing":
        helper.unlink()
    elif change == "added":
        (folder / "extra.py").write_text("raise AssertionError('unsealed code executed')\n")
        (folder / "extra.py").chmod(0o400)
    elif change == "symlink":
        helper.unlink()
        helper.symlink_to(folder / "sibling.py")
    else:
        helper.chmod(0o600)
    folder.chmod(0o500)
    result = execute(argv)
    assert result.returncode != 0
    assert "SnapshotError" in result.stderr
    assert "unsealed code executed" not in result.stderr


def test_external_evaluated_package_is_not_granted_by_dependency_support(owned, tmp_path):
    frozen, _, _ = owned
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    marker = tmp_path / "candidate-executed"
    (candidate / "helper.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    program = PROGRAM + f"\nload_module(Path({str(candidate)!r}), 'helper', package_name='unadmitted_candidate')\n"
    result = execute(command(frozen, program))
    assert result.returncode != 0
    assert "module unavailable in frozen source receipt" in result.stderr
    assert not marker.exists()
