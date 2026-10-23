"""Live Phase1 provider attribution, not frozen-provider execution qualification."""

import shutil
import socket
import subprocess

import pytest
from merlin_experiments.phase1 import source_inputs as SI
from merlin_experiments.spec import SpecError


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("provider source admission must not execute processes or bind listeners")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(socket.socket, "bind", refuse)


@pytest.fixture
def selected(tmp_path, monkeypatch):
    root = tmp_path / "support"
    (root / "contracts").mkdir(parents=True)
    (root / "backend").mkdir()
    (root / "contracts/target_contract.yaml").write_text("name: source_fixture\nplugin:\n  backend: backend\n")
    (root / "provider.yaml").write_text(
        "schema: merlin.provider.v1\nid: fixture-support\ntarget: source_fixture\n"
        "role: support\ncontract: contracts/target_contract.yaml\n"
    )
    (root / "backend/__init__.py").write_text("raise AssertionError('must not import provider')\n")
    (root / "backend/rtl_checks.py").write_text("CHECKS = ('one',)\n")
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("target: source_fixture\n")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root))
    monkeypatch.delenv("MERLIN_TARGET_CONTRACT", raising=False)
    arguments = {"repo": tmp_path, "entrypoint": tmp_path / "installed.py", "descriptor": descriptor}
    return root, arguments


def test_complete_membership_and_identity(selected):
    root, arguments = selected
    record = SI.record(**arguments)
    inputs = record["inputs"]
    for relative in ("backend/__init__.py", "backend/rtl_checks.py"):
        assert inputs[f"phase1:startup:provider:python:{relative}"]["path"] == str(root / relative)
    assert inputs["phase1:startup:provider:identity"]["sha256"] == SI.fingerprint(root / "provider.yaml")
    assert inputs["phase1:startup:provider:contract"]["path"] == str(root / "contracts/target_contract.yaml")
    SI.verify(record, **arguments)


@pytest.mark.parametrize("damage", ["bytes", "added", "removed", "identity", "contract", "selection", "unselected"])
def test_resume_refuses_provider_drift(selected, monkeypatch, damage):
    root, arguments = selected
    record = SI.record(**arguments)
    if damage == "selection":
        replacement = root.with_name("replacement")
        shutil.copytree(root, replacement)
        monkeypatch.setenv("MERLIN_TARGET_PATH", str(replacement))
    elif damage == "unselected":
        monkeypatch.setenv("MERLIN_TARGET_PATH", "")
    elif damage == "removed":
        (root / "backend/rtl_checks.py").unlink()
    elif damage == "added":
        (root / "new_checks.py").write_text("NEW = True\n")
    else:
        relative = {
            "bytes": "backend/rtl_checks.py",
            "identity": "provider.yaml",
            "contract": "contracts/target_contract.yaml",
        }[damage]
        path = root / relative
        path.write_text(path.read_text() + "\n# changed\n")
    with pytest.raises(SpecError, match="source identity changed"):
        SI.verify(record, **arguments)


def test_adding_selection_to_unselected_receipt_refuses(selected, monkeypatch):
    root, arguments = selected
    monkeypatch.setenv("MERLIN_TARGET_PATH", "")
    record = SI.record(**arguments)
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root))
    with pytest.raises(SpecError, match="source identity changed"):
        SI.verify(record, **arguments)


def test_linked_provider_member_refuses(selected):
    root, arguments = selected
    (root / "linked.py").symlink_to(root / "backend/rtl_checks.py")
    with pytest.raises(SpecError, match="symlinked"):
        SI.record(**arguments)


def test_loaded_plugin_selection_guard_precedes_new_record(selected, monkeypatch):
    from merlin.runtime.backends import base

    root, arguments = selected
    owner = base._PluginOwner("source_fixture", "merlin._oot_backends", root, root / "backend/__init__.py")
    monkeypatch.setattr(base, "_LOADED_PLUGIN_OWNERS", {"fixture": owner})
    SI.record(**arguments)
    replacement = root.with_name("replacement")
    shutil.copytree(root, replacement)
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(replacement))
    with pytest.raises(SpecError, match="loaded plugin"):
        SI.record(**arguments)


def test_selected_checks_in_existing_support_mask(selected):
    from merlin.targetgen.sandbox.answer_surfaces import _support_package_dirs

    root, _ = selected
    assert root in _support_package_dirs()
    assert (root / "backend/rtl_checks.py").is_relative_to(root)


def test_selected_checks_in_existing_snapshot_seal(selected, tmp_path):
    from merlin_experiments import source_snapshot as snapshots

    root, _ = selected
    source = tmp_path / "source"
    (source / "python").mkdir(parents=True)
    (source / "python/fixture.py").write_text("VALUE = 1\n")
    frozen = tmp_path / "frozen"
    snapshots.create(
        source,
        frozen,
        output_root=tmp_path / "output",
        source_roots=("python",),
        python_roots=("python",),
        legacy_roots=(),
        target_name="source_fixture",
        provider={
            "target": "source_fixture",
            "resolved_target": "source_fixture",
            "kind": "external",
            "source": str(root),
        },
    )
    snapshots.verify(frozen)
    copied = frozen / "_selected_provider/backend/rtl_checks.py"
    assert copied.read_bytes() == (root / "backend/rtl_checks.py").read_bytes()
    copied.chmod(0o600)
    copied.write_text("CHECKS = ('tampered',)\n")
    with pytest.raises(snapshots.SnapshotError):
        snapshots.verify(frozen)


@pytest.fixture
def selected_dependency(selected):
    root, arguments = selected
    dependency = root.with_name("compiler_sources")
    dependency.mkdir()
    (dependency / "checker.py").write_text("raise AssertionError('private dependency must not import')\n")
    (dependency / "layout.json").write_text('{"layout": "reviewed"}\n')
    metadata = root / "provider.yaml"
    metadata.write_text(metadata.read_text() + "source_dependencies:\n  compiler: ../compiler_sources\n")
    return dependency, arguments


def test_dependency_python_and_resource_bytes_are_bound(selected_dependency):
    dependency, arguments = selected_dependency
    receipt = SI.record(**arguments)
    for relative in ("checker.py", "layout.json"):
        assert receipt["inputs"][f"phase1:startup:provider:dependency:compiler:{relative}"] == {
            "path": str(dependency / relative),
            "sha256": SI.fingerprint(dependency / relative),
        }
    cache = dependency / "__pycache__"
    cache.mkdir()
    (cache / "checker.cpython-313.pyc").write_bytes(b"cache")
    SI.verify(receipt, **arguments)


@pytest.mark.parametrize("damage", ["python", "resource", "added", "removed", "link", "missing_owner"])
def test_resume_refuses_dependency_member_or_owner_drift(selected_dependency, damage):
    dependency, arguments = selected_dependency
    receipt = SI.record(**arguments)
    if damage in {"python", "resource"}:
        relative = "checker.py" if damage == "python" else "layout.json"
        (dependency / relative).write_text("changed\n")
    elif damage == "added":
        (dependency / "new_resource.bin").write_bytes(b"new")
    elif damage == "removed":
        (dependency / "checker.py").unlink()
    elif damage == "link":
        (dependency / "alias").symlink_to(dependency / "layout.json")
    else:
        shutil.rmtree(dependency)
    with pytest.raises(SpecError, match="source identity changed|source admission failed"):
        SI.verify(receipt, **arguments)
