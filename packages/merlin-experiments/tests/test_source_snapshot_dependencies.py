"""Trusted support sibling sources are frozen without live-provider imports."""

import importlib.util
import subprocess
import sys
from types import SimpleNamespace

import pytest
from merlin_experiments import source_snapshot as SNAP


@pytest.fixture
def selected(tmp_path, monkeypatch):
    from merlin.targetgen import target_registry

    source = tmp_path / "source"
    (source / "python").mkdir(parents=True)
    (source / "python/main.py").write_text("VALUE = 1\n")
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
        "from pathlib import Path\nVALUE = (Path(__file__).parent.parent/'compiler/helper.py').read_text()\n"
    )
    (compiler / "helper.py").write_text("HELPER = 'frozen'\n")
    monkeypatch.setattr(
        target_registry, "resolve", lambda target: SimpleNamespace(kind="external", name=target, base=support)
    )
    frozen = tmp_path / "snapshot"
    yield source, support, compiler, frozen
    if frozen.exists():
        for path in (frozen, *frozen.rglob("*")):
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o700)


def create(selected, **kwargs):
    source, support, compiler, frozen = selected
    SNAP.create(
        source,
        frozen,
        output_root=source.parent / "output",
        source_roots=("python",),
        python_roots=("python",),
        legacy_roots=(),
        target_name="toy",
        provider=SNAP.selected_provider("toy"),
        runtime_links=False,
        **kwargs,
    )
    return SNAP.verify(frozen)


def test_sibling_dependencies_preserve_provider_layout_and_remap_inputs(selected):
    source, support, compiler, frozen = selected
    declaration = SNAP.selected_provider("toy")
    assert declaration["source_dependencies"] == {"compiler": str(compiler)}
    receipt = create(selected, declared_inputs={"helper": compiler / "helper.py"})
    provider = receipt["selected_provider"]
    assert provider["root"] == "_selected_provider/support"
    assert provider["source_dependencies"] == {
        "compiler": {"source": str(compiler), "snapshot": "_selected_provider/compiler"}
    }
    assert SNAP.provider_environment(frozen, receipt)["MERLIN_TARGET_PATH"] == str(frozen / provider["root"])
    helper = SNAP.remap_input(frozen, receipt, compiler / "helper.py")
    assert helper == frozen / "_selected_provider/compiler/helper.py"
    assert SNAP.remap_input(frozen, receipt, compiler / "helper.py", name="helper") == helper
    assert "_declared_inputs" not in receipt["source_roots"]
    assert (frozen / provider["root"] / "provider.yaml").read_bytes() == (support / "provider.yaml").read_bytes()
    assert helper.stat().st_mode & 0o222 == 0
    (compiler / "helper.py").write_text("LIVE_CHANGED = True\n")
    (support / "hook.py").write_text("raise AssertionError('live provider')\n")
    hook = frozen / provider["root"] / "hook.py"
    spec = importlib.util.spec_from_file_location("frozen_test_hook", hook)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    assert loaded.VALUE == "HELPER = 'frozen'\n"
    assert SNAP.verify(frozen) == receipt


def test_legacy_provider_shape_and_layout_are_unchanged(selected):
    source, support, compiler, frozen = selected
    (support / "provider.yaml").write_text("schema: merlin.provider.v1\nid: toy-support\ntarget: toy\nrole: support\n")
    assert set(SNAP.selected_provider("toy")) == {"target", "resolved_target", "kind", "source"}
    receipt = create(selected)
    assert receipt["selected_provider"]["root"] == "_selected_provider"
    assert "source_dependencies" not in receipt["selected_provider"]
    assert "_selected_provider/compiler/helper.py" not in receipt["files"]


def test_reference_provider_dependencies_are_refused_not_omitted(selected, monkeypatch):
    from merlin.targetgen import target_registry

    _, support, _, _ = selected
    monkeypatch.setattr(
        target_registry, "resolve", lambda target: SimpleNamespace(kind="reference", name=target, base=support)
    )
    with pytest.raises(SNAP.SnapshotError, match="reference provider sibling dependencies"):
        SNAP.selected_provider("toy")


@pytest.mark.parametrize("change", ["missing", "added", "symlink", "writable", "changed"])
def test_dependency_inventory_is_immutable(selected, change):
    _, _, _, frozen = selected
    create(selected)
    folder = frozen / "_selected_provider/compiler"
    path = folder / "helper.py"
    folder.chmod(0o700)
    if change == "missing":
        path.unlink()
    if change == "added":
        (folder / "unexpected.py").write_text("EXTRA = True\n")
    if change == "symlink":
        path.unlink()
        path.symlink_to(folder.parent / "support/hook.py")
    if change == "writable":
        path.chmod(0o644)
    if change == "changed":
        path.chmod(0o644)
        path.write_text("CHANGED = True\n")
        path.chmod(0o444)
    folder.chmod(0o500)
    with pytest.raises(SNAP.SnapshotError):
        SNAP.verify(frozen)


@pytest.mark.parametrize("change", ["symlink_resource", "symlink_root", "missing_root", "overlap", "non_sibling"])
def test_creation_refuses_indirect_or_undeclared_dependency_ownership(selected, change):
    source, support, compiler, frozen = selected
    provider = SNAP.selected_provider("toy")
    if change == "symlink_resource":
        (compiler / "escape.py").symlink_to(source / "python/main.py")
    if change == "symlink_root":
        moved = compiler.with_name("original")
        compiler.rename(moved)
        compiler.symlink_to(moved, target_is_directory=True)
    if change == "missing_root":
        (compiler / "helper.py").unlink()
        compiler.rmdir()
    if change == "overlap":
        provider["source_dependencies"]["compiler"] = str(support)
    if change == "non_sibling":
        provider["source_dependencies"]["compiler"] = str(source / "python")
    with pytest.raises(SNAP.SnapshotError):
        SNAP.create(
            source,
            frozen,
            output_root=source.parent / "output",
            source_roots=("python",),
            python_roots=("python",),
            legacy_roots=(),
            target_name="toy",
            provider=provider,
        )


def test_dependency_receipt_verification_is_stdlib_only(selected, tmp_path):
    _, _, _, frozen = selected
    create(selected)
    program = """
import importlib.abc, importlib.util, pathlib, sys
class Forbidden(importlib.abc.MetaPathFinder):
 def find_spec(self, name, *args):
  if name.split('.')[0] in {'merlin', 'merlin_experiments', 'yaml', 'xdsl'}:
   raise AssertionError('nonstdlib snapshot dependency: ' + name)
sys.meta_path.insert(0, Forbidden())
spec = importlib.util.spec_from_file_location('snapshot_authority', sys.argv[1])
owner = importlib.util.module_from_spec(spec); spec.loader.exec_module(owner)
receipt = owner.verify(pathlib.Path(sys.argv[2]))
assert receipt['selected_provider']['source_dependencies']['compiler']['snapshot'] == '_selected_provider/compiler'
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", program, SNAP.__file__, str(frozen)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("change", ["different_source", "different_snapshot", "empty", "extra_owner"])
def test_sealed_dependency_layout_cannot_claim_unowned_sources(selected, change):
    _, _, compiler, frozen = selected
    receipt = create(selected)
    marker, _ = SNAP.load_seal(frozen, "snapshot")
    dependencies = receipt["selected_provider"]["source_dependencies"]
    if change == "different_source":
        dependencies["compiler"]["source"] = str(compiler.parent / "unowned")
    if change == "different_snapshot":
        dependencies["compiler"]["snapshot"] = "python"
    if change == "empty":
        dependencies.clear()
    if change == "extra_owner":
        dependencies["unowned"] = {"source": str(compiler.parent / "unowned"), "snapshot": "_selected_provider/unowned"}
    frozen.chmod(0o700)
    marker.unlink()
    SNAP.seal(frozen, "snapshot", receipt)
    frozen.chmod(0o500)
    with pytest.raises(SNAP.SnapshotError):
        SNAP.verify(frozen)


@pytest.mark.parametrize("change", ["added_file", "missing_file", "root_symlink"])
def test_dependency_source_membership_changes_during_copy_refuse_seal(selected, monkeypatch, change):
    source, support, compiler, frozen = selected
    original = SNAP._copy_tree

    def changed(tree, destination, ignore, ancestors=frozenset()):
        original(tree, destination, ignore, ancestors)
        if tree == compiler:
            if change == "added_file":
                (compiler / "new.py").write_text("NEW = 1\n")
            if change == "missing_file":
                (compiler / "helper.py").unlink()
            if change == "root_symlink":
                moved = compiler.with_name("moved")
                compiler.rename(moved)
                compiler.symlink_to(moved, target_is_directory=True)

    monkeypatch.setattr(SNAP, "_copy_tree", changed)
    with pytest.raises((SNAP.SnapshotError, OSError)):
        create(selected)
    assert not list(frozen.glob("snapshot.*.json"))
