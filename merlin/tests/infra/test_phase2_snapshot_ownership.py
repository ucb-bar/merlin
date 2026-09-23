"""A sealed Phase 2 snapshot must supply relocated Python owners without live fallback."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from merlin.common.paths import repo_root

SOURCE_ROOTS = (
    "src",
    "packages/merlin-experiments/src",
    "merlin/contract",
    "merlin/targets",
    "merlin/schemas",
    "build_tools",
)


def _snapshot_module():
    from merlin_experiments import source_snapshot

    return source_snapshot


def _write(root: Path, relative: str, text: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def source_tree(tmp_path):
    source = tmp_path / "source"
    for relative in SOURCE_ROOTS:
        if relative != "merlin/python":
            (source / relative).mkdir(parents=True, exist_ok=True)
    namespace = "from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n"
    _write(source, "src/merlin/__init__.py", namespace)
    helper = repo_root() / "src/merlin/common/frozen_imports.py"
    _write(source, "src/merlin/common/frozen_imports.py", helper.read_text())
    _write(source, "src/merlin/targetgen/__init__.py", namespace)
    _write(source, "src/merlin/targetgen/contract/build_service.py", "IDENTITY = 'core-frozen'\n")
    (source / "merlin/python").symlink_to(source / "src", target_is_directory=True)
    _write(
        source,
        "packages/merlin-experiments/src/merlin/targetgen/capsule_grade.py",
        "IDENTITY = 'grader-frozen'\n",
    )
    _write(source, "packages/merlin-experiments/src/merlin_experiments/__init__.py", "")
    _write(
        source,
        "packages/merlin-experiments/src/merlin_experiments/corpus/admission.py",
        "IDENTITY = 'workflow-frozen'\n",
    )
    yield source
    # The real snapshot seal makes directories read-only. Restore only this
    # fixture's scratch snapshots so pytest can remove its temporary tree.
    for path in [tmp_path, *tmp_path.rglob("*")]:
        if not path.is_symlink() and path.is_dir():
            path.chmod(0o700)


def test_snapshot_imports_optional_owners_without_live_packages(source_tree, tmp_path):
    snapshot = tmp_path / "snapshot"
    module = _snapshot_module()
    module.create(
        source_tree,
        snapshot,
        output_root=tmp_path / "outputs",
        source_roots=SOURCE_ROOTS,
        python_roots=("packages/merlin-experiments/src", "src"),
        legacy_roots=(),
        internal_aliases={"merlin/python": "src"},
    )
    receipt = module.verify(snapshot)
    # The test launches no target code: these are three-line synthetic modules
    # with the real ownership layout. -I -S disables PYTHONPATH, site packages,
    # editable finders and sitecustomize before adding ONLY snapshot-owned roots.
    program = """
import importlib, importlib.util, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
receipt = json.loads(sys.argv[2])
spec = importlib.util.spec_from_file_location('trusted_frozen_imports', root / 'src/merlin/common/frozen_imports.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
guard.activate(snapshot_root=root, import_roots=[root / relative for relative in receipt['python_roots']],
               sources=receipt['files'], legacy_names=receipt['legacy_names'])
names = ('merlin.targetgen.capsule_grade', 'merlin_experiments.corpus.admission')
records = {}
for name in names:
    module = importlib.import_module(name)
    origin = pathlib.Path(module.__file__).resolve()
    assert origin.is_relative_to(root), (name, origin)
    records[name] = module.IDENTITY
print(json.dumps(records, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", program, str(snapshot), json.dumps(receipt)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "merlin.targetgen.capsule_grade": "grader-frozen",
        "merlin_experiments.corpus.admission": "workflow-frozen",
    }


def test_snapshot_preserves_canonical_core_policy_source_location(source_tree, tmp_path):
    snapshot = tmp_path / "snapshot"
    module = _snapshot_module()
    module.create(
        source_tree,
        snapshot,
        output_root=tmp_path / "outputs",
        source_roots=SOURCE_ROOTS,
        python_roots=("packages/merlin-experiments/src", "src"),
        legacy_roots=(),
        internal_aliases={"merlin/python": "src"},
    )
    receipt = module.verify(snapshot)
    # host_verification_policy_record currently resolves its core source pins
    # beneath repo_root()/src/merlin, not the old merlin/python spelling.
    relative = "src/merlin/targetgen/contract/build_service.py"
    assert relative in receipt["files"]
    assert (snapshot / relative).read_bytes() == (source_tree / relative).read_bytes()


def test_snapshot_preserves_complete_phase2_host_policy_sources(source_tree, tmp_path):
    from merlin.common.paths import module_source_path
    from merlin.common.source_membership import python_members

    package = module_source_path("merlin_experiments.phase2").parent
    members = list(python_members(package).values())
    members.append(module_source_path("merlin.common.source_membership"))
    relative_sources = [path.relative_to(repo_root()).as_posix() for path in members]
    for path, relative in zip(members, relative_sources, strict=True):
        _write(source_tree, relative, path.read_text())
    module = _snapshot_module()
    snapshot = tmp_path / "snapshot"
    module.create(
        source_tree,
        snapshot,
        output_root=tmp_path / "outputs",
        source_roots=SOURCE_ROOTS,
        python_roots=("packages/merlin-experiments/src", "src"),
        legacy_roots=(),
        internal_aliases={"merlin/python": "src"},
    )
    receipt = module.verify(snapshot)
    for relative in relative_sources:
        assert relative in receipt["files"]
        assert (snapshot / relative).read_bytes() == (source_tree / relative).read_bytes()
        assert (snapshot / relative).stat().st_ino != (source_tree / relative).stat().st_ino


def test_snapshot_alias_is_internal_without_duplicate_core_bytes(source_tree, tmp_path):
    module = _snapshot_module()
    snapshot = tmp_path / "snapshot"
    module.create(
        source_tree,
        snapshot,
        output_root=tmp_path / "outputs",
        source_roots=SOURCE_ROOTS,
        python_roots=("packages/merlin-experiments/src", "src"),
        legacy_roots=(),
        internal_aliases={"merlin/python": "src"},
    )
    receipt = module.verify(snapshot)
    assert receipt["internal_aliases"] == {"merlin/python": "src"}
    assert (snapshot / "merlin/python").resolve() == snapshot / "src"
    assert not any(path.startswith("merlin/python/") for path in receipt["files"])
    for relative in receipt["files"]:
        copied, original = snapshot / relative, source_tree / relative
        assert copied.stat().st_ino != original.stat().st_ino
        assert copied.stat().st_nlink == 1
    original.write_text("live change\n")
    assert module.verify(snapshot) == receipt


@pytest.mark.parametrize(
    "change", ["external_alias", "missing_alias", "extra_directory", "python_roots", "legacy_names"]
)
def test_snapshot_refuses_alias_and_import_inventory_drift(source_tree, tmp_path, change):
    module = _snapshot_module()
    snapshot = tmp_path / "snapshot"
    seal = module.create(
        source_tree,
        snapshot,
        output_root=tmp_path / "outputs",
        source_roots=SOURCE_ROOTS,
        python_roots=("packages/merlin-experiments/src", "src"),
        legacy_roots=(),
        internal_aliases={"merlin/python": "src"},
    )
    receipt = module.verify(snapshot)
    snapshot.chmod(0o700)
    if change in {"external_alias", "missing_alias"}:
        (snapshot / "merlin").chmod(0o700)
        (snapshot / "merlin/python").unlink()
        if change == "external_alias":
            (snapshot / "merlin/python").symlink_to(source_tree / "src", target_is_directory=True)
        (snapshot / "merlin").chmod(0o500)
    elif change == "extra_directory":
        (snapshot / "extra").mkdir(mode=0o500)
    else:
        receipt[change] = ["live_sources"]
        seal.unlink()
        module.seal(snapshot, "snapshot", receipt)
    snapshot.chmod(0o500)
    with pytest.raises(module.SnapshotError):
        module.verify(snapshot)


@pytest.mark.parametrize("roots", [("../escape",), ("src", "src/merlin"), ("src", "src"), ("out",), ("/tmp",)])
def test_snapshot_refuses_invalid_roots_before_creating_destination(source_tree, tmp_path, roots):
    module = _snapshot_module()
    destination = tmp_path / "snapshot"
    with pytest.raises(module.SnapshotError):
        module.create(
            source_tree,
            destination,
            output_root=tmp_path / "outputs",
            source_roots=roots,
            python_roots=(),
            legacy_roots=(),
        )
    assert not destination.exists()


def test_snapshot_refuses_destination_inside_source_tree(source_tree):
    module = _snapshot_module()
    destination = source_tree / "src/nested_snapshot"
    with pytest.raises(module.SnapshotError, match="overlaps"):
        module.create(
            source_tree,
            destination,
            output_root=source_tree / "out",
            source_roots=SOURCE_ROOTS,
            python_roots=("packages/merlin-experiments/src", "src"),
            legacy_roots=(),
            internal_aliases={"merlin/python": "src"},
        )
    assert not destination.exists()


def test_snapshot_refuses_source_membership_changes_during_copy(source_tree, tmp_path, monkeypatch):
    module = _snapshot_module()
    original = module.shutil.copy2

    def copy_and_add(source, destination):
        result = original(source, destination)
        _write(source_tree, "src/new_after_enumeration.py", "unexpected = True\n")
        return result

    monkeypatch.setattr(module.shutil, "copy2", copy_and_add)
    with pytest.raises(module.SnapshotError, match="membership changed"):
        module.create(
            source_tree,
            tmp_path / "snapshot",
            output_root=tmp_path / "outputs",
            source_roots=SOURCE_ROOTS,
            python_roots=("packages/merlin-experiments/src", "src"),
            legacy_roots=(),
            internal_aliases={"merlin/python": "src"},
        )


def test_snapshot_legacy_checkout_alias_preserves_runtime_data_without_capsule_mirror(source_tree, tmp_path):
    module = _snapshot_module()
    _write(source_tree, "src/merlin/_data/schemas/registry.yaml", "formats: {}\n")
    _write(source_tree, "src/merlin/_data/contract/abi/header.h", "// ABI\n")
    _write(source_tree, "src/merlin/_data/contract/capsules/private/weights", "do not mirror\n")
    snapshot = tmp_path / "snapshot"
    module.create(
        source_tree,
        snapshot,
        output_root=tmp_path / "outputs",
        source_roots=("merlin/python",),
        exclude_paths=("merlin/python/merlin/_data/contract/capsules",),
        python_roots=("merlin/python",),
        legacy_roots=(),
    )
    receipt = module.verify(snapshot)
    assert "merlin/python/merlin/_data/schemas/registry.yaml" in receipt["files"]
    assert "merlin/python/merlin/_data/contract/abi/header.h" in receipt["files"]
    assert not (snapshot / "merlin/python/merlin/_data/contract/capsules").exists()


def test_legacy_receipt_is_read_without_rewriting_or_adding_live_owners(source_tree, tmp_path):
    module = _snapshot_module()
    root = tmp_path / "legacy"
    source = _write(root, "merlin/python/merlin/__init__.py", "")
    source.chmod(0o444)
    receipt = {
        "schema": "merlin.performance-source-snapshot.v1",
        "source_root": str(source_tree),
        "source_roots": ["merlin/python"],
        "files": {"merlin/python/merlin/__init__.py": module.sha_file(source)},
        "external_links": {},
    }
    seal = module.seal(root, "snapshot", receipt)
    original = seal.read_bytes()
    assert module.load_seal(root, "snapshot")[1] == receipt
    with pytest.raises(module.SnapshotError, match="newly frozen"):
        module.verify(root)
    with pytest.raises(module.SnapshotError, match="newly frozen"):
        module.import_layout(root, receipt)
    assert seal.read_bytes() == original


def test_snapshot_refuses_nonregular_replacement_without_reading_it(source_tree, tmp_path, monkeypatch):
    module = _snapshot_module()
    snapshot = tmp_path / "snapshot"
    module.create(
        source_tree,
        snapshot,
        output_root=tmp_path / "outputs",
        source_roots=SOURCE_ROOTS,
        python_roots=("packages/merlin-experiments/src", "src"),
        legacy_roots=(),
        internal_aliases={"merlin/python": "src"},
    )
    path = snapshot / "src/merlin/targetgen/contract/build_service.py"
    path.parent.chmod(0o700)
    path.unlink()
    os.mkfifo(path, 0o444)
    path.parent.chmod(0o500)
    original = module.sha_file

    def refuse_fifo_read(candidate):
        assert candidate != path, "verification must reject FIFO before attempting to hash it"
        return original(candidate)

    monkeypatch.setattr(module, "sha_file", refuse_fifo_read)
    with pytest.raises(module.SnapshotError, match="source changed"):
        module.verify(snapshot)


@pytest.mark.parametrize("key, value", [("files", {"unowned/code.py": "0" * 64}), ("directories", ["unowned"])])
def test_snapshot_rejects_resealed_inventory_outside_declared_roots(source_tree, tmp_path, key, value):
    module = _snapshot_module()
    snapshot = tmp_path / "snapshot"
    seal = module.create(
        source_tree,
        snapshot,
        output_root=tmp_path / "outputs",
        source_roots=SOURCE_ROOTS,
        python_roots=("packages/merlin-experiments/src", "src"),
        legacy_roots=(),
        internal_aliases={"merlin/python": "src"},
    )
    receipt = module.verify(snapshot)
    if key == "files":
        receipt[key].update(value)
    else:
        receipt[key] = sorted(receipt[key] + value)
    snapshot.chmod(0o700)
    seal.unlink()
    module.seal(snapshot, "snapshot", receipt)
    snapshot.chmod(0o500)
    with pytest.raises(module.SnapshotError, match="outside declared source roots"):
        module.verify(snapshot)


def test_snapshot_guard_never_uses_live_missing_optional_owner(source_tree, tmp_path):
    module = _snapshot_module()
    relative = "packages/merlin-experiments/src/merlin/targetgen/capsule_grade.py"
    (source_tree / relative).unlink()
    snapshot = tmp_path / "snapshot"
    module.create(
        source_tree,
        snapshot,
        output_root=tmp_path / "outputs",
        source_roots=SOURCE_ROOTS,
        python_roots=("packages/merlin-experiments/src", "src"),
        legacy_roots=(),
        internal_aliases={"merlin/python": "src"},
    )
    receipt = module.verify(snapshot)
    live = tmp_path / "live"
    _write(live, "merlin/targetgen/capsule_grade.py", "raise AssertionError('live code executed')\n")
    _write(live, "merlin_analysis/__init__.py", "raise AssertionError('live optional owner executed')\n")
    program = """
import importlib, importlib.util, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
receipt = json.loads(sys.argv[2])
sys.path.insert(0, sys.argv[3])
spec = importlib.util.spec_from_file_location('trusted_guard', root / 'src/merlin/common/frozen_imports.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
guard.activate(snapshot_root=root, import_roots=[root / p for p in receipt['python_roots']], sources=receipt['files'])
for name in ('merlin_analysis', 'merlin.targetgen.capsule_grade'):
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as exc:
        assert 'frozen source receipt' in str(exc), str(exc)
    else:
        raise AssertionError('missing owner fell through to live packages')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", program, str(snapshot), json.dumps(receipt), str(live)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
