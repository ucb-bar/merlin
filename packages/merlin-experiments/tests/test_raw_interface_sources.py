from __future__ import annotations

import hashlib
import json

import pytest
from merlin_experiments.phase1 import raw_interface_sources as readers


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repository"
    for module in readers.MODULES:
        path = repo / "src" / readers._path(module)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'"""Exact source fixture for {module}."""\n')
    return repo


def _selection(repository, tmp_path):
    return readers.prepare(repository, tmp_path / "selection", review_id="reviewed-interface-only")


def _reselect(selection, data):
    selection.manifest_path.chmod(0o644)
    selection.manifest_path.write_text(json.dumps(data, sort_keys=True) + "\n")
    selection.manifest_path.chmod(0o444)
    digest = hashlib.sha256(selection.manifest_path.read_bytes()).hexdigest()
    return readers.select(selection.root, selection.manifest_path, digest)


def test_exact_existing_sources_no_recursive_export(repository, tmp_path):
    hidden = repository / "src/merlin/compile/scheduling/secret.py"
    hidden.parent.mkdir(parents=True)
    hidden.write_text("do_not_supply = True\n")
    selected = _selection(repository, tmp_path)
    assert len(selected.members) == len(readers.MODULES) == 15
    assert selected.diagnostics == readers.UNAVAILABLE
    assert selected.mutable_namespaces == readers.MUTABLE_NAMESPACES
    assert readers.verify_sources(selected) is selected
    for relative, digest, _module in selected.members:
        assert (selected.root / relative).read_bytes() == (repository / "src" / relative).read_bytes()
        assert hashlib.sha256((selected.root / relative).read_bytes()).hexdigest() == digest
        assert not (selected.root / relative).stat().st_mode & 0o222
    assert not (selected.root / "merlin/compile/scheduling").exists()
    document = json.loads(selected.manifest_path.read_text())
    assert document["source_roots"] == []
    assert document["entry_modules"] == list(readers.ENTRIES)
    assert "prelowered inputs" in document["scope"]
    assert "no author-callable frontend lowering" in document["scope"]


@pytest.mark.parametrize(
    "declaration",
    [
        "import merlin.compile.scheduling\n",
        "from merlin.xdsl_dialects import schedule\n",
        "from merlin.targetgen.rtl.facts import load\n",
        "from merlin.targetgen.generate import mlir_scaffold\n",
    ],
)
def test_dependency_drift_rejected_before_destination(repository, tmp_path, declaration):
    path = repository / "src" / readers._path(readers.ENTRIES[0])
    path.write_text(declaration)
    with pytest.raises(ValueError, match="unreviewed dependency"):
        _selection(repository, tmp_path)
    assert not (tmp_path / "selection").exists()


def test_only_documented_optional_dependency_branches_allowed(repository, tmp_path):
    path = repository / "src" / readers._path(readers.ENTRIES[0])
    original = (
        "def optional():\n"
        "    from merlin.perf.storage_encoding import GroupedAxesStorage\n"
        "    return GroupedAxesStorage\n"
    )
    path.write_text(original)
    selected = _selection(repository, tmp_path)
    assert (selected.root / readers._path(readers.ENTRIES[0])).read_text() == original
    assert "merlin.perf.storage_encoding" in selected.diagnostics
    assert not (selected.root / "merlin/perf").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("unavailable_dependencies", []),
        ("unavailable_dependencies", ["merlin.perf.storage_encoding", "merlin.compile.scheduling"]),
        ("entry_modules", ["merlin.targetgen.generate.mlir_scaffold"]),
        ("scope", "full compiler"),
        ("raw_interface_sources_schema", "unreviewed"),
        ("source_roots", ["src/merlin/compile"]),
        ("mutable_namespaces", ["merlin.compile"]),
    ],
)
def test_reselection_cannot_widen_or_misdescribe_inventory(repository, tmp_path, field, value):
    selected = _selection(repository, tmp_path)
    data = json.loads(selected.manifest_path.read_text())
    data[field] = value
    with pytest.raises(ValueError):
        _reselect(selected, data)


def test_unlisted_source_rejected_even_with_correct_hash(repository, tmp_path):
    selected = _selection(repository, tmp_path)
    relative = "merlin/compile/scheduling.py"
    path = selected.root / relative
    path.write_text('"""Undeclared source."""\n')
    data = json.loads(selected.manifest_path.read_text())
    data["members"].append(
        {
            "path": relative,
            "module": "merlin.compile.scheduling",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
    with pytest.raises(ValueError, match="inventory"):
        _reselect(selected, data)


def test_pinned_source_mutation_is_rejected(repository, tmp_path):
    selected = _selection(repository, tmp_path)
    path = selected.root / readers._path(readers.ENTRIES[0])
    path.chmod(0o644)
    path.write_text("changed = True\n")
    with pytest.raises(ValueError, match="baseline changed"):
        readers.verify_sources(selected)


def test_missing_or_indirect_source_is_not_substituted(repository, tmp_path):
    path = repository / "src" / readers._path(readers.ENTRIES[0])
    other = repository / "original.py"
    other.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(other)
    with pytest.raises(ValueError, match="absent or indirect"):
        _selection(repository, tmp_path)
    assert not (tmp_path / "selection").exists()


def test_preparation_requires_fresh_destination_and_review(repository, tmp_path):
    with pytest.raises(ValueError, match="review identity"):
        readers.prepare(repository, tmp_path / "empty", review_id="")
    selected = _selection(repository, tmp_path)
    with pytest.raises(ValueError, match="fresh"):
        readers.prepare(repository, selected.root.parent, review_id="review-again")
