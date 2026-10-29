"""All targets and the compatibility example use the same library preparation."""

import importlib.util
import subprocess
import sys

import pytest

from merlin_experiments.phase1 import prepare_author_library as shared

from merlin.common.paths import repo_root


def test_example_is_only_a_compatibility_entrypoint():
    spec = importlib.util.spec_from_file_location(
        "library_preparation_example", repo_root() / "examples/gemmini/phase1/launch/prepare_author_library.py",
    )
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    assert example.prepare is shared.prepare
    assert example.main is shared.main
    assert example.reviewed_modules is shared.reviewed_modules


def test_shared_module_cli_can_be_called_without_a_target_backend():
    result = subprocess.run(
        [sys.executable, "-B", "-m", "merlin_experiments.phase1.prepare_author_library", "--help"],
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--library-version" in result.stdout
    assert "{8,16,17,18,19,20,21,22,23,24}" in result.stdout
    assert "--generated-package" in result.stdout
    assert "--reviewed-module" in result.stdout


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def source_repo(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    _git(root, "init", "-q", "-b", "trunk")
    _git(root, "config", "user.name", "Preparation Test")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "commit.gpgsign", "false")
    (root / "src").mkdir()
    (root / "src/leaf.py").write_text("value = 1\n")
    _git(root, "add", "src/leaf.py")
    _git(root, "commit", "-q", "-m", "initial source")
    return root


class SelectionReached(Exception):
    """The real preparation source gate admitted inputs; stop before preparing unrelated SDK files."""


def _prepare_source(monkeypatch, root, revision, tmp_path):
    def reached(**kwargs):
        raise SelectionReached

    monkeypatch.setattr(shared, "load_selection", reached)
    return shared.prepare(
        repo=root, source_revision=revision, previous_root=tmp_path / "previous",
        previous_contract=tmp_path / "contract.json", previous_sha256="not-read",
        output=tmp_path / "prepared", library_version=22,
    )


def test_nested_export_preparation_uses_its_record_after_outer_head_and_sources_change(source_repo, monkeypatch, tmp_path):
    from merlin.targetgen.provenance import write_source_record

    revision = _git(source_repo, "rev-parse", "HEAD")
    export = source_repo / "out/framework"
    export.mkdir(parents=True)
    write_source_record(export, revision)
    (source_repo / "src/leaf.py").write_text("value = 2\n")
    _git(source_repo, "commit", "-q", "-am", "later checkout revision")
    assert _git(source_repo, "rev-parse", "HEAD") != revision
    (source_repo / "src/leaf.py").write_text("value = 3\n")
    with pytest.raises(SelectionReached):
        _prepare_source(monkeypatch, export, revision, tmp_path)
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("record", ["missing", "malformed", "wrong_revision"])
def test_an_export_cannot_borrow_the_enclosing_repository_revision(source_repo, monkeypatch, tmp_path, record):
    from merlin.targetgen.provenance import SOURCE_RECORD, write_source_record

    revision = _git(source_repo, "rev-parse", "HEAD")
    export = source_repo / "out/framework"
    export.mkdir(parents=True)
    if record == "malformed":
        (export / SOURCE_RECORD).write_text("not JSON")
    elif record == "wrong_revision":
        write_source_record(export, "0" * 40)
    with pytest.raises(ValueError, match="export's own source record"):
        _prepare_source(monkeypatch, export, revision, tmp_path)
    assert not (tmp_path / "prepared").exists()


def test_own_checkout_still_refuses_dirty_compiler_sources(source_repo, monkeypatch, tmp_path):
    revision = _git(source_repo, "rev-parse", "HEAD")
    with pytest.raises(SelectionReached):
        _prepare_source(monkeypatch, source_repo, revision, tmp_path)
    (source_repo / "src/leaf.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="uncommitted edits"):
        _prepare_source(monkeypatch, source_repo, revision, tmp_path)


def test_own_checkout_record_does_not_override_its_actual_head(source_repo, monkeypatch, tmp_path):
    from merlin.targetgen.provenance import write_source_record

    previous = _git(source_repo, "rev-parse", "HEAD")
    (source_repo / "src/leaf.py").write_text("value = 2\n")
    _git(source_repo, "commit", "-q", "-am", "later checkout revision")
    write_source_record(source_repo, previous)
    with pytest.raises(ValueError, match="selected repository HEAD"):
        _prepare_source(monkeypatch, source_repo, previous, tmp_path)


def test_a_reviewed_package_selects_only_its_own_initializer(tmp_path):
    from merlin_experiments.phase1.prepare_author_library import reviewed_modules, reviewed_package_sources

    (tmp_path / "merlin/compile/analysis").mkdir(parents=True)
    (tmp_path / "merlin/compile/analysis/__init__.py").write_text('"""analyses"""\n')
    (tmp_path / "merlin/compile/analysis/shapes.py").write_text("X = 1\n")
    requested = reviewed_modules(["merlin.compile.analysis", "merlin.compile.analysis.shapes"], existing=())
    assert reviewed_package_sources(requested, tmp_path) == {
        "merlin/compile/analysis/__init__.py": "merlin.compile.analysis",
        "merlin/compile/analysis/shapes.py": "merlin.compile.analysis.shapes",
    }
    # A name that is neither a leaf nor a package keeps its leaf path (and preparation refuses it).
    assert reviewed_package_sources(reviewed_modules(["merlin.compile.absent"], existing=()), tmp_path) == {
        "merlin/compile/absent.py": "merlin.compile.absent"
    }
