"""Reviewed APIs do not admit siblings, answer providers, or changed bytes."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from merlin.targetgen.compiler_library import CompilerLibraryError, freeze_compiler_library, selected_library_record
from merlin.targetgen.package_runtime import CertFailure, _py_imports_merlin, integrity_scan


def _library(tmp_path, *, body="def lower(value):\n    return value\n"):
    root = tmp_path / "installed"
    (root / "merlin").mkdir(parents=True)
    (root / "merlin/__init__.py").write_text("")
    (root / "merlin/portable.py").write_text(body)
    return root, freeze_compiler_library(
        root,
        review_id="independent-generic-api-review",
        public_modules=("merlin.portable",),
        sources=(("merlin/__init__.py", "merlin"), ("merlin/portable.py", "merlin.portable")),
    )


def test_exact_module_admission_does_not_grant_ancestors_or_children(tmp_path):
    root, library = _library(tmp_path)
    library.verify(root)
    for source in ("import merlin.portable", "from merlin.portable import lower", "from merlin import portable"):
        assert _py_imports_merlin(source, compiler_library=library) is None
        assert _py_imports_merlin(source) is not None
    for source in ("import merlin", "import merlin.portable.reference", "from merlin import portable, runtime"):
        assert _py_imports_merlin(source, compiler_library=library) is not None


def test_candidate_manifest_cannot_grant_itself_library_access(tmp_path):
    root, library = _library(tmp_path)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "tool.py").write_text("from merlin.portable import lower\n")
    package = SimpleNamespace(
        directory=candidate, integrity_exempt=False, manifest={"compiler_library": library.record()}
    )
    with pytest.raises(CertFailure):
        integrity_scan(package)
    integrity_scan(package, compiler_library=library, compiler_library_root=root)
    (root / "merlin/portable.py").write_text("def lower(value):\n    return 0\n")
    with pytest.raises(CertFailure, match="bytes changed"):
        integrity_scan(package, compiler_library=library, compiler_library_root=root)


@pytest.mark.parametrize(
    "body",
    [
        "import merlin.runtime.reference\n",
        "import merlin_experiments.phase2.campaign\n",
        "from . import unreviewed\n",
        "import merlin.unreviewed\n",
    ],
)
def test_missing_or_withheld_dependency_cannot_hide_in_approved_source(tmp_path, body):
    with pytest.raises(CompilerLibraryError):
        _library(tmp_path, body=body)


def test_resource_hash_and_namespace_initializer_are_mandatory(tmp_path):
    root, library = _library(tmp_path)
    with pytest.raises(CompilerLibraryError, match="initializer"):
        freeze_compiler_library(
            root,
            review_id="review",
            public_modules=("merlin.portable",),
            sources=(("merlin/portable.py", "merlin.portable"),),
        )
    (root / "merlin/__init__.py").write_text("import merlin.unreviewed\n")
    with pytest.raises(CompilerLibraryError, match="bytes changed"):
        library.verify(root)


@pytest.mark.parametrize("relative", ["../secret.py", "docs/journey.md", "out/answer.py", ".git/config"])
def test_history_and_reports_are_not_library_members(tmp_path, relative):
    root, _ = _library(tmp_path)
    with pytest.raises(CompilerLibraryError):
        freeze_compiler_library(
            root, review_id="review", public_modules=("merlin.portable",), sources=((relative, None),)
        )


def test_symlink_cannot_replace_qualified_bytes(tmp_path):
    root, library = _library(tmp_path)
    path = root / "merlin/portable.py"
    other = tmp_path / "outside.py"
    other.write_bytes(path.read_bytes())
    assert hashlib.sha256(other.read_bytes()).hexdigest() == library.members[1].sha256
    path.unlink()
    path.symlink_to(other)
    with pytest.raises(CompilerLibraryError, match="linked"):
        library.verify(root)


def test_library_policy_cannot_combine_with_exemption_or_partial_binding(tmp_path):
    root, library = _library(tmp_path)
    package = SimpleNamespace(directory=tmp_path, integrity_exempt=True)
    with pytest.raises(CertFailure, match="exemption"):
        integrity_scan(package, compiler_library=library, compiler_library_root=root)
    with pytest.raises(CertFailure, match="together"):
        integrity_scan(package, compiler_library=library)


def test_transport_library_selection_requires_the_original_explicit_pair(tmp_path):
    root, library = _library(tmp_path)
    assert selected_library_record(None, None) is None
    record = selected_library_record(library, root)
    assert record == {"root": str(root), "contract_sha256": library.sha256, "contract": library.record()}
    for contract, source in ((library, None), (None, root), (library.record(), root), (library, str(root))):
        with pytest.raises(CompilerLibraryError, match="together"):
            selected_library_record(contract, source)
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(CompilerLibraryError, match="canonical"):
        selected_library_record(library, alias)
