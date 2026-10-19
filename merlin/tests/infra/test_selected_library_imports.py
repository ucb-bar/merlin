"""Startup chooses exact source bytes without replacing live foreign leaves."""

import hashlib
import importlib
import json
import sys

import pytest

from merlin.targetgen.selected_library_imports import activate_selected_sources


@pytest.fixture
def library(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    root = tmp_path / "sdk"
    package = root / "selected_policy_test"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('"""Selected namespace."""\n')
    (package / "mechanism.py").write_text('def result():\n    return "selected"\n')
    contract = dict(
        schema="merlin.compiler_library.v1",
        review_id="original-host-test",
        public_modules=["selected_policy_test.mechanism"],
        members=[
            dict(
                module="selected_policy_test",
                path="selected_policy_test/__init__.py",
                sha256=hashlib.sha256((package / "__init__.py").read_bytes()).hexdigest(),
            ),
            dict(
                module="selected_policy_test.mechanism",
                path="selected_policy_test/mechanism.py",
                sha256=hashlib.sha256((package / "mechanism.py").read_bytes()).hexdigest(),
            ),
        ],
    )
    selection = dict(
        root=str(root),
        contract=contract,
        contract_sha256=hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )
    yield selection, root
    for name in tuple(sys.modules):
        if name.startswith("selected_policy_test"):
            sys.modules.pop(name)


def test_startup_imports_exact_selected_sources(library):
    selection, root = library
    report = activate_selected_sources(selection)
    module = importlib.import_module("selected_policy_test.mechanism")
    assert module.result() == "selected" and report["loaded_modules"] == 2
    assert module.__file__ == str(root / "selected_policy_test/mechanism.py")
    assert "source custody only" in report["authority"]


@pytest.mark.parametrize("change", ["source", "contract", "missing", "linked"])
def test_changed_or_missing_selection_refuses_before_import(library, tmp_path, change):
    selection, root = library
    path = root / "selected_policy_test/mechanism.py"
    if change == "source":
        path.write_text('def result():\n    return "foreign"\n')
    elif change == "contract":
        selection["contract"]["review_id"] = "another selection"
    elif change == "missing":
        path.unlink()
    else:
        target = tmp_path / "linked.py"
        path.rename(target)
        path.symlink_to(target)
    with pytest.raises(ValueError, match="changed|absent|linked"):
        activate_selected_sources(selection)
    assert "selected_policy_test.mechanism" not in sys.modules


def test_preloaded_foreign_leaf_is_never_replaced(library, tmp_path, monkeypatch):
    selection, root = library
    host = tmp_path / "original-host"
    package = host / "selected_policy_test"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes((root / "selected_policy_test/__init__.py").read_bytes())
    (package / "mechanism.py").write_text('def result():\n    return "foreign"\n')
    monkeypatch.syspath_prepend(str(host))
    foreign = importlib.import_module("selected_policy_test.mechanism")
    with pytest.raises(ValueError, match="foreign compiler leaf"):
        activate_selected_sources(selection, original_host_roots=(host,))
    assert importlib.import_module("selected_policy_test.mechanism") is foreign


def test_byte_identical_original_namespace_uses_snapshot_leaves(library, tmp_path, monkeypatch):
    selection, root = library
    host = tmp_path / "original-host"
    package = host / "selected_policy_test"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes((root / "selected_policy_test/__init__.py").read_bytes())
    (package / "host_dependency.py").write_text("original_host_dependency = True\n")
    monkeypatch.syspath_prepend(str(host))
    original = importlib.import_module("selected_policy_test")
    activate_selected_sources(selection, original_host_roots=(host,))
    assert importlib.import_module("selected_policy_test") is original
    assert importlib.import_module("selected_policy_test.mechanism").result() == "selected"
    assert importlib.import_module("selected_policy_test.host_dependency").original_host_dependency
    assert original.__path__[0] == str(root / "selected_policy_test")
