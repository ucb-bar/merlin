"""Startup chooses exact source bytes without replacing live foreign leaves."""

import hashlib
import importlib
import json
import sys

import pytest

from merlin.targetgen import selected_library_imports as S
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


def test_v2_uses_only_exact_relocated_leaves_and_missing_host_adds_nothing(library, tmp_path, monkeypatch):
    selection, root = library
    guest = tmp_path / "already-admitted-guest"
    root.rename(guest)
    monkeypatch.setattr(S, "_guest_root", lambda digest: guest)
    missing = tmp_path / "unmounted-original-host"
    report = S.activate_selected_sources_v2(selection, original_host_roots=(missing,))
    package = importlib.import_module("selected_policy_test")
    assert package.__path__ == [str(guest / "selected_policy_test")]
    assert report["schema"] == "merlin.selected_compiler_source_startup.v2"
    assert report["root"] == str(guest)
    assert report["contract_sha256"] == selection["contract_sha256"]
    assert not root.exists() and not missing.exists()


def test_v1_still_requires_original_root(library, tmp_path, monkeypatch):
    selection, root = library
    guest = tmp_path / "already-admitted-guest"
    root.rename(guest)
    monkeypatch.setattr(S, "_guest_root", lambda digest: guest)
    with pytest.raises(FileNotFoundError):
        activate_selected_sources(selection)


@pytest.mark.parametrize("change", ["missing", "source", "contract", "linked", "linked-parent"])
def test_v2_relocation_retains_current_source_checks(library, tmp_path, monkeypatch, change):
    selection, root = library
    guest = tmp_path / "guest-parent" / "already-admitted-guest"
    guest.parent.mkdir()
    root.rename(guest)
    monkeypatch.setattr(S, "_guest_root", lambda digest: guest)
    leaf = guest / "selected_policy_test/mechanism.py"
    if change == "missing":
        leaf.unlink()
    elif change == "source":
        leaf.write_text("def result(): return 'wrong'\n")
    elif change == "contract":
        selection["contract"]["review_id"] = "forged"
    elif change == "linked":
        target = tmp_path / "foreign-source.py"
        leaf.rename(target)
        leaf.symlink_to(target)
    else:
        real = tmp_path / "real-parent"
        guest.parent.rename(real)
        guest.parent.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="changed|absent|linked|canonical"):
        S.activate_selected_sources_v2(selection)
    assert "selected_policy_test.mechanism" not in sys.modules


def test_v2_existing_invalid_original_never_falls_back(library, tmp_path, monkeypatch):
    import shutil

    selection, root = library
    guest = tmp_path / "valid-guest"
    shutil.copytree(root, guest)
    monkeypatch.setattr(S, "_guest_root", lambda digest: guest)
    (root / "selected_policy_test/mechanism.py").write_text("def result(): return 'wrong'\n")
    with pytest.raises(ValueError, match="changed"):
        S.activate_selected_sources_v2(selection)


@pytest.mark.parametrize("linked", [False, True])
def test_v2_invalid_host_dependency_root_refuses(library, tmp_path, linked):
    selection, _ = library
    host = tmp_path / "host"
    if linked:
        host.symlink_to(tmp_path / "absent", target_is_directory=True)
    else:
        host.write_text("not a directory")
    with pytest.raises(ValueError, match="dependency roots"):
        S.activate_selected_sources_v2(selection, original_host_roots=(host,))


def test_v2_preloaded_same_byte_foreign_leaf_still_refuses(library, tmp_path, monkeypatch):
    import shutil

    selection, root = library
    foreign = tmp_path / "foreign"
    shutil.copytree(root, foreign)
    monkeypatch.syspath_prepend(str(foreign))
    loaded = importlib.import_module("selected_policy_test.mechanism")
    with pytest.raises(ValueError, match="foreign compiler leaf"):
        S.activate_selected_sources_v2(selection, original_host_roots=(foreign,))
    assert importlib.import_module("selected_policy_test.mechanism") is loaded


def test_v2_never_replaces_live_native_modules_with_guest_modules(library, tmp_path, monkeypatch):
    selection, root = library
    S.activate_selected_sources_v2(selection)
    loaded = importlib.import_module("selected_policy_test.mechanism")
    guest = tmp_path / "guest"
    root.rename(guest)
    monkeypatch.setattr(S, "_guest_root", lambda digest: guest)
    with pytest.raises(ValueError, match="absent|foreign"):
        S.activate_selected_sources_v2(selection)
    assert importlib.import_module("selected_policy_test.mechanism") is loaded


def test_v1_file_host_root_keeps_its_original_empty_dependency_meaning(library, tmp_path):
    selection, _ = library
    host = tmp_path / "original-file"
    host.write_text("not a source directory")
    assert activate_selected_sources(selection, original_host_roots=(host,))["loaded_modules"] == 2
