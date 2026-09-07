"""A declared target build helper must affect identity, never widen its package boundary."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.runtime.backends import base
from merlin.targetgen import build_cache as BC, target_registry


def setup_target(tmp_path, monkeypatch):
    package = tmp_path / "target"
    home, support = package / "backend", package / "build_support"
    home.mkdir(parents=True)
    support.mkdir()
    module, helper = home / "__init__.py", support / "whole_program.py"
    module.write_text("# backend\n")
    helper.write_text("VALUE=1\n")
    backend = SimpleNamespace(__file__=str(module), build_source_paths=lambda: [helper])
    monkeypatch.setattr(BC, "_BUILD_MODULES", ())
    monkeypatch.setattr(base, "get_backend", lambda _: backend)
    monkeypatch.setattr(target_registry, "resolve", lambda _: SimpleNamespace(base=package, external_root=None))
    return backend, helper, package


def test_declared_sibling_helper_changes_build_identity(tmp_path, monkeypatch):
    _, helper, _ = setup_target(tmp_path, monkeypatch)
    monkeypatch.setattr(BC, "recipe_token", lambda _: {"compile": ["synthetic-compiler"]})
    monkeypatch.setattr(BC, "toolchain_token", lambda _: "synthetic-toolchain-digest")
    monkeypatch.delenv("MERLIN_ELF_BUILD_CACHE", raising=False)
    def identity():
        return BC.build_identity(target="synthetic-target", lowered_mlir_text="module {}",
                                 cb={}, inputs=None, recipe=object())
    first_paths = BC.build_path("synthetic-target")
    assert helper in first_paths
    first = BC._build_path_digest(first_paths)
    first_key = identity()
    assert first_key is not None
    helper.write_text("VALUE=2\n")
    assert BC._build_path_digest(BC.build_path("synthetic-target")) != first
    assert identity() != first_key


@pytest.mark.parametrize("case", ["missing", "outside", "relative", "directory", "not_python", "symlink", "parent_symlink", "empty", "raises"])
def test_invalid_declared_closure_disables_partial_reuse(tmp_path, monkeypatch, case):
    backend, helper, package = setup_target(tmp_path, monkeypatch)
    if case == "missing":
        selected = package / "absent.py"
    elif case == "outside":
        selected = tmp_path / "outside.py"
        selected.write_text("# outside\n")
    elif case == "relative":
        selected = Path("build_support/whole_program.py")
    elif case == "directory":
        selected = helper.parent
    elif case == "not_python":
        selected = package / "blob.txt"
        selected.write_text("# not Python source\n")
    elif case == "symlink":
        selected = package / "linked.py"
        selected.symlink_to(helper)
    elif case == "parent_symlink":
        linked = package / "linked"
        linked.symlink_to(helper.parent, target_is_directory=True)
        selected = linked / helper.name
    else:
        selected = helper
    def declare():
        if case == "raises":
            raise RuntimeError("incomplete source declaration")
        return [] if case == "empty" else [selected]
    backend.build_source_paths = declare
    assert BC.build_path("synthetic-target") is None


def test_absent_hook_preserves_existing_backend_subtree(tmp_path, monkeypatch):
    backend, helper, _ = setup_target(tmp_path, monkeypatch)
    del backend.build_source_paths
    assert BC.build_path("synthetic-target") == (Path(backend.__file__),)
    assert helper not in BC.build_path("synthetic-target")
