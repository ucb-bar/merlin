"""A declared target build helper must affect identity, never widen its package boundary."""

from pathlib import Path
from types import SimpleNamespace

import time

import pytest

from merlin.runtime.backends import base
from merlin.targetgen import build_cache as BC
from merlin.targetgen import target_registry


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
        return BC.build_identity(
            target="synthetic-target", lowered_mlir_text="module {}", cb={}, inputs=None, recipe=object()
        )

    first_paths = BC.build_path("synthetic-target")
    assert helper in first_paths
    first = BC._build_path_digest(first_paths)
    first_key = identity()
    assert first_key is not None
    helper.write_text("VALUE=2\n")
    assert BC._build_path_digest(BC.build_path("synthetic-target")) != first
    assert identity() != first_key


@pytest.mark.parametrize(
    "case",
    ["missing", "outside", "relative", "directory", "not_python", "symlink", "parent_symlink", "empty", "raises"],
)
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


def test_a_same_size_rewrite_in_one_tick_still_changes_the_build_identity(tmp_path, monkeypatch):
    """The stat signature is not the content, and the memo must not pretend otherwise.

    The build-path digest is memoized on (path, size, mtime). A file rewritten with the SAME number
    of bytes inside one clock tick has an identical signature, so the memo returned the previous
    digest and a build identity did not change when the code that performs the build did. Here the
    collision is forced rather than raced: the rewrite is given back the original mtime exactly.
    """
    import os

    _, helper, _ = setup_target(tmp_path, monkeypatch)
    paths = BC.build_path("synthetic-target")
    assert helper in paths

    helper.write_text("VALUE=1\n")
    before = helper.stat()
    first = BC._build_path_digest(paths)

    helper.write_text("VALUE=2\n")
    os.utime(helper, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert helper.stat().st_size == before.st_size
    assert helper.stat().st_mtime_ns == before.st_mtime_ns  # the signature is now identical
    assert helper.read_text() == "VALUE=2\n"

    assert BC._build_path_digest(paths) != first, (
        "the build-path digest did not change although the code that performs the build did"
    )


def test_a_quiescent_tree_is_still_memoized(tmp_path, monkeypatch):
    """The correctness rule must not quietly turn the cache off: ~75 files are re-read per capsule
    per tier without it. A file old enough that no later write can share its timestamp still hits."""
    import os

    _, helper, _ = setup_target(tmp_path, monkeypatch)
    paths = BC.build_path("synthetic-target")
    old = time.time_ns() - 10 * 1_000_000_000
    for path in paths:
        if path.exists():
            os.utime(path, ns=(old, old))

    reads = []
    import merlin.common.provenance as PR

    real = PR.source_digest
    monkeypatch.setattr(PR, "source_digest", lambda names: (reads.append(tuple(names)), real(names))[1])

    BC._BUILD_PATH_MEMO.clear()
    first = BC._build_path_digest(paths)
    assert len(reads) == 1, "the first call must read the files"
    assert BC._build_path_digest(paths) == first
    assert len(reads) == 1, "a quiescent tree must be answered from the memo"
