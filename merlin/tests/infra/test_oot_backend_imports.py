"""An out-of-tree target backend may not use a PARENT-relative import.

These packages live under ``merlin/targets/<t>/backend/`` but are loaded under the synthetic name
``merlin._oot_backends.<t>``, so ``..`` resolves to that namespace rather than to ``merlin.runtime`` —
the layout they were written against before the eviction. A leftover ``from ..commandbuffer import ...``
therefore raises ModuleNotFoundError not at import time but LAZILY, inside the harness renderer at GRADE
time, where the runner reports it as an opaque "spike invocation failed / tool_crash". Every capsule in a
run fails that way, which reads as a broken agent rather than a broken import.

Sibling-relative imports (``from .gemmini_codegen import ...``) are fine: those move with the package.
"""

from __future__ import annotations

import ast
import symtable

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.target_registry import explicit_targets

BACKEND_ROOT = repo_root() / "merlin" / "targets"


def _backend_modules():
    # Retained native packages are migration debt; selected OOT owners are
    # checked directly, without borrowing copies from the reference tree.
    paths = set(BACKEND_ROOT.glob("*/backend/*.py"))
    for root in explicit_targets().values():
        paths.update((root / "backend").glob("*.py"))
    return sorted(paths)


def test_there_are_backend_packages_to_check():
    assert _backend_modules(), "no out-of-tree backend modules found — has the layout moved?"


@pytest.mark.parametrize("path", _backend_modules(), ids=lambda p: f"{p.parent.parent.name}/{p.name}")
def test_no_parent_relative_import(path):
    """level >= 2 is a parent-relative import; level == 1 is a sibling and is allowed."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bad = [
        f"line {n.lineno}: from {'.' * n.level}{n.module or ''} import ..."
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and (n.level or 0) >= 2
    ]
    assert not bad, (
        f"{path} uses a parent-relative import, which resolves to the synthetic "
        f"merlin._oot_backends namespace once the package is loaded out-of-tree; make it absolute:\n  "
        + "\n  ".join(bad)
    )


def test_the_gemmini_harness_renderer_resolves_its_lazy_imports():
    """The renderer's imports are function-local, so only calling into it proves they resolve."""
    from merlin.runtime.backends.base import get_backend

    gem = get_backend("gemmini")
    assert hasattr(gem, "render_harness")
    # The two modules the movement path reaches lazily — import them the way the fixed code does.
    from merlin.runtime.commandbuffer import materialize_inputs  # noqa: F401
    from merlin.targetgen.contract.harness_abi import for_target  # noqa: F401


def _lazy_global_reads(source, filename):
    tree = ast.parse(source, filename=filename)
    lazy = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_DERIVED" for t in node.targets):
            lazy.update(
                e.value for e in ast.walk(node.value) if isinstance(e, ast.Constant) and isinstance(e.value, str)
            )
    pending = [symtable.symtable(source, filename, "exec")]
    bad = []
    while pending:
        scope = pending.pop()
        pending.extend(scope.get_children())
        if scope.get_type() != "function":
            continue
        for symbol in scope.get_symbols():
            if symbol.get_name() in lazy and symbol.is_referenced() and symbol.is_global():
                bad.append((scope.get_lineno(), scope.get_name(), symbol.get_name()))
    return sorted(bad)


def test_lazy_attribute_check_respects_lexical_closures():
    source = """
_DERIVED = {"DIM"}
def outer():
    DIM = 16
    def inner():
        return DIM
    return inner
def broken():
    return DIM
"""
    assert _lazy_global_reads(source, "synthetic.py") == [(8, "broken", "DIM")]


def test_no_module_reads_its_own_lazy_attribute_as_a_bare_global():
    """A PEP 562 module ``__getattr__`` serves ``mod.NAME`` from OUTSIDE the module only.

    Inside the module a bare ``NAME`` is an ordinary global lookup: Python checks the module dict, then
    builtins, then raises NameError — it never consults ``__getattr__``. So moving module constants
    behind a lazy ``__getattr__`` silently breaks every in-module use of them, and only at the moment
    that code path runs. gemmini's MLIR codegen shipped exactly that: eleven derived facts read as bare
    globals inside ``emit_kernel_mlir``, which raised NameError the first time the emit path was
    exercised. Read them off the resolver into locals instead.
    """
    for path in _backend_modules():
        bad = _lazy_global_reads(path.read_text(encoding="utf-8"), str(path))
        assert not bad, f"{path}: module __getattr__ cannot serve these function-global reads: {bad}"
