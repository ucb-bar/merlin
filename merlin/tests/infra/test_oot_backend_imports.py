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

import pytest

from merlin.common.paths import repo_root

BACKEND_ROOT = repo_root() / "merlin" / "targets"


def _backend_modules():
    return sorted(BACKEND_ROOT.glob("*/backend/*.py"))


def test_there_are_backend_packages_to_check():
    assert _backend_modules(), "no out-of-tree backend modules found — has the layout moved?"


@pytest.mark.parametrize("path", _backend_modules(), ids=lambda p: f"{p.parent.parent.name}/{p.name}")
def test_no_parent_relative_import(path):
    """level >= 2 is a parent-relative import; level == 1 is a sibling and is allowed."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bad = [f"line {n.lineno}: from {'.' * n.level}{n.module or ''} import ..."
           for n in ast.walk(tree)
           if isinstance(n, ast.ImportFrom) and (n.level or 0) >= 2]
    assert not bad, (
        f"{path.relative_to(repo_root())} uses a parent-relative import, which resolves to the synthetic "
        f"merlin._oot_backends namespace once the package is loaded out-of-tree; make it absolute:\n  "
        + "\n  ".join(bad))


def test_the_gemmini_harness_renderer_resolves_its_lazy_imports():
    """The renderer's imports are function-local, so only calling into it proves they resolve."""
    from merlin.runtime.backends.base import get_backend
    gem = get_backend("gemmini")
    assert hasattr(gem, "render_harness")
    # The two modules the movement path reaches lazily — import them the way the fixed code does.
    from merlin.runtime.commandbuffer import materialize_inputs  # noqa: F401
    from merlin.targetgen.contract.harness_abi import for_target  # noqa: F401
