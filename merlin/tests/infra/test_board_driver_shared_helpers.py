"""The board drivers' duplicated helpers live once, in ``build_tools/scripts/_k1_common.py``.

Two ways that lift can regress without anything else failing:

  * a driver loaded BY PATH -- how the tests and sibling drivers load them, with build_tools/scripts
    not on sys.path -- must still import, and the helper name it exposes must be the shared object,
    because siblings read it off the driver (``k1_fp16_gemm`` calls ``k1_cross_framework_ops._cc``);
  * a driver may not grow a copy of a shared helper back. Two copies are how one silently drifts
    from the other while both drivers keep producing numbers.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

from merlin.common.paths import repo_root

SCRIPTS = repo_root() / "build_tools" / "scripts"
COMMON = SCRIPTS / "_k1_common.py"
MODULE = COMMON.stem


def _defs_without_docstrings(source: str) -> dict[str, str]:
    """Top-level function name -> ast dump with the docstring dropped (a docstring is not behaviour)."""
    out = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            node = ast.parse(ast.unparse(node)).body[0]
            first = node.body[0] if node.body else None
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                node.body = node.body[1:]
            out[node.name] = ast.dump(node)
    return out


def _copies_of_shared(source: str) -> list[str]:
    shared = _defs_without_docstrings(COMMON.read_text(encoding="utf-8"))
    return sorted(name for name, dump in _defs_without_docstrings(source).items() if shared.get(name) == dump)


def _importers() -> dict[Path, list[str]]:
    out = {}
    for path in sorted(SCRIPTS.glob("*.py")):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.ImportFrom) and node.module == MODULE:
                out[path] = [alias.name for alias in node.names]
    return out


def test_the_shared_module_is_actually_used():
    """Not vacuous: the parametrized checks below run over real importers of real helpers."""
    importers = _importers()
    assert len(importers) >= 2, importers
    shared = set(_defs_without_docstrings(COMMON.read_text(encoding="utf-8")))
    assert shared and all(set(names) <= shared for names in importers.values())


@pytest.mark.parametrize("path", sorted(_importers()), ids=lambda p: p.name)
def test_a_driver_loaded_by_path_exposes_the_shared_helper(path, monkeypatch):
    scripts = str(SCRIPTS.resolve())
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p and str(Path(p).resolve()) != scripts])
    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    name = f"_shared_helper_probe_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    driver = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, driver)   # registered before exec: some drivers define dataclasses
    spec.loader.exec_module(driver)
    common = sys.modules[MODULE]
    for helper in _importers()[path]:
        assert getattr(driver, helper) is getattr(common, helper), (path.name, helper)


def test_the_copy_detector_can_fire():
    """A detector that cannot fire would pass every tree; the shared module is a copy of itself."""
    assert _copies_of_shared(COMMON.read_text(encoding="utf-8"))


def test_no_driver_redefines_a_shared_helper():
    copies = {p.name: found for p in sorted(SCRIPTS.glob("*.py")) if p != COMMON
              for found in [_copies_of_shared(p.read_text(encoding="utf-8"))] if found}
    assert not copies, f"import these from {COMMON.name} instead of redefining them: {copies}"
