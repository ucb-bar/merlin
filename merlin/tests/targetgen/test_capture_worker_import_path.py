"""The capture worker puts its OWN Merlin package on sys.path for every capture, not only recipe ones.

WHY THIS EXISTS. Every capture reads ``merlin.targetgen.integerization_export.receipt_export_complete``
after lowering, but the worker inserted its own package root only when ``--recipe`` was given. In a
capture interpreter that does not have Merlin installed (a separate model2MLIR venv), every fp32 capture
then died with ``ModuleNotFoundError: merlin`` -- all 193 Core ATen captures in a Phase-0 run, until the
operator set PYTHONPATH by hand.
"""

from __future__ import annotations

import ast

from merlin.common.paths import repo_root

_WORKER = repo_root() / "src/merlin/targetgen/_m2m_capture_worker.py"


def _is_package_root_insert(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call) and ast.unparse(node.func) == "sys.path.insert" and "parents[2]" in ast.unparse(node)
    )


def test_the_package_root_insert_is_unconditional():
    tree = ast.parse(_WORKER.read_text(encoding="utf-8"))
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    inserts = [node for node in ast.walk(tree) if _is_package_root_insert(node)]
    assert inserts, "the worker no longer inserts its own package root"
    for call in inserts:
        node = call
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.If):
                raise AssertionError(f"package-root insert is conditional on {ast.unparse(node.test)!r}")
            if isinstance(node, ast.FunctionDef):
                break
