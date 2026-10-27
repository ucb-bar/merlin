"""The capture worker reads the integer partition on every path, including captures with no integer contractions.

WHY THIS EXISTS. The export admission after conversion takes the contraction partition as an argument. The partition
is only computed inside the PT2E integerization block, so a capture that never enters that block (a program with no
integer contractions, an already-integer graph, FP32 staging) must still have the name bound, otherwise the worker
dies with ``UnboundLocalError`` after a successful conversion. The behavioural tests that run the whole worker
(test_session_capture_worker, test_fp32_capture_staging) skip without an explicit trace-capable interpreter, so this
check reads the source and needs none.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from merlin.targetgen import _m2m_capture_worker as worker


def _main() -> ast.FunctionDef:
    tree = ast.parse(Path(worker.__file__).read_text())
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")


def _reads(statement: ast.stmt, name: str) -> bool:
    return any(
        isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load)
        for node in ast.walk(statement)
    )


def _binds(statement: ast.stmt, name: str) -> bool:
    return isinstance(statement, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name for target in statement.targets
    )


@pytest.mark.parametrize("name", ["integer_partition", "integerization_receipt"])
def test_name_is_bound_unconditionally_before_any_conditional_use(name):
    body = _main().body
    first_read = next((i for i, statement in enumerate(body) if _reads(statement, name)), None)
    assert first_read is not None, f"{name} is no longer read by the worker; update this guard"
    assert any(_binds(statement, name) for statement in body[: first_read + 1]), (
        f"{name} is assigned only inside a conditional block before it is read at the top level of main()"
    )


def test_partition_is_none_without_integerization():
    """With no integerization receipt the export admission does not need a partition (the parent's behaviour)."""
    from merlin.targetgen.integerization_export import receipt_export_complete

    assert receipt_export_complete(None, None) is True
