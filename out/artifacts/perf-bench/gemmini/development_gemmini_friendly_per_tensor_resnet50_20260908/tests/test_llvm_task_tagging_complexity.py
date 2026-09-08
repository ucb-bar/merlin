from __future__ import annotations

import ast
from pathlib import Path

from xdsl.dialects.builtin import IntegerAttr, i64

from mlir_oot.codegen.builder import FnBuilder, iconst


ROOT = Path(__file__).resolve().parents[1]


def test_scoped_task_attribute_is_applied_at_insertion() -> None:
    builder = FnBuilder([])
    builder.scoped_attributes = {"merlin.global_task": IntegerAttr(7, i64)}
    ordinary = builder.add(iconst(1))
    prologue = builder.prologue(iconst(2))
    assert ordinary.attributes["merlin.global_task"] == IntegerAttr(7, i64)
    assert prologue.attributes["merlin.global_task"] == IntegerAttr(7, i64)


def test_emitter_does_not_rescan_all_blocks_for_each_task() -> None:
    """Do not regress whole-model emission to O(tasks * accumulated blocks)."""
    source = (ROOT / "compiler/mlir_oot/codegen/llvm_emit.py").read_text()
    tree = ast.parse(source)
    build = next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "build")
    assert not any(
        isinstance(node, ast.For)
        and ast.unparse(node.iter) == "self.fb.region.blocks"
        for node in ast.walk(build)
    )
