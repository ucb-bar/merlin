"""Structural Python regex-call findings, without repository or grading exemptions.

This is the shared AST mechanism, not a sandbox or complete dynamic-import analysis.
Callers own exemptions and failure policy. Malformed Python and unreadable files raise;
absence of inspection must never be indistinguishable from an empty finding list.
"""

from __future__ import annotations

import ast
from pathlib import Path

REGEX_FUNCS = frozenset(
    {
        "compile",
        "match",
        "search",
        "fullmatch",
        "sub",
        "subn",
        "findall",
        "finditer",
        "split",
        "escape",
    }
)


class _RegexVisitor(ast.NodeVisitor):
    """Collect line numbers of ``re``-module call sites, following the file's import aliases."""

    def __init__(self) -> None:
        self.aliases: set[str] = set()  # module aliases bound to `re` (e.g. {"re", "_re"})
        self.from_funcs: dict[str, str] = {}  # local binding -> canonical function imported from re
        self.hits: list[tuple[int, str]] = []  # (lineno, what)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "re":
                self.aliases.add(alias.asname or "re")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "re":
            for alias in node.names:
                self.from_funcs[alias.asname or alias.name] = alias.name
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        fn = node.func
        # <alias>.<func>(...)
        if (
            isinstance(fn, ast.Attribute)
            and isinstance(fn.value, ast.Name)
            and fn.value.id in self.aliases
            and fn.attr in REGEX_FUNCS
        ):
            self.hits.append((fn.lineno, f"{fn.value.id}.{fn.attr}"))
        # bare <func>(...) from `from re import <func>`
        elif isinstance(fn, ast.Name) and self.from_funcs.get(fn.id) in REGEX_FUNCS:
            self.hits.append((fn.lineno, fn.id))
        self.generic_visit(node)


def scan_source(source: str, *, filename: str = "<unknown>") -> list[tuple[int, str]]:
    """Return ordered distinct ``(line, call)`` findings; do not honor inline markers."""
    tree = ast.parse(source, filename=filename)
    visitor = _RegexVisitor()
    visitor.visit(tree)
    return sorted(set(visitor.hits))


def scan_file(path: Path) -> list[tuple[int, str]]:
    """Scan UTF-8 source using the repository's historical replacement decoding."""
    return scan_source(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
