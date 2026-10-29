"""Cheap smoke for a background snapshot: skip grading a compiler that cannot even load.

Run 1 of the joint campaign graded a 63-line skeleton against 89 capsules for ~100 minutes; every one
failed at ``parse`` with ``ModuleNotFoundError``. Two independent, hidden-data-free checks end that early:

* ``static_smoke(tree)``: before any capsule is dispatched, every submission ``.py`` must compile and
  every unconditional module-level import must resolve to the stdlib, the submission itself, or a
  module this interpreter can find.
* ``LoadFailureScreen``: while the snapshot grades, the first N completed rows are watched; if all N
  failed with a load-class error and none passed, the rest of the snapshot is skipped.

A skipped snapshot is recorded as skipped (``prescreen_failed``, gated rows), never as pass or fail of
the capsules it did not run.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import re
import sys
import threading
from pathlib import Path

ENV_ENABLE = "MERLIN_SNAPSHOT_SMOKE"          # "0" disables both checks
ENV_ROWS = "MERLIN_SNAPSHOT_SMOKE_ROWS"       # N first completed rows (default 4)
DEFAULT_ROWS = 4
LOAD_FAILURE = re.compile(r"ModuleNotFoundError|ImportError|SyntaxError|IndentationError|No module named")
_SKIP_DIRS = {"__pycache__", ".git", "build", "_build"}


def enabled() -> bool:
    raw = os.environ.get(ENV_ENABLE, "1")
    if raw not in {"0", "1"}:
        raise ValueError(f"{ENV_ENABLE} must be 0 or 1")
    return raw == "1"


def rows_required() -> int:
    raw = os.environ.get(ENV_ROWS, str(DEFAULT_ROWS))
    if not raw.isascii() or not raw.isdecimal() or not 1 <= int(raw) <= 64:
        raise ValueError(f"{ENV_ROWS} must be an integer between 1 and 64")
    return int(raw)


def _python_files(tree: Path):
    for path in sorted(tree.rglob("*.py")):
        if not any(part in _SKIP_DIRS for part in path.relative_to(tree).parts) and path.is_file() \
                and not path.is_symlink():
            yield path


def _top_level_imports(module: ast.Module):
    """Unconditional module-level imports only: try/if/def bodies may legitimately guard a miss."""
    for node in module.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.lineno, node.module.split(".")[0]


def _local_names(tree: Path) -> set[str]:
    names = set()
    for entry in tree.iterdir():
        if entry.suffix == ".py":
            names.add(entry.stem)
        elif entry.is_dir() and not entry.name.startswith("."):
            names.add(entry.name)
    for path in _python_files(tree):  # nested roots are importable when their dir is on sys.path
        names.add(path.parent.name)
        names.add(path.stem)
    return names


def static_smoke(tree: str | Path) -> str | None:
    """A failure reason, or ``None`` when nothing cheap shows the compiler cannot load."""
    tree = Path(tree)
    if not tree.is_dir():
        return None
    local, missing = _local_names(tree), {}
    for path in _python_files(tree):
        try:
            module = ast.parse(path.read_bytes(), filename=str(path))
        except (SyntaxError, ValueError) as error:
            where = getattr(error, "lineno", None)
            return f"smoke: {path.relative_to(tree)} does not parse ({type(error).__name__}" \
                   f"{f' line {where}' if where else ''})"
        for line, name in _top_level_imports(module):
            if name in local or name in sys.stdlib_module_names or name in missing:
                continue
            try:
                found = importlib.util.find_spec(name) is not None
            except (ImportError, ValueError, AttributeError):
                found = False
            if not found:
                missing[name] = f"{path.relative_to(tree)}:{line}"
    if missing:
        sample = ", ".join(f"{name} ({where})" for name, where in sorted(missing.items())[:6])
        return f"smoke: unresolved module-level import(s): {sample}"
    return None


class LoadFailureScreen:
    """Trips when the first N completed rows all failed to load the compiler and none passed."""

    def __init__(self, rows: int | None = None):
        self.rows = rows_required() if rows is None else rows
        self.reason: str | None = None
        self._seen: list[dict] = []
        self._lock = threading.Lock()

    def observe(self, row: dict) -> None:
        if row.get("status") == "gated" or row.get("background_deferred"):
            return  # a skipped capsule says nothing about the compiler
        with self._lock:
            if self.reason is not None or len(self._seen) >= self.rows:
                return
            self._seen.append(row)
            if len(self._seen) < self.rows:
                return
            if all(r.get("status") == "fail" and isinstance(r.get("failure_detail"), str)
                   and LOAD_FAILURE.search(r["failure_detail"]) for r in self._seen):
                first = self._seen[0]["failure_detail"].replace("\n", " ")[:160]
                self.reason = (f"smoke: the first {self.rows} graded capsules all failed to load the compiler "
                               f"({first}); the remaining capsules of this snapshot were skipped")
