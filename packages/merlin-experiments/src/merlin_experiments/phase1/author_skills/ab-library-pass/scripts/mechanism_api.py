"""Show one catalogue mechanism: what it needs, where its API lives, its signature and its library callers.

    python mechanism_api.py ID [--workspace .] [--library submission/compiler_library]
    python mechanism_api.py --list [--facet FACET]

Reads OPTIMIZATION_CATALOG.json and OPTIMIZATION_SUGGESTIONS.json from the workspace (either may be
absent), finds the mechanism's module in the editable compiler library and prints the symbol's
signature and docstring, then up to ten library modules that already use the symbol -- the nearest
worked examples of its adapter. Read-only; parses source with ``ast`` and imports nothing.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

CATALOG = "OPTIMIZATION_CATALOG.json"
SUGGESTIONS = "OPTIMIZATION_SUGGESTIONS.json"
MAX_CALLERS = 10


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def mechanisms(workspace: Path) -> dict[str, dict]:
    """``{id: merged entry}`` from the catalogue and the target's suggestion file."""
    merged: dict[str, dict] = {}
    catalog = _load(workspace / CATALOG)
    for row in (catalog or {}).get("mechanisms") or ():
        if isinstance(row, dict) and row.get("id"):
            compiler = row.get("compiler") if isinstance(row.get("compiler"), dict) else {}
            merged[row["id"]] = {
                "id": row["id"], "facet": row.get("facet"), "mechanism": row.get("mechanism"),
                "module": compiler.get("module"), "symbol": compiler.get("symbol"), "path": compiler.get("path"),
                "access": compiler.get("access"), "summary": compiler.get("source_summary"),
                "applicability": row.get("applicability"),
                "remaining_adapter_or_feedback": row.get("remaining_adapter_or_feedback"),
                "evidence_to_request": row.get("evidence_to_request"),
                "implementation_status": row.get("implementation_status"),
            }
    suggestions = _load(workspace / SUGGESTIONS)
    for row in (suggestions or {}).get("all_mechanisms") or ():
        if isinstance(row, dict) and row.get("id"):
            entry = merged.setdefault(row["id"], {"id": row["id"]})
            entry.setdefault("facet", row.get("family"))
            entry.setdefault("mechanism", row.get("mechanism"))
            if not entry.get("module"):
                entry["module"], entry["symbol"] = row.get("api"), row.get("symbol")
            entry.update(status_for_this_target=row.get("status"), requirements=row.get("requirements"),
                         legality=row.get("legality"), adapter_requirements=row.get("adapter_requirements"))
    return merged


def source_file(library: Path, module: str | None, catalog_path: str | None = None) -> Path | None:
    """The module's file in the library copy (``pkg/mod.py`` or ``pkg/mod/__init__.py``)."""
    choices = []
    if module:
        base = library.joinpath(*module.split("."))
        choices += [base.with_suffix(".py"), base / "__init__.py"]
    if catalog_path:
        parts = Path(catalog_path).parts
        choices.append(library.joinpath(*(parts[1:] if parts[:1] == ("src",) else parts)))
    return next((path for path in choices if path.is_file()), None)


def definition(path: Path, symbol: str | None) -> dict | None:
    """Signature, line and docstring of a top-level function or class named ``symbol``."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol:
            if isinstance(node, ast.ClassDef):
                init = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
                fields = [ast.unparse(n.target) + ": " + ast.unparse(n.annotation)
                          for n in node.body if isinstance(n, ast.AnnAssign)]
                signature = (f"class {node.name}(" + (ast.unparse(init.args) if init else ", ".join(fields)) + ")")
            else:
                returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
                signature = f"def {node.name}({ast.unparse(node.args)}){returns}"
            return {"signature": signature, "line": node.lineno, "docstring": ast.get_docstring(node) or ""}
    return None


def _absolute(node: ast.ImportFrom, package: list[str]) -> str | None:
    """The absolute module an ``ImportFrom`` names, resolving ``from .x import y`` against ``package``."""
    if not node.level:
        return node.module
    if node.level - 1 > len(package):
        return None
    base = package[:len(package) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def callers(library: Path, module: str | None, symbol: str | None, *, skip: Path | None) -> list[str]:
    """Library files that import ``symbol`` from ``module`` (or the module itself), with line numbers."""
    if not module:
        return []
    found = []
    for path in sorted(library.rglob("*.py")):
        if skip is not None and path == skip or "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError):
            continue
        package = list(path.relative_to(library).with_suffix("").parts[:-1])
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and _absolute(node, package) == module and any(
                    alias.name == symbol for alias in node.names):
                found.append(f"{path.relative_to(library).as_posix()}:{node.lineno}")
                break
            if isinstance(node, ast.Import) and any(alias.name == module for alias in node.names):
                found.append(f"{path.relative_to(library).as_posix()}:{node.lineno}")
                break
        if len(found) >= MAX_CALLERS:
            break
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("id", nargs="?")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--library", default="", help="compiler library root (default: submission/compiler_library)")
    parser.add_argument("--list", action="store_true", help="list mechanism ids")
    parser.add_argument("--facet", default="", help="with --list, only this facet")
    args = parser.parse_args(argv)
    workspace = Path(args.workspace)
    library = Path(args.library) if args.library else workspace / "submission" / "compiler_library"
    known = mechanisms(workspace)
    if args.list or not args.id:
        for entry in known.values():
            if not args.facet or entry.get("facet") == args.facet:
                print(f"{entry['id']:40} {str(entry.get('facet')):16} {entry.get('status_for_this_target') or '-':12} "
                      f"{entry.get('module')}:{entry.get('symbol')}")
        return 0 if known else 2
    entry = known.get(args.id)
    if entry is None:
        print(f"unknown mechanism {args.id!r}; --list shows the ids in {CATALOG} / {SUGGESTIONS}", file=sys.stderr)
        return 2
    path = source_file(library, entry.get("module"), entry.get("path"))
    shown = {key: value for key, value in entry.items() if value not in (None, "", {}, [])}
    if path is not None:
        shown["source"] = path.as_posix()
        found = definition(path, entry.get("symbol"))
        if found:
            shown.update(source=f"{path.as_posix()}:{found['line']}", signature=found["signature"],
                         docstring=found["docstring"])
        shown["library_callers"] = callers(library, entry.get("module"), entry.get("symbol"), skip=path)
    else:
        shown["source"] = f"not found under {library}; check the module name or pass --library"
    print(json.dumps(shown, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
