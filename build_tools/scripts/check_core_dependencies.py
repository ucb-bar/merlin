"""Reject static operational imports from compiler core into optional distributions.

Only exact, function-scoped legacy compatibility adapters are exempt. This is not
a sandbox or a proof about dynamic imports; cold-import and installed execution
tests cover those interfaces separately.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

# Reviewed compatibility interfaces, not a general dependency-debt allowance.
# Core operations never invoke these optional research entrypoints themselves.
COMPATIBILITY_IMPORTS = frozenset(
    {
        ("merlin.mining.from_strategy", ("mint_fork",), "merlin.mining.fork"),
        ("merlin.targetgen.package_runtime", ("__getattr__",), "merlin.targetgen.package_certification"),
    }
)


def _module(path: Path) -> str:
    parts = path.with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def audit(root: Path) -> list[str]:
    core = root / "src"
    if not any((core / "merlin").rglob("*.py")):
        return [f"core dependency scan has no Python sources: {core / 'merlin'}"]
    owners: dict[str, str] = {}
    concrete: dict[str, Path] = {}
    errors = []
    for source in [core, *sorted((root / "packages").glob("*/src"))]:
        owner = "core" if source == core else source.parent.name
        for path in sorted(source.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            name = _module(path.relative_to(source))
            if name in concrete:
                errors.append(f"duplicate source module {name}: {concrete[name]} and {path}")
            concrete[name] = path
            # A concrete initializer wins over an implicit namespace inferred
            # from another distribution's child; importing it executes its owner.
            owners[name] = owner
            # An implicit namespace without a core initializer is optional too.
            while "." in name:
                name = name.rpartition(".")[0]
                owners.setdefault(name, owner)

    def optional_owner(name: str) -> str | None:
        while name:
            if name in owners and owners[name] != "core":
                return owners[name]
            # A core child still executes an optional parent's initializer.
            name = name.rpartition(".")[0]
        return None

    for path in sorted(core.rglob("*.py")):
        module = _module(path.relative_to(core))
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        relative = path.relative_to(root)
        try:
            tree = ast.parse(path.read_text(), filename=str(relative))
        except SyntaxError as exc:
            errors.append(f"core dependency scan cannot parse {relative}: {exc}")
            continue

        def walk(node, scope=()):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                scope = (*scope, (kind, node.name))
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    try:
                        base = importlib.util.resolve_name("." * node.level + base, package)
                    except ImportError as exc:
                        errors.append(f"invalid relative core import {relative}:{node.lineno}: {exc}")
                        return
                names = [base if alias.name == "*" else base + "." + alias.name for alias in node.names]
            reported = set()
            for name in names:
                owner = optional_owner(name)
                if owner is None or owner in reported:
                    continue
                # A from-import's attribute is part of its module, not another
                # compatibility permission for each imported function name.
                if any(
                    module == allowed_module
                    and scope == tuple(("function", name) for name in allowed_scope)
                    and (name == imported or name.startswith(imported + "."))
                    for allowed_module, allowed_scope, imported in COMPATIBILITY_IMPORTS
                ):
                    continue
                errors.append(f"core imports optional {owner}: {relative}:{node.lineno}: {name}")
                reported.add(owner)
            for child in ast.iter_child_nodes(node):
                walk(child, scope)

        walk(tree)
    return errors


if __name__ == "__main__":
    problems = audit(Path(__file__).resolve().parents[2])
    for problem in problems:
        print(problem)
    if not problems:
        print("core has no static operational imports from optional distributions")
    raise SystemExit(bool(problems))
