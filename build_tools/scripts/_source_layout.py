"""Dependency-free source discovery shared by repository gates during the layout cutover.

Scan canonical source once, retaining archived extension layouts. Historical debt remains
keyed by its original path: relocating a file neither forgives it nor grows a ratchet/allowlist.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
from pathlib import Path

SOURCE_SCAN_ROOTS = ("src", "merlin/python/merlin", "packages")
EXCLUDED_PARTS = frozenset({"_data", "__pycache__", ".venv", "venv", "_qa_ws", "build", "dist"})


def module_alias(path: Path, root: Path) -> tuple[str, Path] | None:
    """Resolve a literal, otherwise inert ``sys.modules`` compatibility identity.

    This is static source correspondence, not an access grant or proof of use.
    Any extra executable statement makes the source an ordinary module again.
    """
    try:
        source = path.read_text(encoding="utf-8")
        if "modules" not in source or "__name__" not in source:
            return None
        tree = ast.parse(source)
    except (OSError, UnicodeDecodeError, SyntaxError):
        return None
    body = list(tree.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            body.pop(0)
    owner = binding = None
    imported_sys = assigned = False
    for node in body:
        if isinstance(node, ast.Import) and len(node.names) == 1:
            if node.names[0].name != "sys" or node.names[0].asname:
                return None
            imported_sys = True
        elif isinstance(node, ast.ImportFrom) and not node.level and len(node.names) == 1:
            item = node.names[0]
            if node.module == "__future__" and item.name == "annotations":
                continue
            if owner is not None or not item.asname or not node.module:
                return None
            owner, binding = f"{node.module}.{item.name}", item.asname
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            if not imported_sys:
                return None
            expected = ast.parse("sys.modules[__name__] = owner").body[0].targets[0]
            if ast.dump(node.targets[0]) != ast.dump(expected):
                return None
            if not isinstance(node.value, ast.Name) or node.value.id != binding:
                return None
            assigned = True
        elif isinstance(node, ast.If) and binding is not None:
            expected = ast.parse(f'if __name__ == "__main__":\n    raise SystemExit({binding}.main())').body[0]
            if ast.dump(node) != ast.dump(expected):
                return None
        else:
            return None
    if not imported_sys or not assigned or owner is None:
        return None
    parts = owner.split(".")
    candidates = set()
    for package in source_packages(root):
        if package.name == parts[0]:
            target = package.joinpath(*parts[1:])
            for candidate in (target.with_suffix(".py"), target / "__init__.py"):
                if candidate.is_file():
                    candidates.add(candidate.resolve())
    if len(candidates) != 1:
        return None
    target = candidates.pop()
    return (owner, target) if target != path.resolve() and target.is_relative_to(root.resolve()) else None


def policy_path(path: str) -> str:
    """Stable debt identity for the same merlin module, including shared-namespace distributions."""
    # Declaration/catalog orchestration moved intact from the research namespace.
    # Preserve its original policy identity, rather than granting compiler code
    # permission to read experiment definitions because it now shares a wheel.
    if path == "src/merlin/development/definition.py":
        return "packages/merlin-experiments/src/merlin_experiments/spec.py"
    if path == "src/merlin" or path.startswith("src/merlin/"):
        return "merlin/python/merlin" + path[len("src/merlin") :]
    for component in ("experiments", "analysis", "dse", "mining"):
        namespace = "src/merlin_" + component
        if path == namespace or path.startswith(namespace + "/"):
            return "packages/merlin-" + component + "/src/merlin_" + component + path[len(namespace) :]
    parts = path.split("/")
    if len(parts) >= 4 and parts[0] == "packages" and parts[2:4] == ["src", "merlin"]:
        return "/".join(("merlin", "python", "merlin", *parts[4:]))
    return path


def core_package(root: Path) -> Path:
    modern = root / "src/merlin"
    return modern if modern.is_dir() else root / "merlin/python/merlin"


def source_packages(root: Path) -> tuple[Path, ...]:
    """Package directories, preferring canonical src over its legacy compatibility symlink."""
    candidates = [*sorted((root / "src").glob("*")), root / "merlin/python/merlin"]
    candidates.extend(sorted((root / "packages").glob("*/src/*")))
    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        if path.is_dir() and path.name.isidentifier() and path.resolve() not in seen:
            seen.add(path.resolve())
            out.append(path)
    return tuple(out)


def logical_components(root: Path) -> dict:
    """Reviewed research boundaries; ordinary source placement grants no access."""

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate logical component key: {key}")
            result[key] = value
        return result

    path = root / "build_tools/logical_components.json"
    document = json.loads(path.read_text(), object_pairs_hook=unique)
    expected = {"merlin-experiments", "merlin-dse", "merlin-mining", "merlin-analysis"}
    if (
        set(document) != {"schema", "research"}
        or document["schema"] != "merlin.logical_components.v1"
        or type(document["research"]) is not dict
        or set(document["research"]) != expected
    ):
        raise ValueError("missing or unknown logical component inventory")
    seen = {}
    for owner, fields in document["research"].items():
        if type(fields) is not dict or set(fields) != {"namespaces", "modules"}:
            raise ValueError(f"invalid logical component fields: {owner}")
        for kind, values in fields.items():
            if type(values) is not list or len(values) != len(set(values)):
                raise ValueError(f"invalid logical component {kind}: {owner}")
            for value in values:
                if (
                    type(value) is not str
                    or not value
                    or any(not part.isidentifier() for part in value.split("."))
                    or value in seen
                ):
                    raise ValueError(f"invalid or overlapping logical component member: {value}")
                seen[value] = owner
    namespaces = [(name, owner) for owner, fields in document["research"].items() for name in fields["namespaces"]]
    if any(name != other and name.startswith(other + ".") for name, _ in namespaces for other, _ in namespaces):
        raise ValueError("overlapping logical component namespaces")
    if any(
        module.startswith(namespace + ".") or module == namespace
        for owner, fields in document["research"].items()
        for module in fields["modules"]
        for namespace, _ in namespaces
    ):
        raise ValueError("overlapping logical component module and namespace")
    return document["research"]


def component_owner(module: str, components: dict) -> str:
    for owner, fields in components.items():
        if module in fields["modules"] or any(
            module == prefix or module.startswith(prefix + ".") for prefix in fields["namespaces"]
        ):
            return owner
    return "core"


def module_name(path: Path, root: Path) -> str | None:
    """Import identity of source without importing core or an optional extension."""
    # Include both lexical core paths: callers may be inspecting historical grant/ledger entries.
    packages = (*source_packages(root), root / "src/merlin", root / "merlin/python/merlin")
    for package in packages:
        try:
            relative = path.relative_to(package.parent)
        except ValueError:
            continue
        if not relative.parts or relative.parts[0] != package.name:
            continue
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts)
    return None


def in_scope(rel: str, roots: tuple[str, ...]) -> bool:
    if not any(rel == scope or rel.startswith(scope + "/") for scope in roots):
        return False
    parts = Path(rel).parts
    if any(part in EXCLUDED_PARTS for part in parts):
        return False
    # Extension-owned tests/resources are not library code; their own gates inspect them separately.
    if parts and parts[0] == "packages":
        return len(parts) >= 5 and parts[2] == "src"
    return True


def python_files(root: Path, roots: tuple[str, ...]) -> list[Path]:
    """Return repo-relative files, pruning generated trees before descent and deduplicating aliases."""
    candidates = []
    for scope in roots:
        if scope == "packages":
            candidates.extend(path for path in source_packages(root) if "packages" in path.relative_to(root).parts)
        else:
            candidates.append(root / scope)
    out: list[Path] = []
    seen: set[Path] = set()
    for base in candidates:
        for directory, subdirs, files in os.walk(base):
            subdirs[:] = sorted(name for name in subdirs if name not in EXCLUDED_PARTS and not name.startswith("."))
            for name in sorted(files):
                path = Path(directory) / name
                rel = path.relative_to(root)
                if not name.endswith(".py") or not in_scope(rel.as_posix(), roots) or path.resolve() in seen:
                    continue
                seen.add(path.resolve())
                out.append(rel)
    return out


def scan_python_paths(root: Path, roots: tuple[str, ...], *, staged: bool) -> list[Path]:
    if not staged:
        return python_files(root, roots)
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [Path(rel) for rel in result.stdout.splitlines() if rel.endswith(".py") and in_scope(rel, roots)]
