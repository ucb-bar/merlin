"""Shared checks for the author skill tests: paths, front matter, target-name screening, imports."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

from merlin.common.paths import repo_root

SKILLS = repo_root() / "packages/merlin-experiments/src/merlin_experiments/phase1/author_skills"
#: Hand-written skill bodies stay short; the generated index may be slightly larger.
BODY_LIMIT = 4600


def gate():
    """The no-target-name gate module, so the screen uses its own derived roster."""
    scripts = str(repo_root() / "build_tools" / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import check_no_target_name

    return check_no_target_name


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def front_matter(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), path
    head, body = text[4:].split("\n---\n", 1)
    fields = {}
    for line in head.splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields, body


def target_mentions(folder: Path) -> list[tuple[str, str]]:
    """``(file, name)`` for every skill file naming a target or substrate (case-insensitive, whole word)."""
    module = gate()
    names = {name.lower() for name in module.TARGET_NAMES | module.SUBSTRATE_NAMES}
    hits = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix in (".md", ".py", ".json", ".txt"):
            low = path.read_text(encoding="utf-8").lower()
            hits += [
                (path.relative_to(folder).as_posix(), name)
                for name in sorted(names)
                if module._contains_identifier(low, name)
            ]
    return hits


def top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots
