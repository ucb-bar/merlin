"""Dependency-free source discovery shared by repository gates during the layout cutover.

Scan the physical source once, including separately packaged extensions. Historical debt remains
keyed by its original path: relocating a file neither forgives it nor grows a ratchet/allowlist.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SOURCE_SCAN_ROOTS = ("src/merlin", "merlin/python/merlin", "packages")
EXCLUDED_PARTS = frozenset({"_data", "__pycache__", ".venv", "venv", "_qa_ws", "build", "dist"})


def policy_path(path: str) -> str:
    """Stable debt identity for the same merlin module, including shared-namespace distributions."""
    if path == "src/merlin" or path.startswith("src/merlin/"):
        return "merlin/python/merlin" + path[len("src/merlin") :]
    parts = path.split("/")
    if len(parts) >= 4 and parts[0] == "packages" and parts[2:4] == ["src", "merlin"]:
        return "/".join(("merlin", "python", "merlin", *parts[4:]))
    return path


def core_package(root: Path) -> Path:
    modern = root / "src/merlin"
    return modern if modern.is_dir() else root / "merlin/python/merlin"


def source_packages(root: Path) -> tuple[Path, ...]:
    """Package directories, preferring canonical src over its legacy compatibility symlink."""
    candidates = [root / "src/merlin", root / "merlin/python/merlin"]
    candidates.extend(sorted((root / "packages").glob("*/src/*")))
    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        if path.is_dir() and path.name.isidentifier() and path.resolve() not in seen:
            seen.add(path.resolve())
            out.append(path)
    return tuple(out)


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
