"""Static source-package layout for newly published compiler backends.

This checks organization, not numerical correctness or build success. It never
imports candidate code. Editable SDK inputs belong to the author workspace;
they are not copies to include in the backend's published source package.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

CURRENT_LAYOUT = "3.0"
HISTORICAL_LAYOUTS = frozenset({"1.0", "2.0"})
_GENERATED = frozenset(
    {"build", "out", "results", "runs", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv"}
)
_COPIED_SDK = frozenset({"compiler_library", "generic-library", "merlin"})
_BINARY = frozenset({".pyc", ".o", ".obj", ".a", ".so", ".dll", ".exe", ".elf", ".bin"})
_NATIVE_SOURCE = frozenset({".c", ".cc", ".cpp", ".cxx", ".td"})


class SourceLayoutError(ValueError):
    """A compiler source publication does not meet the selected host layout."""


def _qualified_name(value: str) -> bool:
    return all(part.isascii() and part.isidentifier() for part in value.split("."))


def _entrypoint(root: Path, declaration: str) -> str:
    if not isinstance(declaration, str):
        raise SourceLayoutError("Python entrypoints must be module or module:attribute strings")
    module, separator, attribute = declaration.partition(":")
    if not _qualified_name(module) or (separator and not _qualified_name(attribute)):
        raise SourceLayoutError(f"invalid Python entrypoint: {declaration!r}")
    stem = root / "src" / Path(*module.split("."))
    if not stem.with_suffix(".py").is_file() and not (stem / "__init__.py").is_file():
        raise SourceLayoutError(f"entrypoint {module!r} has no source under src/")
    return module


def python_layout(root: Path) -> dict:
    """Check the declared installed Python shape without executing its build backend."""
    try:
        document = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise SourceLayoutError("Python backend requires a readable pyproject.toml") from exc
    project = document.get("project", {})
    if not isinstance(project, dict) or not isinstance(project.get("name"), str) or not project["name"].strip():
        raise SourceLayoutError("Python backend requires project.name in pyproject.toml")
    tools = document.get("tool", {})
    setuptools = tools.get("setuptools", {})
    package_dirs = setuptools.get("package-dir", {})
    finder = setuptools.get("packages", {})
    where = finder.get("find", {}).get("where", []) if isinstance(finder, dict) else []
    hatch = tools.get("hatch", {}).get("build", {}).get("targets", {}).get("wheel", {}).get("packages", [])
    src_layout = (
        (
            isinstance(package_dirs, dict)
            and bool(package_dirs)
            and all(
                isinstance(value, str) and (value == "src" or value.startswith("src/"))
                for value in package_dirs.values()
            )
        )
        or where == ["src"]
        or (bool(hatch) and all(isinstance(value, str) and value.startswith("src/") for value in hatch))
    )
    if not src_layout or not (root / "src").is_dir():
        raise SourceLayoutError("Python backend must declare its installed packages under src/")
    entrypoints = project.get("entry-points", {}).get("merlin.backends", {})
    if not isinstance(entrypoints, dict) or not entrypoints:
        raise SourceLayoutError("Python backend must declare a merlin.backends entrypoint")
    modules = [_entrypoint(root, value) for value in entrypoints.values()]
    scripts = project.get("scripts", {})
    if not isinstance(scripts, dict):
        raise SourceLayoutError("project.scripts must be a table")
    for declaration in scripts.values():
        _entrypoint(root, declaration)
    return {"kind": "python", "project": project["name"], "backend_modules": sorted(set(modules))}


def _native_layout(root: Path) -> dict:
    if not (root / "CMakeLists.txt").is_file():
        raise SourceLayoutError("native backend requires CMakeLists.txt")
    sources = [
        path.relative_to(root).as_posix()
        for owner in (root / "lib", root / "tools")
        if owner.is_dir()
        for path in owner.rglob("*")
        if path.is_file() and path.suffix in _NATIVE_SOURCE
    ]
    if not sources:
        raise SourceLayoutError("native implementation sources belong under lib/ or tools/")
    # include/ is needed only when the package has public headers; no empty
    # include/lib/tools scaffolding is required merely to satisfy this check.
    return {"kind": "native", "sources": sorted(sources)}


def require_source_layout(root: Path) -> dict:
    """Require the current source shape, including when packaging headers are absent."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise SourceLayoutError("compiler publication requires a source directory")
    for directory, folders, files in os.walk(root):
        folders[:] = [name for name in folders if name != ".git"]
        for name in (*folders, *files):
            relative = (Path(directory) / name).relative_to(root)
            if name in _COPIED_SDK:
                raise SourceLayoutError(f"embedded Merlin/SDK copy is not a backend source: {relative}")
            if name in _GENERATED or name.endswith(".egg-info"):
                raise SourceLayoutError(f"generated run/build payload must stay outside publication: {relative}")
            if Path(name).suffix in _BINARY:
                raise SourceLayoutError(f"generated binary is not compiler source: {relative}")
    if (root / "pyproject.toml").is_file():
        result = python_layout(root)
    elif (root / "CMakeLists.txt").is_file():
        result = _native_layout(root)
    else:
        raise SourceLayoutError(
            "new backend source requires pyproject.toml with src/ entrypoints or a native CMake layout"
        )
    return {"layout_version": CURRENT_LAYOUT, "status": "conforming_source_layout", **result}


def publication_layout(root: Path, version: str) -> dict:
    """Apply a host-selected export version; the candidate cannot turn this off."""
    if version in HISTORICAL_LAYOUTS:
        return {"layout_version": version, "status": "historical_payload_layout_not_validated"}
    if version != CURRENT_LAYOUT:
        raise SourceLayoutError(f"unsupported host source-layout version: {version!r}")
    return require_source_layout(root)
