#!/usr/bin/env python3
"""Gate: an instrument nothing calls is not an instrument.

A module that computes a quality signal is "done" when it has tests, and nothing has ever required
that production code call it. Measured: the module that reports what fraction of a model reached
the accelerator, the one that records why a capability was refused, and the one that names every
unjustified host placement each had tests and no caller, while the defects they exist to show went
unreported for a year. An unwired check reads exactly like a passing one.

This gate builds the import graph structurally (``ast``; a word search over-counts, because a
module's name appears in comments and docstrings of code that never imports it) and requires every
module under the instrumented packages to have a PRODUCTION importer: code under the library, the
experiments, the targets or the build tools, excluding the test suite and the module itself. A
declared console script counts as wired. Tests do not: a test proves a module works, not that
anything uses it.

Known debt lives in ``unwired_ratchet.txt`` beside this file, one repo-relative path per line. It
may only shrink (``check_ratchets_shrink.py`` holds every ``*_ratchet.txt``), and an entry for a
module that has since been wired or deleted fails this gate until it is removed, so the ledger
cannot rot into an allowlist.

    python build_tools/scripts/check_wiring.py            # exit 1 on a new unwired module
    python build_tools/scripts/check_wiring.py --write    # regenerate the ledger (review the diff)
    python build_tools/scripts/check_wiring.py --list     # print every unwired module
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _source_layout  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = _source_layout.core_package(ROOT).parent
LEDGER = ROOT / "build_tools" / "scripts" / "unwired_ratchet.txt"
#: Packages whose modules compute quality signals, verdicts or evidence.
INSTRUMENTED = tuple(
    f"{prefix}/{package}"
    for prefix in ("src/merlin", "merlin/python/merlin")
    for package in ("perf", "targetgen", "verify")
) + ("packages/merlin-experiments/src/merlin_experiments/evaluation",)
#: Where a production importer may live.
PRODUCTION = (*_source_layout.SOURCE_SCAN_ROOTS, "merlin/experiments", "merlin/targets", "build_tools")
EXCLUDED_PARTS = ("_data", "__pycache__", "tests", "_qa_ws")


def _python_files(relative_root: str) -> list[Path]:
    return [
        ROOT / path
        for path in _source_layout.python_files(ROOT, (relative_root,))
        if not any(part in EXCLUDED_PARTS for part in path.parts)
    ]


def _module_name(path: Path) -> str | None:
    module = _source_layout.module_name(path, ROOT)
    if module is not None:
        return module
    # Preserve the standalone worker-inspection seam, where callers supply a package root outside
    # the checkout rather than changing ROOT (e.g. a frozen worker source tree).
    try:
        relative = path.relative_to(PACKAGE_ROOT)
    except ValueError:
        return None
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imports(path: Path) -> set[str]:
    """Every dotted module name ``path`` imports, with relative imports resolved."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    own = _module_name(path)
    package = None
    if own is not None:
        package = own if path.name == "__init__.py" else own.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
            # A worker that runs under another interpreter cannot import the package, so it puts
            # its own directory on the path and imports a sibling by bare name. That is a
            # production import of the sibling module, and the file beside it is the evidence.
            for alias in node.names:
                if package and "." not in alias.name and (path.parent / f"{alias.name}.py").is_file():
                    found.add(f"{package}.{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                if package is None:
                    continue
                anchor = package.split(".")
                anchor = anchor[: len(anchor) - (node.level - 1)]
                base = ".".join([*anchor, base] if base else anchor)
            if base:
                found.add(base)
                found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


def _console_script_modules() -> set[str]:
    modules: set[str] = set()
    for pyproject in (ROOT / "pyproject.toml", *sorted((ROOT / "packages").glob("*/pyproject.toml"))):
        if not pyproject.is_file():
            continue
        in_scripts = False
        for line in pyproject.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("["):
                in_scripts = stripped == "[project.scripts]"
                continue
            if in_scripts and "=" in stripped:
                target = stripped.split("=", 1)[1].strip().strip('"')
                modules.add(target.split(":", 1)[0])
    return modules


def unwired() -> list[str]:
    candidates = {}
    # Evaluators can live in any separately installed shared-namespace distribution. Relocation
    # must not remove their obligation to have a production caller.
    extension_roots = (
        (package / part).relative_to(ROOT).as_posix()
        for package in _source_layout.source_packages(ROOT)
        if package.name == "merlin" and package.is_relative_to(ROOT / "packages")
        for part in ("perf", "targetgen", "verify")
    )
    for relative_root in (*INSTRUMENTED, *extension_roots):
        for path in _python_files(relative_root):
            if path.name == "__init__.py":
                continue
            name = _module_name(path)
            if name:
                candidates.setdefault(name, path.resolve())
    imported_by: dict[str, set[Path]] = {name: set() for name in candidates}
    for relative_root in PRODUCTION:
        for path in _python_files(relative_root):
            for name in _imports(path):
                if name in imported_by and candidates[name] != path.resolve():
                    imported_by[name].add(path)
    scripts = _console_script_modules()
    return sorted(
        str(candidates[name].relative_to(ROOT))
        for name, importers in imported_by.items()
        if not importers and name not in scripts
    )


def _ledger() -> list[str]:
    if not LEDGER.is_file():
        return []
    return [
        line.split("#", 1)[0].strip()
        for line in LEDGER.read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    ]


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    found = unwired()
    if "--list" in arguments:
        print("\n".join(found))
        return 0
    if "--write" in arguments:
        header = (
            "# Modules under the instrumented packages that no production code imports.\n"
            "# May only shrink: wire the module into a production path, or delete it.\n"
            "# growth-accepted: the gate is new; this is debt it discovered, not debt added.\n"
        )
        LEDGER.write_text(header + "".join(f"{path}\n" for path in found), encoding="utf-8")
        print(f"wrote {LEDGER.relative_to(ROOT)} ({len(found)} entries)")
        return 0
    known = set(_ledger())
    stable_found = {_source_layout.policy_path(path) for path in found}
    new = [path for path in found if path not in known and _source_layout.policy_path(path) not in known]
    stale = sorted(known - stable_found - set(found))
    for path in new:
        print(
            f"[FAIL] unwired: {path} has tests at most -- no production code imports it. Call it "
            f"from the path it was built for, or delete it."
        )
    for path in stale:
        print(
            f"[FAIL] stale ledger entry: {path} is wired or gone -- remove it from "
            f"{LEDGER.relative_to(ROOT)} so the ledger keeps shrinking."
        )
    if new or stale:
        return 1
    print(f"[  ok] wiring: {len(found)} known unwired module(s) in the ledger; no new one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
