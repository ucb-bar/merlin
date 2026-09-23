#!/usr/bin/env python3
"""Gate: no two target packages implement the same function.

The cardinal rule has a gate for one direction -- a target's name appearing in generic code -- and none
for the other: GENERIC LOGIC living inside a target package. That is the same overfit seen from the
other side, and it is invisible while there is one target, because nothing is duplicated yet. It becomes
visible at exactly the moment the second target copies the function, and that is the moment to catch it:
before the two copies drift, which is when the cost is paid.

WHY A DUPLICATION CHECK RATHER THAN A HEURISTIC. The obvious rule -- "a function in a target package
that imports core and names no target is misplaced" -- was measured on this tree before this gate was
written: 36 of 602 functions match it, and most are genuine target glue. A gate that is mostly false
positives gets everything allowlisted and enforces nothing; this repo has the scar tissue to prove it.
Duplication is the opposite: measured on this tree it flags ZERO, so every finding is a real copy.

COMPARISON IS BY SHAPE, NOT BY TEXT. Identifiers, attribute names, constants and argument names are
erased before hashing, so a copy that renamed its variables and swapped its constants still matches.
Renaming is exactly what someone does while copying, which is why comparing text would miss it.

Once target implementations have been extracted to OOT providers, the in-tree comparison is
not applicable. That state is accepted only when the tracked migration manifest declares the
extraction and no Python implementation remains under ``merlin/targets``. An unrecorded empty
scan still refuses: a gate that could not run must never report success.

Exit codes: 0 clean or recorded extraction with no in-tree implementation, 1 a shape is
implemented in more than one target package, 2 CANNOT DECIDE.
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_GATE = "target-package-duplication"
RATCHET = Path(__file__).resolve().parent / "target_package_duplication_ratchet.txt"

#: Below this a shared shape says nothing: two three-line accessors that both return a dict entry are
#: not a copied implementation, they are the same obvious line written twice.
MIN_LINES = 6


class _Shape(ast.NodeTransformer):
    """Erase what a copier would rename, keep what they would not: the control and call structure."""

    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id="_", ctx=node.ctx), node)

    def visit_Attribute(self, node):
        self.generic_visit(node)
        return ast.copy_location(ast.Attribute(value=node.value, attr="_", ctx=node.ctx), node)

    def visit_Constant(self, node):
        return ast.copy_location(ast.Constant(value=0), node)

    def visit_arg(self, node):
        return ast.copy_location(ast.arg(arg="_", annotation=None), node)

    def visit_FunctionDef(self, node):
        # The function's OWN name has to go too. Leaving it in was the first version of this gate, and
        # it missed every renamed copy -- which is to say, every copy anyone actually makes.
        self.generic_visit(node)
        node.name = "_"
        node.decorator_list = []
        node.returns = None
        return node


def shape_of(fn: ast.FunctionDef) -> str:
    normalised = _Shape().visit(ast.parse(ast.unparse(fn)))
    return hashlib.sha256(ast.dump(normalised).encode()).hexdigest()[:16]


def shapes(root: Path | None = None) -> dict[str, list[tuple[str, str, str]]]:
    """``{shape: [(package, file, function)]}`` over every target package's code."""
    base = root or ROOT
    found: dict[str, list[tuple[str, str, str]]] = collections.defaultdict(list)
    for path in sorted(base.glob("merlin/targets/*/**/*.py")):
        package = path.relative_to(base / "merlin" / "targets").parts[0]
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
            try:
                if len(ast.unparse(fn).splitlines()) < MIN_LINES:
                    continue
                found[shape_of(fn)].append((package, path.name, fn.name))
            except (RecursionError, ValueError):
                continue
    return dict(found)


def verdict(found: dict[str, list[tuple[str, str, str]]], ratcheted: set[str]) -> tuple[list[str], int]:
    """``(problems, rc)`` -- pure, so a test can plant a duplicate without touching the tree."""
    problems: list[str] = []
    for shape, sites in sorted(found.items()):
        packages = {pkg for pkg, _, _ in sites}
        if len(packages) < 2 or shape in ratcheted:
            continue
        where = ", ".join(f"{pkg}/{file}:{fn}" for pkg, file, fn in sorted(sites))
        problems.append(
            f"{shape}: the same implementation appears in {len(packages)} target packages -- {where}. "
            "Generic logic in a target package is overfit seen from the other side: move it to the core, "
            "where the no-target-name gate covers it, and leave each target only what it spells "
            "differently."
        )
    return problems, (1 if problems else 0)


def _ratchet() -> set[str]:
    if not RATCHET.is_file():
        return set()
    return {
        line.split("#", 1)[0].strip()
        for line in RATCHET.read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    }


def _recorded_extraction(root: Path) -> bool:
    manifest = root / "build_tools" / "upstreams" / "target_support.json"
    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return record.get("canonical_sources_removed") is True and bool(record.get("companions"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--stop-hook", action="store_true")
    args = parser.parse_args(argv)

    targets = ROOT / "merlin" / "targets"
    found = shapes()
    if not found:
        remaining_code = sorted(targets.rglob("*.py")) if targets.is_dir() else []
        if not remaining_code and _recorded_extraction(ROOT):
            text = (
                f"{_GATE}: no in-tree target Python implementations; OOT extraction is recorded. "
                "Cross-provider duplication requires the selected companion checkouts."
            )
            if args.json:
                print(json.dumps({"status": "not_applicable", "shapes": 0, "problems": [], "reason": text}, indent=2))
            elif args.stop_hook:
                print(json.dumps({"decision": "approve", "reason": text}))
            else:
                print(f"[  ok] {text}")
            return 0
        text = (
            f"{_GATE}: no functions were parsed out of {targets}; "
            "unrecorded extraction or unreadable target code is not a clean scan."
        )
        if args.stop_hook:
            print(json.dumps({"decision": "block", "reason": text}))
            return 0
        print(f"[FAIL] {text}", file=sys.stderr)
        return 2

    problems, rc = verdict(found, _ratchet())
    if args.json:
        print(json.dumps({"shapes": len(found), "problems": problems}, indent=2))
        return rc
    for problem in problems:
        print(f"[DEBT] {_GATE}: {problem}")
    packages = len({pkg for sites in found.values() for pkg, _, _ in sites})
    print(
        f"[{'FAIL' if problems else '  ok'}] {_GATE}: {len(found)} function shape(s) across {packages} "
        f"target package(s); {len(problems)} implemented in more than one."
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
