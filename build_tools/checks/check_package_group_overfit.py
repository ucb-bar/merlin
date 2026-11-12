#!/usr/bin/env python3
"""Lint gate: a submitted compiler PACKAGE may not key its behavior on a whole-model group id, or
carry a hardcoded per-model shape table.

The phase-2 whole-model loop hands an authoring agent's package the WHOLE program to edit, group by
group, against a measured objective (see the phase-2 guardrails). That objective is supposed to reward
a general lowering -- a scheduler, a binding rule, a fusion pass -- not a program that recognizes
"this is group 70 of THIS captured model" and special-cases it. The second kind passes today's
measurement and answers nothing tomorrow: a different model, or the same model recaptured with its
groups renumbered, gets none of the win. Two structural patterns are refused, both detected on the
AST (no regex, per this repo's own rule) so a differently-spelled instance is still caught:

1. **Group-id keying.** A comparison, dict/set literal, or match-case whose key/comparator side is an
   integer constant while the other side names something containing "group" (``group``, ``group_id``,
   ``gid``, ``self.group``, ...) -- ``if group == 70:``, ``GROUP_FIX = {1: ..., 70: ...}``,
   ``match group_index: case 70:``. A package's OWN kernel objects are already named ``g<N>`` by the
   harness that links them (see ``src/merlin/analysis/perf/isa_prohibition.py``), so a group number
   legitimately appears in build output and file NAMES; what is refused is a group number used as a
   VALUE inside the package's OWN control flow.
2. **Hardcoded shape tables.** A literal container (list/tuple/dict) with several entries that are
   themselves small integer tuples of the same length (3 or 4) -- the shape of "one row per layer of
   one captured model". A real compiler derives a group's shape from the interface program it is
   handed each time, not from a table baked in at authoring time.

Both patterns are refused wherever they appear in a package's own Python source (every ``*.py`` file
under the package root except its ``devtools/`` development aids, which are not part of any declared
entrypoint). Run::

    python build_tools/checks/check_package_group_overfit.py --package <dir>   # one package
    python build_tools/checks/check_package_group_overfit.py --package <dir> --stop-hook
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ALLOW_FILE = ROOT / "build_tools" / "checks" / "package_group_overfit_allowlist.txt"
INLINE_MARKER = "# overfit-ok:"
#: Development aids a package's own manifest declares are not reachable through any entrypoint (see
#: the mlir_oot backend's own ``devtools/`` note); a probe script kept there for a human to run by hand
#: is not the compiler the gate exists to hold to this rule.
EXEMPT_DIRNAME = "devtools"
#: Shape-table entries this narrow: a tuple/list of this many small ints is "a shape", not "some ints".
SHAPE_TUPLE_LENGTHS = (3, 4)
#: A shape table needs at least this many same-shaped entries to be a table rather than one literal.
MIN_TABLE_ENTRIES = 4
GROUP_NAME_FRAGMENT = "group"


def _line_text(path: Path, lineno: int) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[lineno - 1]
    except (OSError, IndexError):
        return ""


def _is_inline_allowed(path: Path, lineno: int) -> bool:
    return INLINE_MARKER in _line_text(path, lineno)


def _names(node: ast.AST) -> list[str]:
    """Every identifier this expression is built from (``self.group_id`` -> ``["self", "group_id"]``)."""
    out: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            out.append(child.id)
        elif isinstance(child, ast.Attribute):
            out.append(child.attr)
    return out


def _names_group_like(node: ast.AST) -> bool:
    return any(GROUP_NAME_FRAGMENT in name.lower() for name in _names(node))


def _int_const(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    return None


def _parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    return parents


def _enclosing_group_like(node: ast.AST, parents: Mapping[int, ast.AST]) -> bool:
    """Whether ``node`` sits on the right-hand side of an assignment whose OWN target names a group
    (``GROUP_TABLE = {...}``), or lexically inside a function whose name or parameters do."""
    cursor: ast.AST | None = node
    while cursor is not None:
        if isinstance(cursor, (ast.Assign, ast.AnnAssign)):
            targets = cursor.targets if isinstance(cursor, ast.Assign) else [cursor.target]
            if any(_names_group_like(target) for target in targets):
                return True
        if isinstance(cursor, ast.FunctionDef) and (
            GROUP_NAME_FRAGMENT in cursor.name.lower()
            or any(GROUP_NAME_FRAGMENT in arg.arg.lower() for arg in cursor.args.args)
        ):
            return True
        cursor = parents.get(id(cursor))
    return False


def _group_id_keying_hits(tree: ast.AST) -> list[tuple[int, str]]:
    parents = _parent_map(tree)
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            sides = [node.left, *node.comparators]
            consts = [side for side in sides if _int_const(side) is not None]
            named = [side for side in sides if side not in consts]
            if consts and any(_names_group_like(side) for side in named):
                hits.append((node.lineno, f"compares a group-named value to {consts[0].value!r}"))
        elif isinstance(node, ast.Dict):
            int_keys = [key for key in node.keys if key is not None and _int_const(key) is not None]
            # A dict literal keyed by plain integers, several of them, is suspicious ONLY when
            # something in reach of it is itself named for a group -- its own assignment target, or
            # an enclosing function/parameter -- never for an int-keyed table in general (an opcode
            # table, a bit-width table): those name no group anywhere near them.
            if len(int_keys) >= 2 and _enclosing_group_like(node, parents):
                hits.append((node.lineno, "a group-named int-keyed table"))
        elif isinstance(node, ast.Match):
            subject_group_like = _names_group_like(node.subject)
            for case in node.cases:
                if (
                    subject_group_like
                    and isinstance(case.pattern, ast.MatchValue)
                    and _int_const(case.pattern.value) is not None
                ):
                    hits.append(
                        (case.pattern.lineno, f"matches a group-named subject against {case.pattern.value.value!r}")
                    )
    return hits


def _shape_table_hits(tree: ast.AST) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple, ast.Dict)):
            values = node.elts if isinstance(node, (ast.List, ast.Tuple)) else list(node.values)
            shapes = [
                v
                for v in values
                if isinstance(v, (ast.List, ast.Tuple))
                and len(v.elts) in SHAPE_TUPLE_LENGTHS
                and all(_int_const(e) is not None for e in v.elts)
            ]
            if len(shapes) >= MIN_TABLE_ENTRIES and len({len(v.elts) for v in shapes}) == 1:
                hits.append((node.lineno, f"{len(shapes)} literal shape-sized entries of the same length"))
    return hits


def scan_file(path: Path) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return []
    violations: list[dict[str, Any]] = []
    for lineno, why in _group_id_keying_hits(tree):
        if _is_inline_allowed(path, lineno):
            continue
        violations.append({"path": str(path), "line": lineno, "kind": "group_id_keying", "why": why})
    for lineno, why in _shape_table_hits(tree):
        if _is_inline_allowed(path, lineno):
            continue
        violations.append({"path": str(path), "line": lineno, "kind": "hardcoded_shape_table", "why": why})
    return violations


def scan_package(package: Path) -> list[dict[str, Any]]:
    """Every group-id-keying or hardcoded-shape-table hit in ``package``'s own Python source."""
    violations: list[dict[str, Any]] = []
    for path in sorted(package.rglob("*.py")):
        if EXEMPT_DIRNAME in path.relative_to(package).parts:
            continue
        violations.extend(scan_file(path))
    return violations


def _allowlisted(package: Path) -> set[str]:
    if not ALLOW_FILE.is_file():
        return set()
    return {
        line.split("#", 1)[0].strip()
        for line in ALLOW_FILE.read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    }


def refusal_line(violations: list[dict[str, Any]]) -> str:
    kinds = sorted({v["kind"] for v in violations})
    return "package_group_overfit: " + ", ".join(kinds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path, help="a candidate package's root directory")
    parser.add_argument("--stop-hook", action="store_true", help="emit Claude Code Stop-hook JSON")
    args = parser.parse_args(argv)
    package = args.package.resolve()
    if not package.is_dir():
        print(f"[FAIL] package-group-overfit: not a directory: {package}")
        return 1
    allow = _allowlisted(package)
    violations = [v for v in scan_package(package) if str(Path(v["path"]).resolve().relative_to(package)) not in allow]
    if args.stop_hook:
        if violations:
            print(
                json.dumps(
                    {
                        "decision": "block",
                        "reason": refusal_line(violations)
                        + ":\n- "
                        + "\n- ".join(f"{v['path']}:{v['line']}: {v['why']}" for v in violations),
                    }
                )
            )
        else:
            print(json.dumps({}))
        return 0
    if violations:
        print(f"[FAIL] package-group-overfit: {len(violations)} hit(s):")
        for v in violations:
            print(f"  - {v['path']}:{v['line']}: {v['kind']} ({v['why']})")
        return 1
    print("[  ok] package-group-overfit: no group-id keying or hardcoded shape table in this package.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
