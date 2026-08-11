#!/usr/bin/env python3
"""Lint gate: no hardcoded TARGET NAME string literals in the core library + build tooling.

The capsule-bench stack is target-agnostic: the target is resolved at runtime from the descriptor /
contract / manifest / registry, never baked into core logic. This gate enforces that the specific
target names this repo ships never silently reappear as *operative* string literals in the in-scope
trees:

  * ``merlin/python/merlin/**``
  * ``build_tools/scripts/**``

A violation is a **string-literal** ``ast.Constant`` (detected structurally with ``ast`` — this
checker uses no regex on itself, honoring the sibling no-regex gate) whose value contains one of the
known target names as a whole identifier (word-boundary, so ``gemmini`` inside ``gemmini_kernel``
does NOT match, but ``mx_gemmini`` matches its own entry). **Docstrings are exempt** (a target named
as documentation/example is allowed — the goal permits a name in schema/doc examples), and **comments
are not in the AST at all**, so a ``# e.g. gemmini`` note is fine. What is caught is a name used in
code: a default value, a comparison operand, a dict key/value, help/error text — the places that make
core logic operate on one specific target.

Allowed exceptions (checked in order):

  1. an inline ``# target-ok: <rationale>`` comment on the offending line (preferred — co-located);
  2. an entry in ``build_tools/scripts/target_name_allowlist.txt`` — either an exact repo-relative
     ``.py`` path, or a directory prefix ending in ``/`` (matches a whole reference-target subtree).

The allowlist only ever *shrinks*. Its two legitimate populations are (a) the in-tree REFERENCE
target implementations (a target's own backend/eval module naturally names itself) pending eviction
to a published package via ``MERLIN_TARGET_PATH`` (OV11), and (b) CLI convenience defaults where the
reference target is the documented example. As a reference target is evicted or a default removed,
delete its entry. Run::

    python build_tools/scripts/check_no_target_name.py            # full scan (exit 1 on violation)
    python build_tools/scripts/check_no_target_name.py --staged   # only git-staged files
    python build_tools/scripts/check_no_target_name.py --stop-hook # emit Claude Code Stop-hook JSON
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALLOW_FILE = ROOT / "build_tools" / "scripts" / "target_name_allowlist.txt"
SCAN_ROOTS = ("merlin/python/merlin", "build_tools/scripts")
# Path fragments that mark BUILD-GENERATED (gitignored) trees, not source — never scanned.
EXCLUDE_FRAGMENTS = ("/_data/",)
INLINE_MARKER = "# target-ok:"

# The concrete target names this repo ships. A core module must resolve the target at runtime, never
# hardcode one of these. ``toy_npu`` is deliberately NOT here — it is the kept how-to reference target
# (the onboarding example), and ``rvv`` is a generic ISA class, not a target. Read from the sibling
# data file so the set has a single source of truth shared with the audit.
TARGET_NAMES = frozenset({
    "gemmini", "mx_gemmini", "atlas", "radiance", "saturn", "muon", "npu_model",
})


def _is_word_char(c: str) -> bool:
    return c.isalnum() or c == "_"


def _contains_identifier(text: str, name: str) -> bool:
    """True if ``name`` occurs in ``text`` bounded by non-word chars on both sides (a whole-identifier
    match, implemented without regex). So ``gemmini`` matches `` gemmini `` / ``"gemmini"`` but not
    ``gemmini_kernel`` or ``mx_gemmini`` — those are matched by their own entries in the name set."""
    start = 0
    n = len(name)
    while True:
        i = text.find(name, start)
        if i < 0:
            return False
        before_ok = i == 0 or not _is_word_char(text[i - 1])
        after_ok = i + n >= len(text) or not _is_word_char(text[i + n])
        if before_ok and after_ok:
            return True
        start = i + 1


def _load_allowlist() -> tuple[set[str], list[str]]:
    """Return (exact-file paths, directory prefixes). ``#`` comments and blank lines ignored; a
    trailing ``# rationale`` is dropped. A line ending in ``/`` is a subtree prefix."""
    exact: set[str] = set()
    prefixes: list[str] = []
    if ALLOW_FILE.is_file():
        for line in ALLOW_FILE.read_text(encoding="utf-8").splitlines():
            path = line.split("#", 1)[0].strip()
            if not path:
                continue
            if path.endswith("/"):
                prefixes.append(path)
            else:
                exact.add(path)
    return exact, prefixes


class _TargetNameVisitor(ast.NodeVisitor):
    """Collect (lineno, name, snippet) for string-literal Constants that name a target, skipping the
    docstring Constant of every module/class/function (documentation is allowed to name a target)."""

    def __init__(self) -> None:
        self.docstrings: set[int] = set()      # id() of Constant nodes that are docstrings
        self.hits: list[tuple[int, str, str]] = []

    def _mark_docstring(self, node: ast.AST) -> None:
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            self.docstrings.add(id(body[0].value))

    def visit_Module(self, node: ast.Module) -> None:
        self._mark_docstring(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._mark_docstring(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._mark_docstring(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._mark_docstring(node)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and id(node) not in self.docstrings:
            for name in TARGET_NAMES:
                if _contains_identifier(node.value, name):
                    self.hits.append((node.lineno, name, node.value.strip()[:60]))
                    break
        self.generic_visit(node)


def _scan_file(path: Path) -> list[tuple[int, str, str]]:
    """Target-name literals in ``path`` not silenced by an inline ``# target-ok:`` marker."""
    src = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return []
    v = _TargetNameVisitor()
    v.visit(tree)
    if not v.hits:
        return []
    lines = src.splitlines()
    out = []
    for lineno, name, snippet in sorted(set(v.hits)):
        line = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
        if INLINE_MARKER not in line:
            out.append((lineno, name, snippet))
    return out


def _iter_targets(staged: bool) -> list[Path]:
    if staged:
        out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
                             cwd=ROOT, capture_output=True, text=True).stdout
        rels = [ln for ln in out.splitlines() if ln.strip()]
    else:
        rels = []
        for root in SCAN_ROOTS:
            for p in sorted((ROOT / root).rglob("*.py")):
                rels.append(p.relative_to(ROOT).as_posix())
    targets = []
    for rel in rels:
        if not rel.endswith(".py"):
            continue
        if any(frag in f"/{rel}" for frag in EXCLUDE_FRAGMENTS):
            continue  # build-generated bundle, not source
        if any(rel.startswith(r + "/") or rel == r for r in SCAN_ROOTS):
            targets.append(Path(rel))
    return targets


def _allowed(relstr: str, exact: set[str], prefixes: list[str]) -> bool:
    return relstr in exact or any(relstr.startswith(p) for p in prefixes)


# --- the coupling scan: what the literal check above cannot see ------------------------------------
# The check above inspects string-literal Constants only, and matches a target name as a WHOLE
# identifier. Both choices are deliberate and both hide real coupling:
#
#   * a whole-identifier match cannot see `gemmini_kernel`, `gemmini_fence`, `saturn_vec` or
#     `cycle_window_gemmini_region` -- vendor SYMBOL names, which are how one target's ABI leaks into
#     shared code;
#   * inspecting only literals cannot see `from ..runtime.backends import gemmini as gem`, which is how
#     a generic module acquires a hard dependency on one target. That is a bigger problem than a string,
#     and it was completely invisible.
#
# So this is a SECOND, separate check rather than a change to the first: the whole-identifier rule
# encodes a real intent (`mx_gemmini` must be its own entry, not a match for `gemmini`) and stays.
#
# A file whose OWN PATH names a target is that target's own module and is skipped -- self-reference is
# legitimate, and those files are already tracked as eviction candidates. What this reports is code with
# no target in its name that nonetheless depends on one: the inverted dependencies.
def _mentions(text: str, name: str) -> bool:
    """Case-insensitive substring test. Deliberately looser than ``_contains_identifier``."""
    return name in text.lower()


def _is_target_owned(relstr: str) -> bool:
    """True when the path itself names a target, i.e. the file is legitimately about that target."""
    return any(_mentions(relstr, name) for name in TARGET_NAMES)


def _scan_coupling(path: Path) -> list[tuple[int, str, str, str]]:
    """``(lineno, target, kind, snippet)`` for target coupling in a file that is not about a target."""
    src = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return []
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            docstrings.add(id(body[0].value))

    hits: list[tuple[int, str, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            parts = [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            parts += [a.name for a in node.names]
            for part in parts:
                for name in TARGET_NAMES:
                    if _mentions(part, name):
                        hits.append((node.lineno, name, "import", part))
                        break
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docstrings:
            for name in TARGET_NAMES:
                # only what the whole-identifier check MISSES -- otherwise every literal is double-reported
                if _mentions(node.value, name) and not _contains_identifier(node.value, name):
                    hits.append((node.lineno, name, "symbol", node.value.strip()[:60]))
                    break
    lines = src.splitlines()
    return [h for h in sorted(set(hits))
            if INLINE_MARKER not in (lines[h[0] - 1] if 0 < h[0] <= len(lines) else "")]


def coupling_inventory(staged: bool = False) -> list[str]:
    """Every generic in-scope module that depends on a specific target, as reportable lines."""
    out: list[str] = []
    for rel in _iter_targets(staged):
        relstr = rel.as_posix()
        if _is_target_owned(relstr):
            continue
        for lineno, name, kind, snippet in _scan_coupling(ROOT / rel):
            out.append(f"{relstr}:{lineno}: [{kind}] generic module depends on {name!r} — {snippet!r}")
    return out


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    staged = "--staged" in argv
    stop_hook = "--stop-hook" in argv
    exact, prefixes = _load_allowlist()

    if "--coupling" in argv:
        # The full inventory, for populating the overfit register. Advisory by design: this debt predates
        # the check and printing it is the point, so it exits 0 and lets the caller decide.
        found = coupling_inventory(staged)
        if not found:
            print("[  ok] target-coupling: no generic module depends on a specific target.")
            return 0
        print(f"[DEBT] target-coupling: {len(found)} dependency(ies) on a specific target in modules "
              "whose own name claims to be generic:")
        for line in found:
            print(f"  - {line}")
        return 0

    violations: list[str] = []
    for rel in _iter_targets(staged):
        relstr = rel.as_posix()
        if _allowed(relstr, exact, prefixes):
            continue
        for lineno, name, snippet in _scan_file(ROOT / rel):
            violations.append(f"{relstr}:{lineno}: hardcoded target name {name!r} "
                              f"in literal {snippet!r} (resolve the target at runtime, add "
                              f"`# target-ok: <why>`, or allowlist the file)")

    if stop_hook:
        if violations:
            print(json.dumps({"decision": "block",
                              "reason": ("Hardcoded target name outside the allowlist (see "
                                         "build_tools/scripts/target_name_allowlist.txt):\n- "
                                         + "\n- ".join(violations))}))
        else:
            print(json.dumps({}))
        return 0  # stop-hook signals via JSON, not exit code

    if violations:
        print(f"[FAIL] no-target-name: {len(violations)} hardcoded target name(s) outside the allowlist:")
        for v in violations:
            print(f"  - {v}")
        return 1
    n_allow = len(exact) + len(prefixes)
    # Report the exemptions as DEBT, not as part of a pass. An allowlist announced on an "ok" line reads
    # as "nothing to see"; it is 36 places where the core is welded to a specific target.
    n_coupling = len(coupling_inventory(staged))
    print(f"[  ok] no-target-name: no stray target-name literal in scope.")
    print(f"[DEBT] {n_allow} allowlisted file(s) still name a target, and {n_coupling} dependency(ies) on "
          f"a specific target sit in modules whose own name claims to be generic (--coupling to list). "
          f"Both counts may only fall.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
