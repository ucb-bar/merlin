#!/usr/bin/env python3
"""Lint gate: no hardcoded TARGET NAME coupling in the core library + build tooling.

The capsule-bench stack is target-agnostic: the target is resolved at runtime from the descriptor /
contract / manifest / registry, never baked into core logic. This gate enforces that the specific
target names this repo ships never silently reappear as *operative* coupling in the in-scope trees:

  * ``merlin/python/merlin/**``
  * ``merlin/contract/**``
  * ``build_tools/scripts/**``

Three coupling surfaces are caught (all detected structurally with ``ast`` — this checker uses no
regex on itself, honoring the sibling no-regex gate):

  1. **String literals** — an ``ast.Constant`` string whose value contains one of the known target
     names as a whole identifier (word-boundary, so ``gemmini`` inside ``gemmini_kernel`` does NOT
     match, but ``mx_gemmini`` matches its own entry). **Docstrings are exempt** (a target named as
     documentation/example is allowed), and **comments are not in the AST at all**.
  2. **Import paths** — an ``ast.Import`` / ``ast.ImportFrom`` whose dotted module names a target as a
     token-run (``import merlin.targets.gemmini.backend`` couples the importer to one target even with
     zero string literals).
  3. **Module filename** — the scanned module's OWN repo-relative path names a target as a token-run
     (``cost_model/gemmini.py`` is a gemmini-specific module inside the shared library; the identity
     hides in the filename, not a literal).

For surfaces (2)/(3) a *token-run* match is used: the path/import is split on ``/`` and ``.`` into
segments, each segment split on ``_`` into tokens, and a target name matches when ITS ``_``-split
token sequence appears as a contiguous run (so ``gemmini_dispatcher`` and ``saturn_vec_codegen`` are
caught, and ``npu_model`` matches only a real ``npu``+``model`` run — never a lone ``model``).

Allowed exceptions (checked in order):

  1. an inline ``# target-ok: <rationale>`` comment on the offending line (preferred — co-located;
     works for a string-literal or an import line; a filename hit has no line, so allowlist it);
  2. an entry in ``build_tools/scripts/target_name_allowlist.txt`` — either an exact repo-relative
     ``.py`` path, or a directory prefix ending in ``/`` (matches a whole reference-target subtree).
     An allowlisted file is skipped on ALL three surfaces.

The allowlist only ever *shrinks*. Its two legitimate populations are (a) the in-tree REFERENCE
target implementations (a target's own backend/eval module naturally names itself, in its filename
and in the modules that still import it) pending eviction to a published package via
``MERLIN_TARGET_PATH`` (OV11), and (b) CLI convenience defaults where the reference target is the
documented example. As a reference target is evicted or a default removed, delete its entry. Run::

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
SCAN_ROOTS = ("merlin/python/merlin", "merlin/contract", "build_tools/scripts")
# Path fragments that mark BUILD-GENERATED (gitignored) trees, not source — never scanned.
EXCLUDE_FRAGMENTS = ("/_data/",)
INLINE_MARKER = "# target-ok:"

# The concrete target names this repo ships. A core module must resolve the target at runtime, never
# hardcode one of these. ``toy_npu`` is deliberately NOT here — it is the kept how-to reference target
# (the onboarding example), and ``rvv`` is a generic ISA class, not a target.
TARGET_NAMES = frozenset({
    "gemmini", "mx_gemmini", "atlas", "radiance", "saturn", "muon", "npu_model",
})
# Precompute each name's ``_``-split token sequence for token-run matching on paths/imports.
_TARGET_TOKEN_SEQS = {name: name.split("_") for name in TARGET_NAMES}
# Per-target DATA subtrees (capsule corpora authored per hardware family) — the sanctioned per-target
# data dirs, NOT shared library code, so they are out of scan scope (a target dir is an allowed edge).
_DATA_SUBTREES = tuple(f"merlin/contract/capsules/{t}/" for t in TARGET_NAMES)


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


def _seq_in(hay: list[str], needle: list[str]) -> bool:
    """True if ``needle`` appears as a contiguous sublist of ``hay`` (both are token lists)."""
    n, m = len(hay), len(needle)
    if m == 0 or m > n:
        return False
    return any(hay[i:i + m] == needle for i in range(n - m + 1))


def _path_target_names(dotted: str) -> list[str]:
    """Target names that appear as a token-run in a ``/``- or ``.``-separated path/import string.
    Split into segments on ``/`` and ``.``, each segment into ``_`` tokens, then match each target's
    own token sequence as a contiguous run. Structural (no regex)."""
    found: set[str] = set()
    for seg in dotted.replace("/", ".").split("."):
        toks = seg.split("_")
        for name, seq in _TARGET_TOKEN_SEQS.items():
            if _seq_in(toks, seq):
                found.add(name)
    return sorted(found)


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
    """Collect target-name coupling: string-literal Constants (skipping docstrings) and import module
    paths. Each hit is ``(lineno, name, snippet, kind)`` with kind ``literal`` or ``import``."""

    def __init__(self) -> None:
        self.docstrings: set[int] = set()      # id() of Constant nodes that are docstrings
        self.hits: list[tuple[int, str, str, str]] = []

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
                    self.hits.append((node.lineno, name, node.value.strip()[:60], "literal"))
                    break
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            for name in _path_target_names(alias.name):
                self.hits.append((node.lineno, name, alias.name, "import"))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # Check both the module path (``from a.gemmini_x import y``) AND each imported symbol name
        # (``from a import saturn_vec``) — either couples the importer to one target.
        mods = [node.module] if node.module else []
        mods += [alias.name for alias in node.names]
        for mod in mods:
            for name in _path_target_names(mod):
                self.hits.append((node.lineno, name, mod, "import"))
        self.generic_visit(node)


def _scan_file(path: Path, rel: str) -> list[tuple[int, str, str, str]]:
    """Target-name coupling in ``path`` (string literals + imports + the module's own filename) not
    silenced by an inline ``# target-ok:`` marker. ``rel`` is the repo-relative path (for the
    filename surface, which has no line to carry an inline marker)."""
    out: list[tuple[int, str, str, str]] = []
    # Surface 3: the module's own filename names a target.
    for name in _path_target_names(rel):
        out.append((0, name, rel, "filename"))

    src = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return out
    v = _TargetNameVisitor()
    v.visit(tree)
    lines = src.splitlines()
    for lineno, name, snippet, kind in sorted(set(v.hits)):
        line = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
        if INLINE_MARKER not in line:
            out.append((lineno, name, snippet, kind))
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
        if any(rel.startswith(p) for p in _DATA_SUBTREES):
            continue  # per-target capsule corpus (a sanctioned per-target data dir), not library code
        if any(rel.startswith(r + "/") or rel == r for r in SCAN_ROOTS):
            targets.append(Path(rel))
    return targets


def _allowed(relstr: str, exact: set[str], prefixes: list[str]) -> bool:
    return relstr in exact or any(relstr.startswith(p) for p in prefixes)


_KIND_HINT = {
    "literal": "resolve the target at runtime, add `# target-ok: <why>`, or allowlist the file",
    "import": "import a target via the plugin/registry, add `# target-ok: <why>`, or allowlist",
    "filename": "this is a target-specific module in the shared library — evict it to the target's "
                "package (MERLIN_TARGET_PATH), or allowlist it as a pending-eviction reference",
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    staged = "--staged" in argv
    stop_hook = "--stop-hook" in argv
    exact, prefixes = _load_allowlist()

    violations: list[str] = []
    for rel in _iter_targets(staged):
        relstr = rel.as_posix()
        if _allowed(relstr, exact, prefixes):
            continue
        for lineno, name, snippet, kind in _scan_file(ROOT / rel, relstr):
            where = f"{relstr}:{lineno}" if lineno else f"{relstr} (module filename)"
            violations.append(f"{where}: target-name {kind} {name!r} in {snippet!r} "
                              f"({_KIND_HINT[kind]})")

    if stop_hook:
        if violations:
            print(json.dumps({"decision": "block",
                              "reason": ("Hardcoded target coupling outside the allowlist (see "
                                         "build_tools/scripts/target_name_allowlist.txt):\n- "
                                         + "\n- ".join(violations))}))
        else:
            print(json.dumps({}))
        return 0  # stop-hook signals via JSON, not exit code

    if violations:
        print(f"[FAIL] no-target-name: {len(violations)} target coupling(s) outside the allowlist:")
        for v in violations:
            print(f"  - {v}")
        return 1
    n_allow = len(exact) + len(prefixes)
    print(f"[  ok] no-target-name: {n_allow} allowlisted entr(y/ies); no stray target coupling in scope.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
