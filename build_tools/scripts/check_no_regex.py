#!/usr/bin/env python3
"""Lint gate: no ``re`` (regex) usage in the core library + build tooling.

This repo's principle is that facts are *compiled/derived from structure*, not scraped with
pattern-matching. This check enforces that regex does not silently return to the in-scope trees:

  * ``merlin/python/merlin/**``
  * ``merlin/contract/**``
  * ``build_tools/scripts/**``

A regex *call site* is any call to the ``re`` module (``re.compile``/``search``/``sub``/…) — reached
via ``import re``, ``import re as X``, or ``from re import …`` — detected structurally with the
``ast`` module (this checker uses no regex on itself). Because every compiled pattern begins with a
``re.compile(...)`` call, flagging module/alias calls catches compiled-pattern usage too.

Allowed exceptions are the genuinely-irreducible cases (filename conventions, external-tool stdout
with no ``--json``, opaque inline-asm strings, the ``markers.py`` motif table) plus not-yet-migrated
files during the de-regex sweep. Two mechanisms, checked in order:

  1. an inline ``# regex-ok: <rationale>`` comment on the offending line (preferred — co-located);
  2. a whole-file entry in ``build_tools/scripts/regex_allowlist.txt`` (for files that are entirely
     a pattern table, e.g. ``markers.py``).

The allowlist only ever *shrinks*: as each file is converted, delete its entry. Run::

    python build_tools/scripts/check_no_regex.py            # full scan (exit 1 on violation)
    python build_tools/scripts/check_no_regex.py --staged   # only git-staged files
    python build_tools/scripts/check_no_regex.py --stop-hook # emit Claude Code Stop-hook JSON
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALLOW_FILE = ROOT / "build_tools" / "scripts" / "regex_allowlist.txt"
SCAN_ROOTS = ("merlin/python/merlin", "merlin/contract", "build_tools/scripts")
# Path fragments that mark BUILD-GENERATED (gitignored) trees, not source — never scanned. `_data`
# is the read-only data bundle setup.py copies into the package at wheel-build time.
EXCLUDE_FRAGMENTS = ("/_data/",)
INLINE_MARKER = "# regex-ok:"

# re-module functions that constitute a regex call site.
REGEX_FUNCS = frozenset({
    "compile", "match", "search", "fullmatch", "sub", "subn",
    "findall", "finditer", "split", "escape",
})


def _load_allowlist() -> set[str]:
    """Whole-file exemptions (repo-relative paths); ``#`` comments and blank lines ignored.
    An entry may carry a trailing ``# rationale`` — everything after the first ``#`` is dropped."""
    allow: set[str] = set()
    if ALLOW_FILE.is_file():
        for line in ALLOW_FILE.read_text(encoding="utf-8").splitlines():
            path = line.split("#", 1)[0].strip()
            if path:
                allow.add(path)
    return allow


class _RegexVisitor(ast.NodeVisitor):
    """Collect line numbers of ``re``-module call sites, following the file's import aliases."""

    def __init__(self) -> None:
        self.aliases: set[str] = set()       # module aliases bound to `re` (e.g. {"re", "_re"})
        self.from_funcs: set[str] = set()     # names bound via `from re import <name>`
        self.hits: list[tuple[int, str]] = []  # (lineno, what)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "re":
                self.aliases.add(alias.asname or "re")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "re":
            for alias in node.names:
                self.from_funcs.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        fn = node.func
        # <alias>.<func>(...)
        if (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)
                and fn.value.id in self.aliases and fn.attr in REGEX_FUNCS):
            self.hits.append((fn.lineno, f"{fn.value.id}.{fn.attr}"))
        # bare <func>(...) from `from re import <func>`
        elif isinstance(fn, ast.Name) and fn.id in self.from_funcs and fn.id in REGEX_FUNCS:
            self.hits.append((fn.lineno, fn.id))
        self.generic_visit(node)


def _scan_file(path: Path) -> list[tuple[int, str]]:
    """Regex call sites in ``path`` not silenced by an inline ``# regex-ok:`` marker."""
    src = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return []
    v = _RegexVisitor()
    v.visit(tree)
    if not v.hits:
        return []
    lines = src.splitlines()
    out = []
    for lineno, what in sorted(set(v.hits)):
        line = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
        if INLINE_MARKER not in line:
            out.append((lineno, what))
    return out


def _iter_targets(staged: bool) -> list[Path]:
    if staged:
        # FAIL CLOSED on an unreadable index. This gate's entire work list comes from `git`, so a `git`
        # that cannot run (bad GIT_DIR, no repo, no binary) yielded an EMPTY list and the gate printed
        # OK -- a green that could not have gone red. `check=True` turns that into an exception the
        # caller reports; see check_no_answer_keys.py, which fixed the same shape first.
        out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout
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


#: This gate's name in its own messages.
_GATE = "no-regex"

def _unexaminable(stop_hook: bool, exc: BaseException) -> int:
    """Refuse when the work list could not be read.

    "We could not look" is not "there is nothing to find". A `git` failure used to yield an empty
    work list and a printed OK, so an unreadable tree was indistinguishable from a clean one.
    Reported in whichever dialect the caller speaks (a Stop hook BLOCKS via JSON on stdout, not via
    the exit status), so the two cannot drift apart.
    """
    reason = (f"{_GATE}: could not list the files to examine ({exc}); NOTHING was examined, which is "
              f"not the same as clean. Fix the tree/index and re-run.")
    if stop_hook:
        print(json.dumps({"decision": "block", "reason": reason}))
        return 0  # stop-hook signals via JSON, not exit code
    print(f"[FAIL] {reason}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    staged = "--staged" in argv
    stop_hook = "--stop-hook" in argv
    allow = _load_allowlist()

    violations: list[str] = []
    try:
        targets = _iter_targets(staged)
    except (OSError, subprocess.CalledProcessError) as exc:
        return _unexaminable(stop_hook, exc)
    for rel in targets:
        relstr = rel.as_posix()
        if relstr in allow:
            continue
        for lineno, what in _scan_file(ROOT / rel):
            violations.append(f"{relstr}:{lineno}: regex call `{what}` "
                              f"(replace with a structured impl, add `# regex-ok: <why>`, "
                              f"or allowlist the file)")

    if stop_hook:
        if violations:
            print(json.dumps({"decision": "block",
                              "reason": ("Stray regex outside the allowlist (see docs / "
                                         "build_tools/scripts/regex_allowlist.txt):\n- "
                                         + "\n- ".join(violations))}))
        else:
            print(json.dumps({}))
        return 0  # stop-hook signals via JSON, not exit code

    if violations:
        print(f"[FAIL] no-regex: {len(violations)} regex call site(s) outside the allowlist:")
        for v in violations:
            print(f"  - {v}")
        return 1
    print(f"[  ok] no-regex: {len(allow)} allowlisted file(s); no stray regex in scope.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
