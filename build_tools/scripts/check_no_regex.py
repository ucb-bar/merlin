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

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _source_layout  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from merlin.common.regex_scan import scan_source  # noqa: E402

ALLOW_FILE = ROOT / "build_tools" / "scripts" / "regex_allowlist.txt"
SCAN_ROOTS = (*_source_layout.SOURCE_SCAN_ROOTS, "merlin/contract", "build_tools/scripts")
# Path fragments that mark BUILD-GENERATED (gitignored) trees, not source — never scanned. `_data`
# is the read-only data bundle setup.py copies into the package at wheel-build time.
EXCLUDE_FRAGMENTS = ("/_data/",)
INLINE_MARKER = "# regex-ok:"


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


def _scan_file(path: Path) -> list[tuple[int, str]]:
    """Regex call sites in ``path`` not silenced by an inline ``# regex-ok:`` marker."""
    src = path.read_text(encoding="utf-8", errors="replace")
    try:
        hits = scan_source(src, filename=str(path))
    except SyntaxError:
        return []
    if not hits:
        return []
    lines = src.splitlines()
    out = []
    for lineno, what in hits:
        line = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
        if INLINE_MARKER not in line:
            out.append((lineno, what))
    return out


def _iter_targets(staged: bool) -> list[Path]:
    return _source_layout.scan_python_paths(ROOT, SCAN_ROOTS, staged=staged)


#: This gate's name in its own messages.
_GATE = "no-regex"


def _unexaminable(stop_hook: bool, exc: BaseException) -> int:
    """Refuse when the work list could not be read.

    "We could not look" is not "there is nothing to find". A `git` failure used to yield an empty
    work list and a printed OK, so an unreadable tree was indistinguishable from a clean one.
    Reported in whichever dialect the caller speaks (a Stop hook BLOCKS via JSON on stdout, not via
    the exit status), so the two cannot drift apart.
    """
    reason = (
        f"{_GATE}: could not list the files to examine ({exc}); NOTHING was examined, which is "
        f"not the same as clean. Fix the tree/index and re-run."
    )
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
        if _source_layout.policy_path(relstr) in allow or relstr in allow:
            continue
        for lineno, what in _scan_file(ROOT / rel):
            violations.append(
                f"{relstr}:{lineno}: regex call `{what}` "
                f"(replace with a structured impl, add `# regex-ok: <why>`, "
                f"or allowlist the file)"
            )

    if stop_hook:
        if violations:
            print(
                json.dumps(
                    {
                        "decision": "block",
                        "reason": (
                            "Stray regex outside the allowlist (see docs / "
                            "build_tools/scripts/regex_allowlist.txt):\n- " + "\n- ".join(violations)
                        ),
                    }
                )
            )
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
