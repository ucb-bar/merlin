#!/usr/bin/env python3
"""Lint docs + AGENT.md for references to RETIRED repo paths (living-docs anti-drift).

Not a general path-exists checker (that would false-positive on globs, illustrative
paths, and not-yet-generated ``artifacts/``/``runs/`` outputs). Instead a precise
**deny-list** of paths that were deleted or deprecated in the repo reorg — a doc that
still points at one is stale and will mislead agents.

Retired tokens (see CLAUDE.md "Generated-output convention" + docs/design/):
  - ``merlin/compiler/``       -> future C++ plane, not built (docs/design/compiler_plane.md)
  - ``merlin/integrations/``   -> adapters live in-package (docs/design/integrations.md)
  - ``generated_targets/``     -> folded into artifacts/targets/
  - ``mined_knowledge/``       -> folded into artifacts/kernel-mining/
  - ``output/<sub>/`` write targets -> deprecated; use artifacts/ or runs/

A line is EXEMPT if it documents the retirement itself (contains one of the
allow-words below) — that's how the design notes + repo_structure.md legitimately
name these paths. Add genuine one-offs to build_tools/scripts/doc_paths_allow.txt
(``<relpath>:<substring>`` per line).

Usage:
  check_doc_paths.py           # list violations
  check_doc_paths.py --check   # exit 1 on any violation (wired into check_structure)
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALLOW_FILE = ROOT / "build_tools" / "scripts" / "doc_paths_allow.txt"

# (regex, human message). Kept deliberately small + specific.
RETIRED = [
    (re.compile(r"merlin/compiler\b"), "retired tree merlin/compiler/ (see docs/design/compiler_plane.md)"),
    (re.compile(r"merlin/integrations\b"), "retired tree merlin/integrations/ (see docs/design/integrations.md)"),
    (re.compile(r"\bgenerated_targets/"), "retired generated_targets/ -> artifacts/targets/"),
    (re.compile(r"\bmined_knowledge/"), "retired mined_knowledge/ -> artifacts/kernel-mining/"),
    (re.compile(r"(?<![\w./])output/\w"), "deprecated output/ write target -> artifacts/ or runs/"),
]

# A line naming a retired path only to say it's retired is fine.
ALLOW_WORDS = ("deprecated", "retired", "gitignored", "recaptures", "regenerable",
               "not built", "not yet built", "folded into", "replaces", "no longer",
               "former", "removed to keep", "no standalone",
               # a line that points at the design note is contextualizing, not misdirecting
               "docs/design/")

# Trees we don't scan (generated / vendored / design notes that legitimately name retired paths).
SKIP_PARTS = {".git", ".venv", "venv", "build", "artifacts", "runs", "output",
              "third_party", "tmp", "__pycache__", "node_modules"}


def _doc_files() -> list[Path]:
    out: list[Path] = []
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_PARTS]
        for f in files:
            if f.endswith(".md") and (f == "AGENT.md" or Path(base) == ROOT / "docs"
                                      or (ROOT / "docs") in Path(base).parents):
                out.append(Path(base) / f)
    return sorted(out)


def _load_allow() -> set[str]:
    if not ALLOW_FILE.is_file():
        return set()
    return {ln.strip() for ln in ALLOW_FILE.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")}


def _is_design_note(rel: str) -> bool:
    # docs/design/* exist to describe the retired trees — never flag them.
    return rel.startswith("docs/design/")


def scan() -> list[str]:
    allow = _load_allow()
    problems: list[str] = []
    for p in _doc_files():
        rel = p.relative_to(ROOT).as_posix()
        if _is_design_note(rel):
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            low = line.lower()
            if any(w in low for w in ALLOW_WORDS):
                continue
            for rx, msg in RETIRED:
                m = rx.search(line)
                if not m:
                    continue
                if any(a == f"{rel}:{m.group(0)}" or (a.startswith(rel + ":") and a.split(":", 1)[1] in line)
                       for a in allow):
                    continue
                problems.append(f"{rel}:{i}: {msg}")
    return problems


def main(argv: list[str]) -> int:
    problems = scan()
    if problems:
        sys.stderr.write("doc-path check FAILED (retired paths referenced):\n")
        for pr in problems:
            sys.stderr.write(f"  - {pr}\n")
        if "--check" in argv:
            return 1
        return 1
    print("doc paths: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
