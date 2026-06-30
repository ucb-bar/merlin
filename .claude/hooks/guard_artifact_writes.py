#!/usr/bin/env python3
"""PreToolUse guard: deny writing generated artifacts outside the three sanctioned roots.

Repo convention (see CLAUDE.md "Generated-output convention" + .claude/skills/artifact-layout):
generated output lives ONLY under runs/ (aet runs), artifacts/ (products/caches/plots/...),
or build/. This hook blocks Write/Edit/MultiEdit/NotebookEdit that would drop a generated-looking
file into an old/forbidden location (output/, results/, selfcheck_out/, mined_knowledge/,
docs/presentation/, *_dse_analysis, *_recap) or write a generated extension outside artifacts/.

Source edits (merlin/, build_tools/, experiments/*/scripts/, tests/, *.md docs, etc.) are always allowed.
Escape hatch: env MERLIN_ALLOW_ARTIFACT_WRITE=1, or list a path prefix in
.claude/hooks/artifact_allowlist.txt. Contract: exit 0 = allow; exit 2 + stderr message = deny.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SANCTIONED_ROOTS = ("runs/", "artifacts/", "build/", "tmp/")
SKELETON_NAMES = {"AGENT.md", "README.md", ".gitkeep"}
GENERATED_EXTS = {".png", ".svg", ".pdf", ".zip", ".jsonl"}
FORBIDDEN_SUBSTR = (
    "output/", "results/", "selfcheck_out/", "mined_knowledge/", "/presentation/",
    "_dse_analysis", "_recap",
)


def _repo_root() -> Path:
    env = os.environ.get("MERLIN_REPO_ROOT")
    if env:
        return Path(env)
    # hook lives at <repo>/.claude/hooks/guard_artifact_writes.py
    return Path(__file__).resolve().parents[2]


def _allowlisted(rel: str, root: Path) -> bool:
    f = root / ".claude" / "hooks" / "artifact_allowlist.txt"
    if not f.exists():
        return False
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and rel.startswith(line):
            return True
    return False


def _target_path(data: dict) -> str | None:
    ti = data.get("tool_input") or {}
    return ti.get("file_path") or ti.get("notebook_path") or ti.get("path")


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0  # never block on malformed input
    if data.get("tool_name") not in {"Write", "Edit", "MultiEdit", "NotebookEdit"}:
        return 0
    raw = _target_path(data)
    if not raw:
        return 0
    if os.environ.get("MERLIN_ALLOW_ARTIFACT_WRITE") == "1":
        return 0

    root = _repo_root()
    try:
        abs_p = Path(raw)
        if not abs_p.is_absolute():
            abs_p = Path(data.get("cwd") or root) / abs_p
        rel = os.path.relpath(abs_p.resolve(), root)
    except Exception:
        return 0
    if rel.startswith(".."):
        return 0  # outside the repo (e.g. /tmp scratchpad) — not our concern

    rel_posix = Path(rel).as_posix()
    name = Path(rel_posix).name
    slashed = f"/{rel_posix}"

    # 1) skeleton docs allowed anywhere
    if name in SKELETON_NAMES:
        return 0
    # 2) sanctioned roots allowed (artifacts/presentation/ beats the /presentation/ deny)
    if any(rel_posix.startswith(r) for r in SANCTIONED_ROOTS):
        return 0
    if "/_qa_ws/" in slashed:
        return 0
    # 3) explicit allowlist escape hatch
    if _allowlisted(rel_posix, root):
        return 0

    ext = Path(rel_posix).suffix.lower()
    if ext in GENERATED_EXTS or any(s in slashed for s in FORBIDDEN_SUBSTR):
        sys.stderr.write(
            "BLOCKED by guard_artifact_writes: "
            f"'{rel_posix}' is a generated artifact outside the sanctioned roots "
            "(runs/, artifacts/, build/).\n"
            "Write generated output via merlin.common.artifacts: start_run() -> runs/<suite>/...,\n"
            "new_product()/cache_dir() -> artifacts/<topic>/...  (see .claude/skills/artifact-layout).\n"
            "Escape hatch: export MERLIN_ALLOW_ARTIFACT_WRITE=1 or add a prefix to "
            ".claude/hooks/artifact_allowlist.txt.\n"
        )
        return 2

    # 4) everything else (source edits) allowed
    return 0


if __name__ == "__main__":
    sys.exit(main())
