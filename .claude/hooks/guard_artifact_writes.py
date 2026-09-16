#!/usr/bin/env python3
"""PreToolUse guard: deny writing generated artifacts outside the sanctioned root.

Repo convention (see CLAUDE.md "Generated-output convention" + .claude/skills/artifact-layout):
generated output lives ONLY under a single top-level out/ root, with subdirs out/runs/ (aet runs),
out/artifacts/ (products/caches/plots/...), and out/build/. This hook blocks
Write/Edit/MultiEdit/NotebookEdit that would drop a generated-looking file into an old/forbidden
location (the retired top-level runs/ artifacts/ build/ output/ results/ selfcheck_out/
mined_knowledge/ docs/presentation/ *_dse_analysis *_recap) or write a generated extension outside
out/.

Source edits (merlin/, build_tools/, experiments/*/scripts/, tests/, *.md docs, etc.) are always allowed.
Escape hatch: env MERLIN_ALLOW_ARTIFACT_WRITE=1, or list a path prefix in
.claude/hooks/artifact_allowlist.txt. Contract: exit 0 = allow; exit 2 + stderr message = deny.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SANCTIONED_ROOTS = ("out/", "tmp/")
SKELETON_NAMES = {"AGENT.md", "README.md", ".gitkeep"}
GENERATED_EXTS = {".png", ".svg", ".pdf", ".zip", ".jsonl"}
# Retired top-level generated roots (now consolidated under out/) + legacy forbidden locations.
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


def _declared(root: Path) -> tuple[set, set]:
    """``(out roots, concerns)`` from merlin/contract/storage.yaml.

    Read with a deliberately small parser rather than a YAML library: this hook runs under whatever
    interpreter the harness gives it, and a missing import here would either crash the hook or,
    worse, teach someone to delete the check. Anything it does not recognise yields empty sets, and
    an empty roster disables the rule -- the guard's standing contract is to never block on input it
    cannot read.
    """
    path = root / "merlin" / "contract" / "storage.yaml"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return set(), set()
    found: dict[str, set] = {"out_roots": set(), "concerns": set()}
    section = None
    for line in lines:
        if line[:1] not in (" ", "\t", "#", ""):
            section = line.split(":", 1)[0] if line.rstrip().endswith(":") else None
            continue
        if section not in found or not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith("  ") or line.startswith("   "):
            continue                      # a continuation of the previous entry's value, not a key
        entry = line.strip()
        name = entry[2:] if entry.startswith("- ") else entry.split(":", 1)[0]
        name = name.strip().strip("'\"")
        if name and "/" not in name:
            found[section].add(name)
    return found["out_roots"], found["concerns"]


def _undeclared_destination(rel_posix: str, root: Path) -> str | None:
    """The part of this path the storage contract does not name, or None if it is all declared."""
    parts = rel_posix.split("/")
    if parts[0] != "out" or len(parts) < 3:
        return None                       # tmp/, or a file sitting directly in the out/ root
    roots, concerns = _declared(root)
    if not roots:
        return None                       # no readable roster: the rule is off, not failing closed
    if parts[1] not in roots:
        return f"out/{parts[1]}/, an undeclared top-level root"
    if parts[1] == "artifacts" and len(parts) >= 4 and parts[2] not in concerns:
        return f"the undeclared concern out/artifacts/{parts[2]}/"
    return None


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
        undeclared = _undeclared_destination(rel_posix, root)
        if undeclared:
            sys.stderr.write(
                f"BLOCKED by guard_artifact_writes: '{rel_posix}' writes into {undeclared}, which\n"
                "merlin/contract/storage.yaml does not declare. The out/ root has three roots and a\n"
                "closed set of concerns; the roster drifted to 52 undeclared concerns against 16\n"
                "declared ones because nothing checked it at the moment one was created.\n"
                "Write into a declared concern, or add this one to that file with a line saying what\n"
                "it holds (`merlin-storage layout` prices the drift, `organize` folds a stray one in).\n"
                "Escape hatch: export MERLIN_ALLOW_ARTIFACT_WRITE=1.\n"
            )
            return 2
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
            f"'{rel_posix}' is a generated artifact outside the sanctioned root out/.\n"
            "Write generated output via merlin.common.artifacts: start_run() -> out/runs/<suite>/...,\n"
            "new_product()/cache_dir() -> out/artifacts/<topic>/...  (see .claude/skills/artifact-layout).\n"
            "Escape hatch: export MERLIN_ALLOW_ARTIFACT_WRITE=1 or add a prefix to "
            ".claude/hooks/artifact_allowlist.txt.\n"
        )
        return 2

    # 4) everything else (source edits) allowed
    return 0


if __name__ == "__main__":
    sys.exit(main())
