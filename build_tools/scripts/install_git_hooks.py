#!/usr/bin/env python3
"""Enable the repo-managed git hooks (docs + artifact-layout anti-drift) for THIS clone.

Git hooks are not tracked and cannot be auto-installed by a commit. This points
`core.hooksPath` at the committed `build_tools/git-hooks/` directory, so every commit runs the
repo-managed hooks: `pre-commit` (check_artifact_layout.py --staged + check_no_regex + check_docs.py)
and `commit-msg` (enforces the commit-message convention — see CLAUDE.md).

Per-clone / opt-in — it edits this checkout's .git/config only. Undo with:
  git config --unset core.hooksPath

Usage:
  python build_tools/scripts/install_git_hooks.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOOKS_DIR = ROOT / "build_tools" / "git-hooks"


def main() -> int:
    hooks = ["pre-commit", "commit-msg"]
    missing = [h for h in hooks if not (HOOKS_DIR / h).is_file()]
    if missing:
        sys.stderr.write(f"missing hook(s): {', '.join(str(HOOKS_DIR/h) for h in missing)}\n")
        return 1
    for h in hooks:
        os.chmod(HOOKS_DIR / h, 0o755)
    rel = HOOKS_DIR.relative_to(ROOT).as_posix()
    subprocess.run(["git", "-C", str(ROOT), "config", "core.hooksPath", rel], check=True)
    print(f"core.hooksPath -> {rel} ({', '.join(hooks)} enabled for this clone)")
    print("undo with: git config --unset core.hooksPath")
    return 0


if __name__ == "__main__":
    sys.exit(main())
