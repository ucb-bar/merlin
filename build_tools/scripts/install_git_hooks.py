#!/usr/bin/env python3
"""Enable the repo-managed git hooks (docs + artifact-layout anti-drift) for THIS clone.

Git hooks are not tracked and cannot be auto-installed by a commit. This points
`core.hooksPath` at the committed `build_tools/git-hooks/` directory, so every commit runs
`build_tools/git-hooks/pre-commit` (which calls check_artifact_layout.py --staged + check_docs.py).

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
    if not (HOOKS_DIR / "pre-commit").is_file():
        sys.stderr.write(f"missing {HOOKS_DIR/'pre-commit'}\n")
        return 1
    os.chmod(HOOKS_DIR / "pre-commit", 0o755)
    rel = HOOKS_DIR.relative_to(ROOT).as_posix()
    subprocess.run(["git", "-C", str(ROOT), "config", "core.hooksPath", rel], check=True)
    print(f"core.hooksPath -> {rel} (pre-commit enabled for this clone)")
    print("undo with: git config --unset core.hooksPath")
    return 0


if __name__ == "__main__":
    sys.exit(main())
