#!/usr/bin/env python3
"""Linter: enforce the three-root artifact convention (runs/ artifacts/ build/).

Fails (exit 1) if any of:
  * a TRACKED file lives under an old/forbidden root (output/, results/, selfcheck_out/,
    mined_knowledge/, docs/presentation/, merlin/experiments/*/runs/);
  * a TRACKED file has a generated extension (.png/.svg/.pdf/.zip/.jsonl) outside artifacts/
    (and not allowlisted);
  * a versioned product dir artifacts/<topic>/v*/<leaf>/ is missing manifest.yaml;
  * a `latest` symlink under artifacts/ is absolute or dangling.

Usage:
  check_artifact_layout.py             # full working tree (tracked files)
  check_artifact_layout.py --staged    # only staged files (pre-commit)
  check_artifact_layout.py --stop-hook  # emit Claude Code Stop-hook JSON instead of plain text
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

GENERATED_EXTS = {".png", ".svg", ".pdf", ".zip", ".jsonl"}
FORBIDDEN_ROOTS = ("output/", "results/", "selfcheck_out/", "mined_knowledge/", "docs/presentation/")
FORBIDDEN_RE = re.compile(r"(^|/)experiments/[^/]+/runs/")
# genuinely-tracked source images, if any (verify with `git ls-files '*.png'` before adding).
ALLOW_TRACKED_GEN: set[str] = set()
# Self-contained, import-isolated eval project with its own runs/ lifecycle (aet-style) — out of
# the main tree's three-root scope. It evaluates merlin as an external subject; lives off root but
# in-repo so its R4 check can `git diff -- merlin/` against this checkout.
SKIP_PREFIXES = ("merlin/experiments/targetgen_evals/",)


def _repo_root() -> Path:
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                         capture_output=True, text=True).stdout.strip()
    return Path(out) if out else Path.cwd()


def _tracked(root: Path, staged: bool) -> list[str]:
    if staged:
        cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"]
    else:
        cmd = ["git", "ls-files"]
    out = subprocess.run(cmd, cwd=root, capture_output=True, text=True).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def check(root: Path, staged: bool) -> list[str]:
    violations: list[str] = []
    for rel in _tracked(root, staged):
        if any(rel.startswith(s) for s in SKIP_PREFIXES):
            continue
        p = Path(rel)
        if any(rel.startswith(r) for r in FORBIDDEN_ROOTS) or FORBIDDEN_RE.search(rel):
            violations.append(f"tracked file under a forbidden root: {rel}")
            continue
        if (p.suffix.lower() in GENERATED_EXTS and not rel.startswith("artifacts/")
                and rel not in ALLOW_TRACKED_GEN):
            violations.append(f"generated file '{rel}' tracked outside artifacts/")
    art = root / "artifacts"
    if art.is_dir():
        for v in art.glob("*/v*"):
            if not v.is_dir():
                continue
            latest = v / "latest"
            if latest.is_symlink():
                tgt = os.readlink(latest)
                if os.path.isabs(tgt):
                    violations.append(f"absolute `latest` symlink (bwrap-unsafe): {latest.relative_to(root)}")
                elif not (v / tgt).exists():
                    violations.append(f"dangling `latest` symlink: {latest.relative_to(root)}")
            for leaf in v.iterdir():
                if leaf.is_dir() and not leaf.name.startswith(".") and leaf.name != "latest":
                    if not (leaf / "manifest.yaml").exists():
                        violations.append(f"product dir missing manifest.yaml: {leaf.relative_to(root)}")
    return violations


def main(argv: list[str]) -> int:
    staged = "--staged" in argv
    stop_hook = "--stop-hook" in argv
    root = _repo_root()
    violations = check(root, staged)
    if stop_hook:
        if violations:
            print(json.dumps({"decision": "block",
                              "reason": "Artifact-layout violations:\n- " + "\n- ".join(violations)}))
        else:
            print(json.dumps({}))
        return 0  # stop-hook signals via JSON, not exit code
    if violations:
        sys.stderr.write("Artifact-layout check FAILED:\n")
        for v in violations:
            sys.stderr.write(f"  - {v}\n")
        sys.stderr.write("\nSee CLAUDE.md 'Generated-output convention' / .claude/skills/artifact-layout.\n")
        return 1
    print("artifact-layout: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
