#!/usr/bin/env python3
"""Rebuild <base>..<tip> as an ordered series of subsystem commits, using only git plumbing.

The working branch carries thousands of commits made by many concurrent sessions: honest history, and
unreviewable as a pull request. Rewriting it would invalidate every worktree and branch built on it.
This writes a SEPARATE branch instead, whose commits each carry one subsystem's whole change with a
message describing it, and whose final tree is asserted byte-identical to the tip's -- a different
path to the same content, not a different content.

Plumbing only: every commit is built in a private index file from the tip's exact (mode, blob) entries,
so symlinks, executable bits and submodule gitlinks carry over exactly, no working tree or shared index
is touched, and no hook runs. The output ref must not already exist; it is created with a
compare-and-swap against the all-zero sha, so an existing branch is never overwritten.

Usage:
  build_publish_branch.py <base> <tip> <plan.json>                    # dry run: print the series
  build_publish_branch.py <base> <tip> <plan.json> refs/heads/<name>  # create the branch

The plan groups paths by prefix, in commit order, each with a subject and a body. A changed path that no
group claims is an error rather than a silent omission.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def git(*args, env=None, input=None) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True, env=env, input=input).stdout


def group_of(path: str, groups: list[dict]) -> str:
    for g in groups:
        if any(path == p or path.startswith(p.rstrip("/") + "/") for p in g["paths"]):
            return g["id"]
    return "other"


def main(base: str, tip: str, plan_file: str, out_ref: str | None) -> int:
    plan = json.loads(Path(plan_file).read_text())
    groups = plan["groups"]
    diff = git("diff", "--name-status", "--no-renames", "-z", base, tip)
    parts = diff.split("\0")
    changes: list[tuple[str, str]] = []
    for i in range(0, len(parts) - 1, 2):
        changes.append((parts[i], parts[i + 1]))
    by_group: dict[str, list[tuple[str, str]]] = {}
    for status, path in changes:
        by_group.setdefault(group_of(path, groups), []).append((status, path))
    if "other" in by_group:
        print("UNGROUPED PATHS:", [p for _, p in by_group["other"]][:20], file=sys.stderr)
        return 2

    tip_entries = {}
    for line in git("ls-tree", "-r", "-z", "--full-tree", tip).split("\0"):
        if not line:
            continue
        meta, path = line.split("\t", 1)
        mode, _typ, sha = meta.split()
        tip_entries[path] = (mode, sha)

    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(td) / "index"))
        git("read-tree", base, env=env)
        parent = git("rev-parse", base).strip()
        made = []
        for g in groups:
            items = by_group.get(g["id"], [])
            if not items:
                continue
            batch = []
            for status, path in items:
                if status == "D":
                    batch.append(f"0 {'0' * 40}\t{path}")
                else:
                    mode, sha = tip_entries[path]
                    batch.append(f"{mode} {sha}\t{path}")
            git("update-index", "--index-info", env=env, input="\n".join(batch) + "\n")
            tree = git("write-tree", env=env).strip()
            msg = g["subject"] + "\n\n" + g["body"].strip() + "\n"
            parent = git("commit-tree", tree, "-p", parent, "-F", "-", input=msg).strip()
            made.append((parent[:10], len(items), g["subject"]))
        final_tree = git("rev-parse", f"{parent}^{{tree}}").strip()
        tip_tree = git("rev-parse", f"{tip}^{{tree}}").strip()
        for sha, n, subj in made:
            print(f"{sha}  {n:5d} paths  {subj}")
        print("final tree == tip tree:", final_tree == tip_tree)
        if final_tree != tip_tree:
            return 1
        if out_ref:
            git("update-ref", out_ref, parent, "0" * 40)  # CAS against "absent": never overwrite
            print("wrote", out_ref, "->", parent[:10])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None))
