#!/usr/bin/env python3
"""Freeze a run's submission identity: hash the artifact and write freeze.json.

Called between the public grading phase and the hidden phase, so the hidden set is graded against an
immutable, hashed artifact (you cannot tune on hidden).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from merlin.benchharness import hash_tree, repo_sha


def freeze(run_dir: Path, *, repo: Path) -> dict:
    sub = run_dir / "submission"
    h = hash_tree(sub)
    rec = {
        "frozen_at": datetime.now(UTC).isoformat(),
        "submission_sha256": h["sha256"],
        "submission_files": h["n_files"],
        "repo_sha": repo_sha(repo=repo),
    }
    from merlin_experiments.phase1 import oot_history

    history = oot_history.freeze(run_dir, sub)
    if history is not None:
        rec["oot"] = history  # the harness-owned `frozen` tag on exactly these bytes
    (run_dir / "freeze.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    if history is not None and history.get("frozen_commit"):
        from merlin.targetgen import target_index

        target_index.refresh_for_run(run_dir, 1)  # the index lists the new frozen compiler
    return rec


def main(argv: list[str] | None = None, *, repo=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--repo", type=Path, help="repository/work root for source identity")
    a = ap.parse_args(argv)
    selected_repo = a.repo if a.repo is not None else (repo() if callable(repo) else repo)
    if selected_repo is None:
        ap.error("installed freezing requires explicit --repo")
    rec = freeze(Path(a.run_dir), repo=Path(selected_repo))
    print(f"froze {a.run_dir}: sha={(rec['submission_sha256'] or 'none')[:16]} files={rec['submission_files']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
