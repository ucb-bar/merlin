#!/usr/bin/env python3
"""Freeze a run's submission: hash the artifact, write freeze.json + a run_manifest.yaml skeleton.

Called between the public grading phase and the hidden phase, so the hidden set is graded against an
immutable, hashed artifact (you cannot tune on hidden).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

import _common as C


def freeze(run_dir: Path) -> dict:
    sub = run_dir / "submission"
    h = C.hash_tree(sub)
    rec = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "submission_sha256": h["sha256"],
        "submission_files": h["n_files"],
        "repo_sha": C.repo_sha(),
    }
    (run_dir / "freeze.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args(argv)
    rec = freeze(Path(a.run_dir))
    print(f"froze {a.run_dir}: sha={ (rec['submission_sha256'] or 'none')[:16] } "
          f"files={rec['submission_files']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
