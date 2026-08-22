#!/usr/bin/env python3
"""Fail if any benchmark ANSWER KEY is tracked in git.

This repo is public (github.com/ucb-bar/merlin). The graded answers — golden outputs, required
instruction-coverage, the held-out HIDDEN capsule sets, and the holdout SPECIFICATION sidecars
(``profiles/<target>.hidden.yaml`` — a holdout's op, dtype and exact shape is itself an answer, and
the tracked profile sits inside the ``merlin/contract/`` grant every arm receives) — must never be
committed: publishing them
defeats the benchmark and lets an agent cheat. They are regenerable from the tracked generators + the
oracle; the public CONTRACT (``capsule.interface.mlir`` + ``capsule.yaml`` + ``MANIFEST.yaml``, outside
``hidden/``) stays tracked.

Structured path matching only (no regex, per the repo's derive-don't-hardcode/no-regex conventions).
Run as a pre-commit + Stop gate. Exit non-zero listing any tracked answer key.
"""
from __future__ import annotations

import subprocess
import sys


def _is_answer_key(path: str) -> bool:
    """True iff ``path`` is a benchmark answer surface that must not be tracked."""
    parts = path.split("/")
    name = parts[-1]
    if "hidden" in parts:                       # the entire held-out subtree, any depth
        return True
    if name.endswith(".hidden.yaml"):           # the holdout SPECIFICATION sidecar (op+dtype+shape)
        return True
    if name == "golden.yaml":                   # graded golden output
        return True
    if name.startswith("golden_") and name.endswith(".yaml"):  # dtype-variant goldens (golden_w8a8.yaml …)
        return True
    if name == "expected_instruction_coverage.yaml":           # required instruction classes = the answer
        return True
    return False


def main() -> int:
    try:
        tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                                 check=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"check_no_answer_keys: could not list tracked files: {e}", file=sys.stderr)
        return 0  # not a git tree / git unavailable — nothing to gate
    leaks = [p for p in tracked if _is_answer_key(p)]
    if leaks:
        print(f"[FAIL] no-answer-keys: {len(leaks)} benchmark answer key(s) are TRACKED "
              f"(public repo — untrack with `git rm --cached`, they stay on disk + are gitignored):",
              file=sys.stderr)
        for p in leaks[:40]:
            print(f"    {p}", file=sys.stderr)
        if len(leaks) > 40:
            print(f"    … and {len(leaks) - 40} more", file=sys.stderr)
        return 1
    print("[  ok] no-answer-keys: no golden/expected/hidden answer surfaces are tracked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
