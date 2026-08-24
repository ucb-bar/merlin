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


#: Filenames that live under a ``golden/`` dir but are handed to the graded method BY ITS OWN PROMPT
#: (``targetgen_evals/methods/*/prompt.md`` tells it to read both "for reference"), so they are
#: stimulus, not answers -- and keeping them tracked is what makes the benchmark reproducible from a
#: clone. Matched by NAME under any dataset, so a second target's dataset behaves identically; the
#: right long-term fix is to move declared inputs out of ``golden/`` so the directory name means one
#: thing. This set may only SHRINK. Anything else in a ``golden/`` dir stays an answer key.
_DECLARED_METHOD_INPUTS = frozenset({"expected_contract.yaml", "expected_dialect_features.yaml"})
#: The dataset layout those inputs belong to -- every path component must be present, so the
#: exemption cannot leak to a ``golden/`` dir somewhere else in the tree.
_EVAL_DATASET_PARTS = ("targetgen_evals", "datasets", "golden")


def _is_answer_key(path: str) -> bool:
    """True iff ``path`` is a benchmark answer surface that must not be tracked."""
    parts = path.split("/")
    name = parts[-1]
    if name in _DECLARED_METHOD_INPUTS and all(seg in parts for seg in _EVAL_DATASET_PARTS):
        return False
    if "hidden" in parts:                       # the entire held-out subtree, any depth
        return True
    if "golden" in parts:                       # a golden/ DIRECTORY is an answer surface too -- the
        return True                             # filename rules below miss `golden/expected_*.yaml`
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
