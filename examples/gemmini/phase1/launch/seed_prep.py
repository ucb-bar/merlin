#!/usr/bin/env python3
"""Pre-launch preparation for a seeded open-compiler run (standalone entry point).

    seed_prep.py --seed <frozen compiler dir> --export <run-framework export> --out <dir> \\
        [--run-dir <seed run dir>] [--variants <seed workspace>/variants] [--cpus 4-7 --workers 4] \\
        [--cases A,B,...] [--compare LABEL=TREE[,perf_history.jsonl] ...] [--env-file env.json]

Writes SEED_BASELINE.json/.md, SEED_STRUCTURE.md and, when a variant is promoted,
promoted-compiler/ into --out. See ``merlin_experiments.phase1.seed_prep``. A from-scratch
(unseeded) launch never runs this.
"""

from __future__ import annotations

import sys

from merlin_experiments.phase1.seed_prep.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
