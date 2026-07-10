# AGENT.md — artifacts/kernel-index

## Purpose

kernel-index/-extract/-audit outputs (kernel-record indexes, feature tables, policies) scanned from source frameworks.

## Invariants

- Contents are gitignored; only AGENT.md / README.md / .gitkeep are tracked.
- Created via `merlin.common.artifacts` (start_run / new_product / cache_dir), never hand-built paths.
- Axis: **source framework (xnnpack/openblas/exo/triton)**.
