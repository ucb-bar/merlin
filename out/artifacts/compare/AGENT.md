# AGENT.md — artifacts/compare

## Purpose

merlin-compare config x workload comparison campaigns. Timestamped product dirs.

## Invariants

- Contents are gitignored; only AGENT.md / README.md / .gitkeep are tracked.
- Created via `merlin.common.artifacts` (start_run / new_product / cache_dir), never hand-built paths.
- Axis: **config x workload cross-product**.
