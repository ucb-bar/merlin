# AGENT.md — artifacts/capsule-bench

## Purpose

gemmini_capsule_bench aggregate report figures/tables (runs themselves live under runs/<target>/capsule-bench/).

## Invariants

- Contents are gitignored; only AGENT.md / README.md / .gitkeep are tracked.
- Created via `merlin.common.artifacts` (start_run / new_product / cache_dir), never hand-built paths.
- Axis: **target -> bench**.
