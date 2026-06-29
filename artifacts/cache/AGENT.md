# AGENT.md — artifacts/cache

## Purpose

Large regenerable caches (kernel caches, intermediate compute). PURGEABLE.

## Invariants

- Contents are gitignored; only AGENT.md / README.md / .gitkeep are tracked.
- Created via `merlin.common.artifacts` (start_run / new_product / cache_dir), never hand-built paths.
- Axis: **namespace**.
