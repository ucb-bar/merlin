# AGENT.md — artifacts/design-pressure

## Purpose

merlin-design-pressure outputs (pressure vector + candidate contracts) per workload.

## Invariants

- Contents are gitignored; only AGENT.md / README.md / .gitkeep are tracked.
- Created via `merlin.common.artifacts` (start_run / new_product / cache_dir), never hand-built paths.
- Axis: **workload/region**.
