# AGENT.md — artifacts/kernel-mining

## Purpose

rvvgen mining/autotune products (mined policies, CCA, action catalogs, beam/headtohead). Versioned products: <name>_v<ver>_<TS>.

## Invariants

- Contents are gitignored; only AGENT.md / README.md / .gitkeep are tracked.
- Created via `merlin.common.artifacts` (start_run / new_product / cache_dir), never hand-built paths.
- Axis: **target backend (rvv/k1) -> op**.
