# AGENT.md — artifacts/plots

## Purpose

Gitignored: generated figures and tables (was output/kernels/ceiling, output/rvv_bench, experiments/*/reports figures).

## What belongs here

- Files appropriate to the purpose above, written via `merlin.common.artifacts` (never by hand-constructed paths).

## What does not belong here

- Hand-authored source or schemas.
- Anything that should be tracked in git (contents are gitignored).

## Invariants

- Contents are gitignored; only AGENT.md / README.md / .gitkeep are tracked.
- Never commit generated artifacts here.
- Figures are regenerable from the plot_*.py scripts; do not hand-edit.
