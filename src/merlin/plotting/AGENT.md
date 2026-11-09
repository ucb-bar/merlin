# AGENT.md — merlin/python/merlin/plotting

## Purpose

Figure-generation runners + the shared house plotting style (`merlin_plotstyle`). Run as
`python -m merlin.plotting.<name>`. Style is imported via `merlin.plotting.merlin_plotstyle`,
never re-derived. These read from `artifacts/...` and write figures under `artifacts/` (presentation/
ceiling/plots) via the artifact convention.

## Invariants

- Source lives here (tracked); generated figures go to `artifacts/` (gitignored), never committed.
- Resolve the repo via `merlin.common.paths.repo_root()`, never `__file__` parents.
