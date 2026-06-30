# AGENT.md — merlin/tests

## Purpose

The single pytest suite for merlin (~754 tests): unit + integration across kernels, MLIR/xDSL
compilation, DSE, runtime/backends, model bringup, validation. This is the sole `testpaths` entry
in `pyproject.toml`. Test fixtures/data live in `fixtures/` and `data/` here.

## What belongs here

- `test_*.py` (the suite), shared `fixtures/` and `data/` consumed by tests.

## What does not belong here

- Library/application source (lives under `merlin/python/merlin/`).
- Generated outputs (those are gitignored under `runs/` / `artifacts/`).

## Invariants

- Resolve repo paths via `merlin.common.paths.repo_root()` / `merlin_dir()`, never `__file__` parents
  (so tests are location-independent).
- Run: `.venv/bin/python -m pytest merlin/tests`.
