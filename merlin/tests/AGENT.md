# AGENT.md — merlin/tests

## Purpose

The cross-subsystem pytest suite for merlin: unit + integration across kernels, MLIR/xDSL
compilation, DSE, runtime/backends, model bringup, validation. This is the root project's
`testpaths` entry; optional distributions also have focused `packages/*/tests/` suites.
Shared test fixtures/data live in `fixtures/` and `data/` here.

## What belongs here

- `test_*.py` (the suite), shared `fixtures/` and `data/` consumed by tests.

## What does not belong here

- Library/application source (`src/merlin/` or `packages/*/src/`).
- Generated outputs (those belong under the configured `out/` root).

## Invariants

- Resolve repo paths via `merlin.common.paths.repo_root()` / `merlin_dir()`, never `__file__` parents
  (so tests are location-independent).
- Run: `.venv/bin/python -m pytest merlin/tests`.
