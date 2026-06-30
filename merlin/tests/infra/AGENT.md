# AGENT.md — merlin/tests/infra

## Purpose

Tests for the **infra** subsystem: repo conventions: artifact layout, smoke/CLI smoke.

## Invariants

- Every test file is `merlin/tests/infra/test_<area>.py`; pytest collects recursively (`testpaths = merlin/tests`).
- Resolve repo paths via `merlin.common.paths.repo_root()` / `merlin_dir()`, never `__file__` parents.
- Place a new test in the subsystem folder it exercises (see CLAUDE.md "Test layout").
