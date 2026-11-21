# AGENT.md — tests/infra

## Purpose

Tests for the **infra** subsystem: repo conventions: artifact layout, smoke/CLI smoke.

## Invariants

- Every test file is `tests/infra/test_<area>.py`; pytest collects recursively (`testpaths = tests`).
- Resolve repo paths via `merlin.common.paths.repo_root()` for checkout fixtures and resource-specific resolvers for installed inputs, never `__file__` parents.
- Place a new test in the subsystem folder it exercises (see CLAUDE.md "Test layout").
