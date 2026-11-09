# AGENT.md — tests/targetgen

## Purpose

Tests for the **targetgen** subsystem: TargetGen synthesis + contract validation.

## Invariants

- Every test file is `tests/targetgen/test_<area>.py`; pytest collects recursively (`testpaths = tests`).
- Resolve repo paths via `merlin.common.paths.repo_root()` for checkout fixtures and resource-specific resolvers for installed inputs, never `__file__` parents.
- Place a new test in the subsystem folder it exercises (see CLAUDE.md "Test layout").
