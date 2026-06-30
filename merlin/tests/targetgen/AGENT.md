# AGENT.md — merlin/tests/targetgen

## Purpose

Tests for the **targetgen** subsystem: TargetGen synthesis + contract validation.

## Invariants

- Every test file is `merlin/tests/targetgen/test_<area>.py`; pytest collects recursively (`testpaths = merlin/tests`).
- Resolve repo paths via `merlin.common.paths.repo_root()` / `merlin_dir()`, never `__file__` parents.
- Place a new test in the subsystem folder it exercises (see CLAUDE.md "Test layout").
