# AGENT.md — merlin/tests/runtime

## Purpose

Tests for the **runtime** subsystem: runtime backends (spike/zephyr/xnnpack/openblas/saturn) + engine.

## Invariants

- Every test file is `merlin/tests/runtime/test_<area>.py`; pytest collects recursively (`testpaths = merlin/tests`).
- Resolve repo paths via `merlin.common.paths.repo_root()` / `merlin_dir()`, never `__file__` parents.
- Place a new test in the subsystem folder it exercises (see CLAUDE.md "Test layout").
