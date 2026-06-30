# AGENT.md — merlin/tests/rvv

## Purpose

Tests for the **rvv** subsystem: RVV codegen (rvvgen) + RVV/K1 board bringup + model-on-RVV.

## Invariants

- Every test file is `merlin/tests/rvv/test_<area>.py`; pytest collects recursively (`testpaths = merlin/tests`).
- Resolve repo paths via `merlin.common.paths.repo_root()` / `merlin_dir()`, never `__file__` parents.
- Place a new test in the subsystem folder it exercises (see CLAUDE.md "Test layout").
