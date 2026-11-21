# AGENT.md — tests/rvv

## Purpose

Tests for the **rvv** subsystem: RVV codegen (rvvgen) + RVV/K1 board bringup + model-on-RVV.

## Invariants

- Every test file is `tests/rvv/test_<area>.py`; pytest collects recursively (`testpaths = tests`).
- Resolve repo paths via `merlin.common.paths.repo_root()` for checkout fixtures and resource-specific resolvers for installed inputs, never `__file__` parents.
- Place a new test in the subsystem folder it exercises (see CLAUDE.md "Test layout").
