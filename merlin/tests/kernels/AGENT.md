# AGENT.md — merlin/tests/kernels

## Purpose

Tests for the **kernels** subsystem: kernel mining/ceiling/CCA/policy/features + kernel backend.

## Invariants

- Every test file is `merlin/tests/kernels/test_<area>.py`; pytest collects recursively (`testpaths = merlin/tests`).
- Resolve repo paths via `merlin.common.paths.repo_root()` / `merlin_dir()`, never `__file__` parents.
- Place a new test in the subsystem folder it exercises (see CLAUDE.md "Test layout").
