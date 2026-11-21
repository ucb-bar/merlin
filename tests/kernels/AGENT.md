# AGENT.md — tests/kernels

## Purpose

Tests for the **kernels** subsystem: kernel mining/ceiling/CCA/policy/features + kernel backend.

## Invariants

- Every test file is `tests/kernels/test_<area>.py`; pytest collects recursively (`testpaths = tests`).
- Resolve repo paths via `merlin.common.paths.repo_root()` for checkout fixtures and resource-specific resolvers for installed inputs, never `__file__` parents.
- Place a new test in the subsystem folder it exercises (see CLAUDE.md "Test layout").
