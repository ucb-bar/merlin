# AGENT.md — tests/dse

## Purpose

Tests for the **dse** subsystem: DSE tools (dse / dse_guidance / design_pressure), cost model, search, compare.

## Invariants

- Every test file is `tests/dse/test_<area>.py`; pytest collects recursively (`testpaths = tests`).
- Resolve repo paths via `merlin.common.paths.repo_root()` for checkout fixtures and resource-specific resolvers for installed inputs, never `__file__` parents.
- Place a new test in the subsystem folder it exercises (see CLAUDE.md "Test layout").
