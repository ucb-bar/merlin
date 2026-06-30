# AGENT.md — merlin/tests/ir

## Purpose

Tests for the **ir** subsystem: xDSL dialects, lowering/passes, dispatch, frontends, llvmlower.

## Invariants

- Every test file is `merlin/tests/ir/test_<area>.py`; pytest collects recursively (`testpaths = merlin/tests`).
- Resolve repo paths via `merlin.common.paths.repo_root()` / `merlin_dir()`, never `__file__` parents.
- Place a new test in the subsystem folder it exercises (see CLAUDE.md "Test layout").
