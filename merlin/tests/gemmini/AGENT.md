# AGENT.md — merlin/tests/gemmini

## Purpose

Tests for the **gemmini** subsystem: Gemmini target: conformance/cert, RTL checks, OOT runner, bench contract.

## Invariants

- Every test file is `merlin/tests/gemmini/test_<area>.py`; pytest collects recursively (`testpaths = merlin/tests`).
- Resolve repo paths via `merlin.common.paths.repo_root()` / `merlin_dir()`, never `__file__` parents.
- Place a new test in the subsystem folder it exercises (see CLAUDE.md "Test layout").
