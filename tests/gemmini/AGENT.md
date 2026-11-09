# AGENT.md — tests/gemmini

## Purpose

Tests for the **gemmini** subsystem: Gemmini target: conformance/cert, RTL checks, OOT runner, bench contract.

## Invariants

- Backend-dependent tests require an explicitly selected external support provider.
  No handwritten Gemmini provider is vendored or selected by default. Inside a test
  session `tests/conftest.py` selects nothing. Provide reviewed RTL facts separately
  for facts-dependent tests. Absence is not
  a passing target qualification. Do not run compiler/simulator tests merely to
  validate a source-layout change.
- Every test file is `tests/gemmini/test_<area>.py`; pytest collects recursively (`testpaths = tests`).
- Resolve repo paths via `merlin.common.paths.repo_root()` for checkout fixtures and resource-specific resolvers for installed inputs, never `__file__` parents.
- Place a new test in the subsystem folder it exercises (see CLAUDE.md "Test layout").
