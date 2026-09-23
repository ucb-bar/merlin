# AGENT.md — merlin/tests/gemmini

## Purpose

Tests for the **gemmini** subsystem: Gemmini target: conformance/cert, RTL checks, OOT runner, bench contract.

## Invariants

- Backend-dependent tests require an OOT support provider explicitly selected with
  `MERLIN_TARGET_PATH`; no in-tree backend/build-support copy remains. Select the
  pinned companion recorded in `build_tools/upstreams/target_support.json` and
  provide reviewed RTL facts separately for facts-dependent tests. Absence is not
  a passing target qualification. Do not run compiler/simulator tests merely to
  validate a source-layout change.
- Every test file is `merlin/tests/gemmini/test_<area>.py`; pytest collects recursively (`testpaths = merlin/tests`).
- Resolve repo paths via `merlin.common.paths.repo_root()` / `merlin_dir()`, never `__file__` parents.
- Place a new test in the subsystem folder it exercises (see CLAUDE.md "Test layout").
