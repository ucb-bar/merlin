# Repository conventions

This file documents the conventions this repo enforces — the generated-output layout, the test
layout, and the documentation model. Read it (and the directory-local `AGENT.md` files) before making
changes; the same gates run in pre-commit and CI.

> Working-tree note: this repo is frequently developed by multiple coding agents against a single
> working tree. Commit on the currently checked-out branch and avoid switching branches mid-task; use
> a separate `git worktree` if you need isolation.

# Generated-output convention — one root (`out/`), three subdirs

All generated/produced output lives under a **single top-level `out/` root**, with exactly three
subdirs (`out/runs/`, `out/artifacts/`, `out/build/`). Nothing generated goes anywhere else (the old
top-level `runs/`/`artifacts/`/`build/`, plus `output/`, `results/`, `selfcheck_out/`,
`docs/presentation/`, and per-experiment `runs/`/`reports/` locations are retired and gitignored).

Root names come from `merlin.common.paths` — `out_dir()` / `runs_dir()` / `artifacts_dir()` /
`build_dir()` (honoring `MERLIN_OUT_ROOT`). Never hard-code the literal strings; call the helpers.

- **`out/runs/<target>/<suite>/<run-id>/`** — aet-managed experiment runs. Create via
  `merlin.common.artifacts.start_run(..., target=...)` (never hand-build a run path); it passes
  `project_root=out_dir()` so aet lays runs under `out/runs/`. Query with `aet runs` — point it at
  the `out/` root (it reads `<project_root>/runs`).
- **`out/artifacts/`** — every other generated product, organized **concern-first** (each tool/concern
  owns a subtree and uses ITS OWN axis — target for compiler/mining/experiments, workload for the
  three DSE tools, model for recaptures/measurements, framework for kernel-index, cross-cutting for
  ceiling/compare). Concerns: `dse-guidance/`, `dse/`, `design-pressure/`, `kernel-mining/<target>/`,
  `kernel-index/<framework>/`, `ceiling/`, `compare/`,
  `measurements/<substrate>/<model>/<exp>_v<ver>_<TS>_<sha>/` (substrate = `k1_spacemit` /
  `firesim_<bitstream>` / `baremetal_<verilator-design>` / `zephyr_<design>` / `spike_<config>`,
  via `new_measurement(...)`), `recaptures/`, `perf-bench/<target>/`, `capsule-bench/<target>/`,
  `targets/<target>/`, `presentation/`, `cache/`, `selfcheck/`.

  **`out/artifacts/targets/<target>/<package_id>/`** is the codegen-package home (schedules/knobs/dialects
  minted by `merlin-rvv-mine` / `merlin-rvv-autotune` / `merlin-targetgen`). It **replaces the retired
  top-level `generated_targets/`** — all references point at `out/artifacts/targets/` directly (no compat
  symlink; paths are repo-root-relative). Packages are **tool-generated**: only the hand-authored
  reference baselines + promoted champions are tracked (`rvv/hand_v0`, `rvv/hand_v0_int8`,
  `rvv/impr_tuned_*`, via `.gitignore` negations); forks and `mlir_oot/build/` trees stay ignored.
  Full buildable OOT repos live under `out/build/generated/`, not here.

  **`output/` is DEPRECATED and retired** — model recaptures now live at `out/artifacts/recaptures/`
  (via `recaptures_dir()`); never write new generated content to `output/` (the guard hook blocks it).
  - versioned products at `out/artifacts/<concern>/<axis>/v<ver>/<concern>_<axis>_v<ver>_<TS>_<sha7>/`
    (+ `manifest.yaml`, relative `latest` symlink) via `new_product(..., target=...)`;
  - regenerable caches under `out/artifacts/cache/<ns>/` via `cache_dir(...)` (PURGEABLE);
  - the 130 GB model recaptures under `out/artifacts/recaptures/` (PURGEABLE) via `recaptures_dir()`.
- **`out/build/`** — compiled / CMake output, baseline toolchains, and buildable OOT codegen repos
  (`out/build/generated/`).

**Naming ("sortable + provenance")**: timestamp token `YYYYMMDDTHHMMSSZ` (UTC, no `:`); runs are
timestamp-first (`<TS>_<method>_seed<NNN>_<sha7>`), products are topic-first
(`<topic>_<target>_v<ver>_<TS>_<sha7>`). The folder name is convenience; `run_record.json`/`manifest.yaml`
(git_sha, timestamp, version) is the source of truth.

**Target at folder level**: pass `target=` so it becomes a folder component
(`out/runs/<target>/<suite>/...`, `out/artifacts/<topic>/<target>/...`) and everything for a target
groups together. Keep inner file names identical across targets (e.g. `perf_results.json`, `findings.csv`,
`manifest.yaml`) so target-vs-target diffs are trivial.

**Enforcement** (do not bypass without cause): a PreToolUse hook
(`.claude/hooks/guard_artifact_writes.py`) blocks generated writes outside the `out/` root;
`build_tools/scripts/check_artifact_layout.py` lints tracked-file violations (pre-commit / Stop hook).
Helper API and examples: `.claude/skills/artifact-layout/SKILL.md` and `merlin/python/merlin/common/artifacts.py`.
Escape hatch for a genuine one-off: `export MERLIN_ALLOW_ARTIFACT_WRITE=1` or add a prefix to
`.claude/hooks/artifact_allowlist.txt`.

# Test layout — one suite, organized by subsystem

All tests live in **`merlin/tests/`** (the sole pytest `testpaths`), organized into **subsystem
buckets**: `kernels/ rvv/ dse/ gemmini/ targetgen/ ir/ runtime/ infra/`. Rules (enforced by
`build_tools/scripts/check_structure.py` "test layout"; see `.claude/skills/test-layout`):

- A test file is `merlin/tests/<bucket>/test_<area>.py` — **never at the `merlin/tests/` root**, and
  `<bucket>` must be one of the eight above. Place a new test in the subsystem it exercises.
- Shared inputs live in `merlin/tests/fixtures/` and `merlin/tests/data/`.
- Resolve repo paths via `merlin.common.paths.repo_root()` / `merlin_dir()` — **never** `Path(__file__).parents[N]`
  (so tests are location-independent and survive moves).
- Run the suite: `.venv/bin/python -m pytest merlin/tests`.

# Documentation convention — durable docs vs reports

Start at the generated hub **`docs/README.md`**. Durable documentation lives in **`docs/`** under
three kind-subdirs — **`reference/`** (code-derived facts), **`guides/`** (how-tos), **`design/`**
(rationale) — each file carrying YAML front-matter (`title, kind, status, owner, last_verified,
related, code_refs`). **Point-in-time reports** (results/findings/status/presentations) are NOT docs
— they live under `artifacts/` (concern-first; see "Generated-output convention"). Rules (see
`.claude/skills/docs-layout`; enforced by `check_structure.py` + the `check_docs.py` Stop/pre-commit/CI gates):

- Generated docs are regenerated, never hand-edited: `gen_cli_docs.py` (`reference/cli.md`),
  `gen_package_docs.py` (`reference/module_index.md`), `gen_schema_docs.py` (`reference/schemas.md`),
  `gen_docs_index.py` (`README.md` hub). Run them after touching CLIs/schemas/packages/docs.
- Front-matter must be schema-valid; `check_doc_paths.py` blocks retired paths; the root README/AGENT.md
  may not carry scaffold-era phrasing.
- **Semantic drift** (a doc whose `code_refs` moved past its `last_verified`) is surfaced by
  `check_docs_freshness.py --json` and fixed by the **`docs-doctor`** skill (reconcile, then bump the date).
- Enable the pre-commit gate per clone: `python build_tools/scripts/install_git_hooks.py`.
