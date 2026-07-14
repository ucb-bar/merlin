# AGENT.md — out

## Purpose

The **single top-level root for all generated/produced output** (see CLAUDE.md,
"Generated-output convention"). Nothing generated goes anywhere else in the repo.

## What belongs here

Exactly three subdirs — never add a fourth:

- `out/runs/` — aet-managed experiment runs (`<target>/<suite>/<run-id>/`). Created via
  `merlin.common.artifacts.start_run(..., target=...)`; queried with `aet runs`.
- `out/artifacts/` — every other generated product, organized concern-first (`dse-guidance/`,
  `dse/`, `kernel-mining/<target>/`, `measurements/`, `recaptures/`, `targets/<target>/`, `cache/`,
  ...). Versioned products via `new_product(...)`; purgeable caches via `cache_dir(...)`.
- `out/build/` — compiled / CMake output, baseline toolchains, and buildable OOT codegen repos
  (`out/build/generated/`).

Root names come from `merlin.common.paths` — `out_dir()` / `runs_dir()` / `artifacts_dir()` /
`build_dir()` (honoring `MERLIN_OUT_ROOT`). Never hard-code the literal strings; call the helpers.

## What does not belong here

- Hand-authored source, schemas, or curated input corpora (those live under `merlin/`, `docs/`, etc.).
- A new top-level subdir beside `runs/`/`artifacts/`/`build/`.

## Invariants

- Contents are gitignored; only `AGENT.md` / `README.md` / `.gitkeep` skeletons and the explicit
  curated negations in `.gitignore` are tracked. Regenerable output is never committed.
- A PreToolUse hook (`.claude/hooks/guard_artifact_writes.py`) blocks generated writes outside
  this root; `build_tools/scripts/check_artifact_layout.py` lints tracked-file violations.
