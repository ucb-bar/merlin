# AGENT.md — merlin/python/merlin/validation

## Purpose

Reusable structural/artifact validation predicates shared by the `build_tools/scripts/`
check scripts and the `targetgen inspect` command.

## What belongs here

- Pure-Python checks that a generated target repo or the repo scaffold has its required
  files, schemas, and AGENT.md coverage.
- Readable diagnostic helpers (return lists of problem strings; never `sys.exit`).

## What does not belong here

- Synthesis or generation logic (that is `merlin.targetgen`).
- CLI argument parsing (the thin scripts under `build_tools/scripts/` own that).
- Anything that mutates the filesystem.

## Interfaces

- Consumes the five plan artifacts via `merlin.validation.load`/`validate` and the shared
  schemas via `merlin.common.schemas`.
- `check_generated_target(repo)` is the entry point used by
  `build_tools/scripts/check_generated_target.py`.

## Invariants

- Validation is read-only and side-effect free.
- Checks return diagnostics; exit codes are the caller's concern.

## Testing expectations

- Covered indirectly by `merlin/python/tests/test_targetgen_toy.py` (a freshly generated
  toy_npu repo must pass `check_generated_target`).

## Notes for future agents

- Keep `REQUIRED_TARGET_PATHS` in sync with what `merlin.targetgen.generate.target_repo`
  actually emits.
