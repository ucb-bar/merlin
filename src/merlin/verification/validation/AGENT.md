# src/merlin/verification/validation

## Purpose

Reusable structural/artifact validation predicates shared by the `build_tools/checks/`
check scripts and the `targetgen inspect` command.

## What belongs here

- Pure-Python checks that a generated target repo or the repo scaffold has its required
  files, schemas, and AGENT.md coverage.
- Readable diagnostic helpers (return lists of problem strings; never `sys.exit`).

## What does not belong here

- Synthesis or generation logic (that is `merlin.development.targetgen`).
- CLI argument parsing (the thin scripts under `build_tools/checks/` own that).
- Anything that mutates the filesystem.

## Interfaces

- Loaders, plan checks, repository checks and report rendering live together in
  `merlin.development.targetgen.generation.generated_target`; schema validation
  delegates to `merlin.common.record_formats`.
- `check_generated_target(repo)` is the entry point used by
  `build_tools/checks/check_generated_target.py`.

## Invariants

- Validation is read-only and side-effect free.
- Checks return diagnostics; exit codes are the caller's concern.

## Qualification

- A generated repository must pass the actual structural and schema checks.
  Source review alone does not qualify generated output or hardware execution.

## Notes for future agents

- Keep `REQUIRED_TARGET_PATHS` in sync with what `merlin.development.targetgen.generation.target_repo`
  actually emits.
