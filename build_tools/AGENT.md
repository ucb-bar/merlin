# AGENT.md — build_tools

## Purpose

Build & developer tooling for merlin: `scripts/` (build/sweep/capture orchestration, K1/board
measurement & analysis harnesses, repo linters `check_structure.py` / `check_artifact_layout.py` /
`gen_cli_docs.py`), `cmake/`, `docker/`, toolchain setup.

## What belongs here

- Tracked Python/shell automation, measurement/analysis runners, and repo linters.

## What does not belong here

- Generated build *output* (that is gitignored under `out/build/`).
- Application/library source (`src/merlin/` or `packages/*/src/`) or schemas (`merlin/schemas/`).

## Invariants

- Scripts/helpers here **ARE tracked in git** (committed automation, not generated output).
- Generated build artifacts go under `out/build/` (gitignored), never here.
