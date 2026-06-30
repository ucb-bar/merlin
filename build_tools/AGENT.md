# AGENT.md — build_tools

## Purpose

Build & developer tooling for merlin: `scripts/` (build/sweep/capture orchestration, K1/board
measurement & analysis harnesses, repo linters `check_structure.py` / `check_artifact_layout.py` /
`gen_cli_docs.py`), `cmake/`, `docker/`, toolchain setup.

## What belongs here

- Tracked Python/shell automation, measurement/analysis runners, and repo linters.

## What does not belong here

- Generated build *output* (that is gitignored under `build/`).
- Application/library source (lives under `merlin/`) or schemas (`merlin/schemas/`).

## Invariants

- Scripts/helpers here **ARE tracked in git** (committed automation, not generated output).
- Generated build artifacts go under `build/` (gitignored), never here.
