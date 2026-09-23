# AGENT.md — build_tools/scripts

## Purpose

Maintenance/validation scripts (e.g. `check_structure.py`).

## What belongs here

- Tracked source for maintenance, build automation, checks and documentation generators.

## What does not belong here

- Application/library implementations (use `src/merlin/` or the owning optional distribution).
- Generated artifacts, logs, caches or schema definitions.

## Invariants

- These scripts are tracked source, not disposable build output.
- Never commit generated artifacts here.
- Write generated products beneath the configured `out/` root using the shared path helpers.
- Keep checks source-layout-aware across core and optional distributions.
