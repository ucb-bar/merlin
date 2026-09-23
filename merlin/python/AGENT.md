# AGENT.md — merlin/python

## Purpose

Compatibility import root. The `merlin` symlink points to canonical `src/merlin`.
Do not add another implementation here; new source belongs under `src/` or `packages/`.

## What belongs here

- Files appropriate to the purpose above.

## What does not belong here

- Generated artifacts (write those to `runs/` or `artifacts/`).

## Invariants

- Keep this directory focused on its stated purpose.
- Every subdirectory must also contain an AGENT.md.
