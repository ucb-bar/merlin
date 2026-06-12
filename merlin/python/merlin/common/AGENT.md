# AGENT.md — merlin/python/merlin/common

## Purpose

Shared utilities: schema loading/validation, IO, common types.

## What belongs here

- Files appropriate to the purpose above.

## What does not belong here

- Workstream-specific logic (this is shared infrastructure only).
- Generated artifacts (write those to `output/`).

## Invariants

- Keep this directory focused on its stated purpose.
- Every subdirectory must also contain an AGENT.md.
- Shared helpers (schema load/validate, yaml, llm summary) are real and dependency-light.
