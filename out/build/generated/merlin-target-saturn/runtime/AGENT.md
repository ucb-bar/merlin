# AGENT.md — merlin-target-saturn/runtime

## Purpose

Target-side runtime ADAPTER (not a runtime). Implements the Merlin runtime ABI: command encoding, simulator semantics, metrics mapping.

## What belongs here

- Files appropriate to the purpose above.

## What does not belong here

- Unrelated code; generated build outputs; vendored external repos.

## Interfaces

- See contracts/ for the plans that drive this directory's generation.

## Invariants

- Keep this directory focused on its stated purpose.
- Every subdirectory must also contain an AGENT.md.

## Testing expectations

- Update tests/ when this directory's contracts change.

## Notes for future agents

- Merlin owns the runtime ABI and command-buffer schema; this directory only ADAPTS them. Never define an independent runtime model here.
