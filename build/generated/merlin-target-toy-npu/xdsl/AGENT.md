# AGENT.md — merlin-target-toy_npu/xdsl

## Purpose

Fast xDSL prototype of the target dialect. Promote to MLIR/C++ once stable.

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

- If xDSL is not installed the prototype still imports as plain Python with TODOs.
