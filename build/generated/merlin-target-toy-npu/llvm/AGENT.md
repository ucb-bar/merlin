# AGENT.md — merlin-target-toy_npu/llvm

## Purpose

LLVM extension plan and (placeholder) out-of-tree TableGen/patches/tests.

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

- Default to out-of-tree. A fork is only justified by the plan's fork_triggers.
