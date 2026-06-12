# AGENT.md — merlin-target-saturn

## Purpose

Generated Merlin target package for `saturn` (dialect + runtime adapter).

## What belongs here

- Target dialect (xDSL prototype + MLIR/C++ scaffold), runtime adapter, Zephyr module, LLVM extension plan, examples, tests, and the five contracts.

## What does not belong here

- Merlin core dialects or the runtime ABI (those live in the Merlin repo).
- An independent runtime model — this target only adapts the Merlin runtime.

## Interfaces

- Consumes the Merlin runtime ABI, command-buffer schema, and metrics schema.
- contracts/*.yaml validate against merlin/schemas/*.schema.yaml.

## Invariants

- Targets implement adapters; they never invent runtime models.
- Every directory contains an AGENT.md.

## Testing expectations

- `python build_tools/scripts/check_generated_target.py <this repo>` must pass.

## Notes for future agents

- Regenerate with `python -m merlin.targetgen.cli build --target-name saturn ...`. Hand-edits after review are expected.
