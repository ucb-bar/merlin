# AGENT.md — merlin/targets

## Purpose

In-tree **toy / reference targets** used for TargetGen development and conformance testing. `toy_npu` is the canonical reference target.

## What belongs here

- Toy/reference targets: `toy_npu/`, `example_vector/`.
- Per-target docs, contracts, examples, generated scaffolds, and tests.

## What does not belong here

- Serious production targets — those become external repos / MLIR plugins.
- Large generated artifacts (those go to gitignored `build/`/`output/`).

## Interfaces

Target contracts/plans validate against `merlin/schemas/target_contract.schema.yaml` and `dialect_plan.schema.yaml`. Consumed by `targetgen`.

## Invariants

- Only toy/reference targets belong in-tree.
- Serious targets should become external repos or plugins.

## Testing expectations

Conformance tests under `merlin/tests/conformance/` and per-target `tests/`.

## Notes for future agents

ToyNPU eventually exposes `toynpu.{res_pack,matmul,commit,evict}` and `!toynpu.{resident_tensor,accumulator}`. See `docs/adding_a_target.md`.
