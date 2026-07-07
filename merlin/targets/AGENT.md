# AGENT.md — merlin/targets

## Purpose
Hand-authored **reference target DEFINITIONS** — the small, curated contracts/docs/examples that
describe a target to TargetGen + the lowering pipeline. `toy_npu` is the generic canonical example;
`gemmini` and `saturn` are reference instances. Reference instances are fine; keep target-specifics
out of general machinery.

## What lives here (per target: toy_npu / gemmini / saturn)
- `contracts/` — `target_contract.yaml`, `dialect_plan.yaml` (validate against `merlin/schemas/`).
- `docs/` — architecture/isa/runtime reference notes. `examples/` — small `.mlir` inputs.
- `contracts/rtl_facts/facts.json` (gemmini) — curated RTL-derived fact table read by the RTL checks.

## What does NOT belong here
- **Generated codegen packages** (schedules/dialects/OOT builds) → `artifacts/targets/<target>/`.
- Generated RTL scratch (`rtl_facts/*.hw.mlir`, `*.ll`, arcilator bins) — **gitignored**; only the
  curated `facts.json` + small derived headers are tracked. `generated/` holds only `.gitkeep`/AGENT.md.
- Serious production targets → external repos / MLIR plugins.

## Used by
`merlin.xdsl_dialects.targets.{toynpu,saturn}`, `merlin.xdsl_dialects.lowering`,
`merlin.targetgen` (+ RTL checks read gemmini `facts.json`; override via `MERLIN_RTL_FACTS`).

## Invariants
Only curated reference definitions in-tree; generated products live under `artifacts/targets/`.
See `docs/guides/adding_a_target.md`. Every subdirectory has an AGENT.md.
