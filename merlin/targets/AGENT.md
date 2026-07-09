# AGENT.md — merlin/targets

## Purpose
Hand-authored **reference target DEFINITIONS** — the small, curated contracts/docs/examples that
describe a target to TargetGen + the lowering pipeline. `toy_npu` is the generic canonical example;
`gemmini` and `saturn` are reference instances. Reference instances are fine; keep target-specifics
out of general machinery.

## Canonical per-target shape (same for every target — predictable for agents)
- `contracts/` **(required)** — `target_contract.yaml`, `dialect_plan.yaml` (validate against
  `merlin/schemas/`). This is the target *definition*.
- `generated/` **(required)** — the per-target scratch output dir; **gitignored** except
  `.gitkeep`/`AGENT.md` (real codegen products go to `artifacts/targets/<target>/`, not here).
- `docs/` *(when there's content)* — architecture/isa/runtime reference notes.
- `examples/` *(when there's content)* — small `.mlir` inputs.
- `contracts/rtl_facts/facts.json` *(RTL-grounded targets, e.g. gemmini)* — the **promoted pin** of a
  `circt_introspect` run (the run is the source of truth; the pin is the offline/CI fallback).

Absence of an optional dir means "no such content", not disorder — we do not keep empty stub dirs.
The **shared** interface-dialect spec is NOT per-target: `merlin_iface.irdl.mlir` lives in
`merlin/contract/` (next to `interface_grammar.md`), never under a target.

## What does NOT belong here
- **Generated codegen packages** (schedules/dialects/OOT builds) → `artifacts/targets/<target>/`.
- **Generated RTL scratch** (`*.hw.mlir`, `*.ll`, arcilator bins) → the PURGEABLE cache
  `artifacts/cache/rtl_introspect/<target>/` (via `merlin.targetgen.rtl.facts.rtl_cache_dir`) —
  **never** inside a target. Only the promoted `facts.json` + small derived headers are tracked.
- Serious production targets → external repos / MLIR plugins.

## Used by
`merlin.xdsl_dialects.targets.{toynpu,saturn}`, `merlin.xdsl_dialects.lowering`, `merlin.targetgen`.
RTL checks resolve facts via `merlin.targetgen.rtl.facts.rtl_facts_path(target)` (default gemmini pin;
override with `$MERLIN_RTL_FACTS` / `--facts` / `explicit=`). Muon has no curated dir here — the
resolver routes it to `artifacts/targets/muon/`.

## Invariants
Only curated reference definitions in-tree; generated products live under `artifacts/targets/`.
See `docs/guides/adding_a_target.md`. Every subdirectory has an AGENT.md.
