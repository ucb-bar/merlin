# AGENT.md — merlin/python/merlin/xdsl_dialects/targets

## Purpose

Generic xDSL target-dialect factory. Authored dialect plans live in examples or OOT support.

## What belongs here

- Generic xDSL dialect construction from a selected target's `dialect_plan.yaml`.

## What does not belong here

- Real-target dialects (gemmini/radiance/etc.) — TargetGen generates those into external repos.
- Lowering logic (that is `../lowering/`).
- Runtime semantics (that is `merlin/python/merlin/runtime/`).

## Interfaces

`../lowering/target_lowering.py` consumes factory-built dialects, driven by the selected target's lowering table. Names must stay aligned with TargetGen-generated dialects (`targetgen/generate/xdsl.py`).

## Invariants

- A target dialect implements interface abstractions; it never defines its own runtime model.
- Keep these byte-honest with the generated dialect for the same target: same op names, same type names, compatible verifiers.

## Testing expectations

Covered by `merlin/tests/ir/test_xdsl_lowering_e2e.py` and per-dialect tests.

## Notes for future agents

If you change generic op construction, verify the authored plans in `examples/*/target/contracts/` and the TargetGen synthesis path in the same commit.
