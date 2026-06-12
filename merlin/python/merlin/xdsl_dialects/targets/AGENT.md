# AGENT.md — merlin/python/merlin/xdsl_dialects/targets

## Purpose

In-tree **reference target dialects** (xDSL) the core lowering pipeline lowers into. Today: `toynpu`.

## What belongs here

- Small xDSL dialects for in-tree reference targets only (`toynpu.py`; later a curated `saturn.py` reference).
- Op/type names must match the target's `dialect_plan.yaml` exactly.

## What does not belong here

- Real-target dialects (gemmini/radiance/etc.) — TargetGen generates those into external repos.
- Lowering logic (that is `../lowering/`).
- Runtime semantics (that is `merlin/python/merlin/runtime/`).

## Interfaces

`../lowering/target_lowering.py` consumes these dialects, driven by the lowering table in `merlin/targets/<t>/contracts/dialect_plan.yaml`. Names must stay aligned with the TargetGen-generated dialect for the same target (`targetgen/generate/xdsl.py`).

## Invariants

- A target dialect implements interface abstractions; it never defines its own runtime model.
- Keep these byte-honest with the generated dialect for the same target: same op names, same type names, compatible verifiers.

## Testing expectations

Covered by `merlin/python/tests/test_xdsl_lowering_e2e.py` (lowering through `toynpu`) and the per-dialect tests.

## Notes for future agents

If you change op names here, change `merlin/targets/toy_npu/contracts/dialect_plan.yaml` and `targetgen/synthesize/dialect_plan.py` in the same commit.
