# AGENT.md — merlin/targets/saturn

## Purpose

In-tree **reference target** for the Saturn RVV vector unit, modeled as a multicore RV64GCV CPU executed on spike (and replayable on a pre-built Saturn VCS sim). This is the reference target for Merlin's real-ISA runtime backend; it stays in-tree for the same reason toy_npu does.

## What belongs here

- Curated `contracts/` (target_contract.yaml, dialect_plan.yaml) and `docs/`.
- Pointers to external toolchains via env vars (`MERLIN_CHIPYARD`, `MERLIN_SATURN_SIMV`) — never vendored checkouts.

## What does not belong here

- RTL, chipyard sources, or simulator binaries.
- A target-owned runtime: saturn adapts to the Merlin runtime (command buffer + metrics + baremetal harness).

## Interfaces

- `xdsl_dialects/targets/saturn.py` implements the dialect named here; `lowering/target_lowering.py` consumes `contracts/dialect_plan.yaml`.
- Execution: `merlin/python/merlin/runtime/backends/spike.py` (bare-metal multicore RVV) and `backends/vcs.py` (gated RTL replay).

## Invariants

- Op/type names here must match `xdsl_dialects/targets/saturn.py` and `targetgen/synthesize/dialect_plan.py` exactly.
- The vector parameters describe the **spike model**; confirm against a concrete Saturn RTL config before making RTL-level claims (see contract `notes`).
- Residency on this target is a working-set budget, not dedicated storage — keep `resident_storage_bytes` conservative.

## Testing expectations

`merlin/python/tests/test_rvv_spike.py` (skips without the chipyard toolchain) and the targetgen tests.

## Notes for future agents

Saturn benchmarks in `$MERLIN_CHIPYARD/generators/saturn/benchmarks` are the reference for flags (`-march=rv64gcv_zfh_zvfh`, `spike -p4`).
