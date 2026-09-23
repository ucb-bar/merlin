# AGENT.md — examples/saturn/target/contracts

## Purpose

Curated saturn plan artifacts: `target_contract.yaml`, `dialect_plan.yaml`.

## What belongs here

- The five plan YAMLs (contract + dialect plan now; runtime_adapter/zephyr/llvm plans as they are curated).

## What does not belong here

- Generated artifacts (TargetGen writes generated repos under `build/`).

## Interfaces

Validated against `merlin/schemas/*.schema.yaml`; consumed by `xdsl_dialects/lowering/` (lowering table) and `targetgen/synthesize/` (kept in sync).

## Invariants

- Op/type names describe the retained reference dialect. Executable dialect support
  and its plan are owned by the explicitly selected OOT provider; no plugin
  declarations or executable implementation belong in this reference tree.
- Vector parameters describe the spike model — see the contract `notes`.

## Testing expectations

`test_rvv_spike.py` loads the dialect plan; targetgen tests validate schemas.
