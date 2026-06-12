# AGENT.md — merlin/targets/saturn/contracts

## Purpose

Curated saturn plan artifacts: `target_contract.yaml`, `dialect_plan.yaml`.

## What belongs here

- The five plan YAMLs (contract + dialect plan now; runtime_adapter/zephyr/llvm plans as they are curated).

## What does not belong here

- Generated artifacts (TargetGen writes generated repos under `build/`).

## Interfaces

Validated against `merlin/schemas/*.schema.yaml`; consumed by `xdsl_dialects/lowering/` (lowering table) and `targetgen/synthesize/` (kept in sync).

## Invariants

- Op/type names must match `xdsl_dialects/targets/saturn.py` and `targetgen/synthesize/dialect_plan.py`.
- Vector parameters describe the spike model — see the contract `notes`.

## Testing expectations

`test_rvv_spike.py` loads the dialect plan; targetgen tests validate schemas.
