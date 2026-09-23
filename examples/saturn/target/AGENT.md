# AGENT.md — examples/saturn/target

## Purpose

Retained reference metadata for the Saturn RVV vector unit. Target-specific backend
and dialect implementation live in the RVV companion's separate `saturn-support/`
provider, selected explicitly through `MERLIN_TARGET_PATH`. This reference tree
does not authorize executable plugin loading; no implementation should be added here.

## What belongs here

- Curated `contracts/` (target_contract.yaml, dialect_plan.yaml) and `docs/`.
- Pointers to external toolchains via env vars (`MERLIN_CHIPYARD`, `MERLIN_SATURN_SIMV`) — never vendored checkouts.

## What does not belong here

- RTL, chipyard sources, or simulator binaries.
- A target-owned runtime: saturn adapts to the Merlin runtime (command buffer + metrics + baremetal harness).

## Interfaces

- The selected OOT provider implements the dialect and owns its execution plan.
  `contracts/dialect_plan.yaml` here is retained reference metadata.
- Execution is selected through the OOT provider; shared runtime transports do not
  supply Saturn-specific compilation or prove hardware behavior.

## Invariants

- Op/type names here describe the reference dialect; executing a selected provider
  consumes that provider's own plan, never a fallback to this tree.
- The vector parameters describe the **spike model**; confirm against a concrete Saturn RTL config before making RTL-level claims (see contract `notes`).
- Residency on this target is a working-set budget, not dedicated storage — keep `resident_storage_bytes` conservative.

## Testing expectations

Exercise the selected OOT provider and its toolchain before claiming executable
support. Pure reference-metadata tests alone are insufficient.

## Notes for future agents

Saturn benchmarks in `$MERLIN_CHIPYARD/generators/saturn/benchmarks` are the reference for flags (`-march=rv64gcv_zfh_zvfh`, `spike -p4`).
