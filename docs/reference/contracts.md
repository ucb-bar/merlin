---
title: Contracts
kind: reference
status: current
owner: ir
last_verified: 2026-07-14
related: [core_dialects]
code_refs: [merlin/python/merlin/xdsl_dialects, merlin/schemas]
---

# Contracts

A **contract** captures what is true about a workload, what the compiler must prove, and what the
hardware/runtime promises. Contracts are the legality/evidence layer.

- Workload facts: shapes, dtypes, reuse counts, mutability, lifetimes.
- Target capabilities: supported ops, layouts, resident storage budget, accumulators.
- Compiler obligations: what must be proven before a feature is legal.
- Hardware/runtime promises: residency, accumulator commit, persistent handles.

Schema: `merlin/schemas/target_contract.schema.yaml` (target side) and
`merlin/schemas/workload_region.schema.yaml` (workload side). The eventual IR home is the
`contract` dialect (see `docs/dialects.md`); prototype it in
`merlin/python/merlin/xdsl_dialects/contract.py`.

Example (ToyNPU): `merlin/targets/toy_npu/contracts/target_contract.yaml`.
