---
title: Architecture
kind: reference
status: current
owner: core
last_verified: 2026-07-14
related: [repo_structure, core_dialects, lowering_pipeline]
code_refs: [merlin/python/merlin]
---

# Architecture

merlin studies which hardware/software abstractions are worth exposing to the compiler. The
architecture is built around **shared artifacts (schemas)** that flow between three workstreams,
and **two compiler planes** (xDSL for prototyping, MLIR/C++ for stable code).

## Data flow

```
External kernels/repos --> integration adapters --> kernel_record / abstraction_candidate / policy_rule
ISA/docs/RTL           --> targetgen            --> target_contract --> dialect_plan --> dialect scaffold
workload_region        --> design_pressure      --> design_pressure --> candidate_contracts
candidate + cost model  --> dse                  --> dse_result --> exploitability_report
```

## Compiler flow (intended)

```
linalg / tensor / scf
  -> contract     (facts, obligations, legality)
  -> schedule     (chosen compiler decisions)
  -> interface    (target-independent HW/SW abstractions)
  -> <target dialect>    (e.g. toynpu)
  -> runtime      (command buffers, dispatch, waits, profiling)
  -> binary / simulator / external runner
```

## Principles

- Root stays clean; almost all code lives under `merlin/`.
- Coordinate through schemas, not prose.
- Integrations are adapters, never vendored repos.
- Prototype in xDSL; promote to MLIR/C++ only when stable.
- A concept becomes a dialect only when it must survive passes, be verified/transformed, and
  lower. Otherwise it is a schema/YAML artifact first.
