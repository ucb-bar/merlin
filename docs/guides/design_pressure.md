---
title: Design pressure
kind: guide
status: current
owner: design_pressure
last_verified: 2026-07-22
related: [getting_started, dse, dse_guidance]
code_refs: [merlin/python/merlin/design_pressure]
---

# Design pressure (Workstream 3, part A)

## Prerequisites

**Shared base only.** Complete the base install in [Getting started](getting_started.md)
(`uv sync --all-extras`); this workflow runs on the committed `workload_region.yaml` inputs and needs
**no** external toolchain, simulator, or board.

Pipeline:

```
workload_region.yaml
  -> multi-cutpoint analysis
  -> design_pressure.json
  -> candidate_contracts.yaml
```

Compute measurable pressures at several compiler cut points and recommend candidate features.

## Cut points

graph, linalg, loop, bufferized, dispatch, trace
(`merlin/python/merlin/design_pressure/cutpoints/`).

## Metrics

shape/dtype distribution, reuse count, mutability, lifetime intervals, pack/unpack count, layout
conversions, intermediate write bytes, dispatch count, work per dispatch
(`merlin/python/merlin/design_pressure/metrics/`).

## Expected recommendations (controlled examples)

- `repeated_rhs_matmul` -> `resident_packed_tensor`
- `matmul_bias_requant_relu` -> `accumulator_commit`
- `no_reuse_matmul` -> none

## Tool

the `merlin-design-pressure` CLI (writes to `out/artifacts/design-pressure/<workload>/`).
