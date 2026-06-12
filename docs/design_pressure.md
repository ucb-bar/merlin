# Design pressure (Workstream 3, part A)

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

`tools/design-pressure/` (writes to `output/dse/<workload>/`).
