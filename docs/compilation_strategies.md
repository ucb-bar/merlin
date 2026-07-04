# Parallel compilation strategies

A **compilation strategy** is a first-class, hashable object describing ONE way of compiling a
workload (schema: `merlin/schemas/compilation_strategy.schema.yaml`). It generalizes the
`baseline / software_visible / hardware_managed / oracle` variant enum.

Because xDSL pass pipelines are assembled from a string of named passes, a strategy's
`lowering_pipeline` field *is* the compilation approach. Two strategies that differ only in
pipeline or exposed interface features are two comparable approaches you can run side by side.

## Substrate

```
merlin/python/merlin/pipelines/   named xDSL passes + build_pipeline(spec)
merlin/python/merlin/dse/strategy.py   Strategy + registry (loads compilation_strategy YAML)
merlin/python/merlin/dse/harness.py    parallel runner over (workload x strategy) matrix
```

## Flow

```
workloads (benchmarks/*)  x  strategies (registry)
   -> per-cell xDSL pipeline -> simulator/cost model -> dse_result.json   (parallel, independent)
   -> collect -> scoreboard.csv + decision_report.md   (artifacts/dse/<workload>/)
```

Runs are keyed by strategy id (hash) so they are reproducible and cacheable. The `search/` layer
sits on top to *generate* the strategy set rather than hand-listing it.

## Deciding what to keep

| Signal | Keep / drop |
| ------ | ----------- |
| Exploitability = software_visible / oracle | high -> keep & expose; low -> push to hardware_managed or drop |
| Win margin vs baseline | small margin -> not worth interface complexity |
| Cross-workload stability | robust wins graduate; narrow wins stay experimental |
| Cost of exposure (area/complexity proxy) | Pareto-filter; keep non-dominated |

**Promotion gate:** an approach stays an xDSL prototype while being compared; it graduates to a
stable MLIR/C++ plane (**not yet built** — see `docs/design/compiler_plane.md`) only when it
consistently survives the scoreboard.
