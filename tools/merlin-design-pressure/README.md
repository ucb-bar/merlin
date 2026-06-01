# merlin-design-pressure — design-pressure analyzer

Thin CLI entrypoint. **Not implemented yet** — this directory documents intent only.

## What it will do

Compute design-pressure metrics for a workload_region at multiple cut points.

## Backing module

`merlin.python.merlin.design_pressure`

## Intended usage

```bash
merlin-design-pressure --workload merlin/benchmarks/semantic_memory/repeated_rhs_matmul.yaml --out output/dse/repeated_rhs/design_pressure.json
```

## Notes

CLI logic is deliberately absent at this scaffold stage. When implemented, this entrypoint
should stay thin and delegate to the backing Python module. Artifacts are written under
`output/` (gitignored).
