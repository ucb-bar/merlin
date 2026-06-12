# compare — parallel strategy comparison

Thin CLI entrypoint. **Not implemented yet.**

## What it will do

Run a fixed set of compilation strategies over a set of workloads (the workload x strategy
matrix), in parallel, and emit a scoreboard + decision report. Reuses `dse_result` /
`exploitability_report`.

## Backing module

`merlin.dse.harness` (+ `merlin.dse.strategy`, `merlin.pipelines.builder`)

## Intended usage

```bash
compare \
  --workloads merlin/benchmarks/semantic_memory/*.yaml \
  --strategies output/dse/strategies/*.yaml \
  --out output/dse/interface_dse/
```
