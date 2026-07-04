# Design-space exploration (Workstream 3, part B)

Compare variants for a candidate feature using a **measurable** cost model.

## Variants

```
baseline           pack/load W every use
software_visible   compiler emits resident_pack / resident_matmul / evict
hardware_managed   ordinary matmul; model assumes HW detects/reuses W
oracle             perfect residency and schedule
```

## Cost-model parameters (measurable, not vague knobs)

`dispatch_fixed_cycles`, `pack_startup_cycles`, `pack_bytes_per_cycle`, `dram_bytes_per_cycle`,
`resident_store_bytes`, `accumulator_entries`.

## Outputs

`dse_result` and, across a parameter sweep, `exploitability_report` — how much of the oracle
benefit a compiler can actually capture.

## Modules / tools

`merlin/python/merlin/dse/`; `tools/dse/`, `tools/exploitability/`
(write to `artifacts/dse/<workload>/`).

## Must not

Build a fake DSE with vague low/medium/high knobs. Use the measurable parameters above.
