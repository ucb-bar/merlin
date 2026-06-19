# Cache vs scratchpad boundary (P20 Tool B)

> Per-operand locality from recovered data movement. Weights (and KV state, where present) are **resident/scratchpad/HAL-object candidates** — reused across the K-loop / decode loop; activations + outputs are **streamed/cache** (within-op). Structural; across-K reuse rests on the configured/assumed K (the loop is unrolled by export).

- 17 resident-candidate operand rows (weight + kv_state) across 10 workloads.
- Full per-region resident capacity by dtype: `capacity_requirement_table.csv` (f32 / bf16 / int8 byte-scaled).
- Per-operand detail + reuse scope: `operand_locality_table.csv`.
