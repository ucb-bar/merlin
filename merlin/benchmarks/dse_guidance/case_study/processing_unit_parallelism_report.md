# Processing-unit & hierarchy guidance

> Which processing-unit shapes the workloads imply, and whether the evidence favors one bigger unit, multiple identical units, or specialized units. **Structural search-space guidance — no speedup, cycle, or area claim.**

## Resource pressure (where the work is)

| resource class | present | ops | MAC share | basis |
|---|---|---|---|---|
| dense_gemm | True | 160 | 35.0% | compute_macs |
| skinny_gemm_or_gemv | True | 879 | 63.7% | compute_macs |
| epilogue_or_requant | True | 667 | 19.6% | op_count |
| dma_or_memory | True | 0 | 0.0% | structural |
| control_or_dispatch | True | 0 | 0.0% | structural |
| attention_softmax_or_reduction | False | 0 | 0.0% | unavailable |
| conv_or_patch_embed | False | 0 | 0.0% | unavailable |
| sparse_or_skip | False | 0 | 0.0% | unavailable |

## One bigger unit vs. many identical vs. specialized

- **Average inter-op parallelism is low (1.29×)** (see `dag_parallelism_report.md`): the dependency DAG is near-sequential, so **many identical units would be hard to keep busy** by inter-op concurrency alone — the parallelism to exploit is *intra-op* sharding.
- **Compute splits across two shapes:** dense GEMM is 35% of MACs (pi05, rdt, rdt2, smolvla, xr0) while skinny/GEMV is 64% (bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, small_llama, smolvla, tiny_llama, xr0). This favors **specialized units** (a matrix engine *and* a GEMV/vector engine) over one universal unit.
- **Plus structural units:** an `epilogue_requant_unit` (the addmm bias/activation fuses onto the GEMM), a `dma_engine` (resident loop-invariant weights), and a `loop_controller` (the bounded K-loop) — each backed by recovered structure.
- **Honestly absent:** `attention_kv_engine` and `conv_engine` have no supporting operators in the captures (attention is lowered, no conv) — listed `unavailable`.

## Candidate units

| unit | evidence_for | supporting workloads |
|---|---|---|
| matrix_engine | serves dense_gemm: 35% of MACs across pi05, rdt, rdt2, smolvla, xr0 | pi05, rdt, rdt2, smolvla, xr0 |
| skinny_gemm_or_gemv_engine | serves skinny_gemm_or_gemv: 64% of MACs across bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, small_llama, smolvla, tiny_llama, xr0 | bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, small_llama, smolvla, tiny_llama, xr0 |
| attention_kv_engine | no supporting operators in the captures | — |
| conv_engine | no supporting operators in the captures | — |
| epilogue_requant_unit | serves epilogue_or_requant (op_count; 667 across groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, smolvla) | groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, smolvla |
| dma_engine | serves dma_or_memory (structural; structural across bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, small_llama, smolvla, tiny_llama, xr0) | bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, small_llama, smolvla, tiny_llama, xr0 |
| loop_controller | serves control_or_dispatch (structural; structural across bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, small_llama, smolvla, tiny_llama, xr0) | bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, small_llama, smolvla, tiny_llama, xr0 |
| scalar_control_unit | serves control_or_dispatch (structural; structural across bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, small_llama, smolvla, tiny_llama, xr0) | bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, small_llama, smolvla, tiny_llama, xr0 |

**Caveat (structural, not realized):** resource pressure and unit candidates are structural. They are **not a speedup**, throughput, cycle, or area claim; the missing measurements (per-unit throughput, communication latency, energy/area) are named per unit in `processing_unit_candidates.yaml`.
