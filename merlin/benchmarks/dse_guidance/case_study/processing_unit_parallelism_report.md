# Processing-unit & hierarchy guidance

> Which processing-unit shapes the workloads imply, and whether the evidence favors one bigger unit, multiple identical units, or specialized units. **Structural search-space guidance — no speedup, cycle, or area claim.**

## Resource pressure (where the work is)

| resource class | present | ops | MAC share | basis |
|---|---|---|---|---|
| dense_gemm | True | 3 | 67.1% | compute_macs |
| skinny_gemm_or_gemv | True | 131 | 32.9% | compute_macs |
| epilogue_or_requant | True | 60 | 83.5% | op_count |
| dma_or_memory | True | 0 | 0.0% | structural |
| control_or_dispatch | True | 0 | 0.0% | structural |
| attention_softmax_or_reduction | False | 0 | 0.0% | unavailable |
| conv_or_patch_embed | False | 0 | 0.0% | unavailable |
| sparse_or_skip | False | 0 | 0.0% | unavailable |

## One bigger unit vs. many identical vs. specialized

- **Average inter-op parallelism is low (1.24×)** (see `dag_parallelism_report.md`): the dependency DAG is near-sequential, so **many identical units would be hard to keep busy** by inter-op concurrency alone — the parallelism to exploit is *intra-op* sharding.
- **Compute splits across two shapes:** dense GEMM is 67% of MACs (rdt, rdt2) while skinny/GEMV is 33% (groot_n1d7, molmoact, openvla, rdt, rdt2, small_llama, tiny_llama). This favors **specialized units** (a matrix engine *and* a GEMV/vector engine) over one universal unit.
- **Plus structural units:** an `epilogue_requant_unit` (the addmm bias/activation fuses onto the GEMM), a `dma_engine` (resident loop-invariant weights), and a `loop_controller` (the bounded K-loop) — each backed by recovered structure.
- **Honestly absent:** `attention_kv_engine` and `conv_engine` have no supporting operators in the captures (attention is lowered, no conv) — listed `unavailable`.

## Candidate units

| unit | evidence_for | supporting workloads |
|---|---|---|
| matrix_engine | serves dense_gemm: 67% of MACs across rdt, rdt2 | rdt, rdt2 |
| vector_gemv_engine | serves skinny_gemm_or_gemv: 33% of MACs across groot_n1d7, molmoact, openvla, rdt, rdt2, small_llama, tiny_llama | groot_n1d7, molmoact, openvla, rdt, rdt2, small_llama, tiny_llama |
| attention_kv_engine | no supporting operators in the captures | — |
| conv_engine | no supporting operators in the captures | — |
| epilogue_requant_unit | serves epilogue_or_requant (op_count; 60 across groot_n1d7, molmoact, openvla, rdt, rdt2) | groot_n1d7, molmoact, openvla, rdt, rdt2 |
| dma_engine | serves dma_or_memory (structural; structural across groot_n1d7, molmoact, openvla, rdt, rdt2, small_llama, tiny_llama) | groot_n1d7, molmoact, openvla, rdt, rdt2, small_llama, tiny_llama |
| loop_controller | serves control_or_dispatch (structural; structural across groot_n1d7, molmoact, openvla, rdt, rdt2, small_llama, tiny_llama) | groot_n1d7, molmoact, openvla, rdt, rdt2, small_llama, tiny_llama |
| scalar_control_unit | serves control_or_dispatch (structural; structural across groot_n1d7, molmoact, openvla, rdt, rdt2, small_llama, tiny_llama) | groot_n1d7, molmoact, openvla, rdt, rdt2, small_llama, tiny_llama |

**Caveat (structural, not realized):** resource pressure and unit candidates are structural. They are **not a speedup**, throughput, cycle, or area claim; the missing measurements (per-unit throughput, communication latency, energy/area) are named per unit in `processing_unit_candidates.yaml`.
