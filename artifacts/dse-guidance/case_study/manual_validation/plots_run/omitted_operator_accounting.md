# Operator recovery accounting (all)

> Attention bmm / softmax / norm are **NOT erased** — the flat capture lowered them to `linalg.generic` but they are re-parsed. Linear-GEMM and attention MACs are both recovered from IR shapes (no model-card config). `visible_linear_fraction` = linear / (linear+attention) answers how much of the recovered MAC work is the linear-GEMM geometry. What genuinely STAYS erased: the K-loop, KV state across the decode loop, and packed low-bit layout + scales.

| workload | linear ops | linear MACs | attn ops | attn MACs | **visible_linear_frac** | softmax | norm | conv | elementwise |
|---|---|---|---|---|---|---|---|---|---|
| bitvla | 30 | 39452672 | 0 | 0 | **1.000** | 12 | 0 | 0 | 324 |
| groot_n1d7 | 116 | 20393361408 | 32 | 105799680 | **0.979** | 48 | 240 | 0 | 158 |
| molmoact | 34 | 8419016704 | 0 | 0 | **1.000** | 24 | 0 | 0 | 215 |
| openvla | 30 | 15269888 | 0 | 0 | **1.000** | 12 | 0 | 0 | 116 |
| pi05 | 777 | 2146035695616 | 232 | 81230200832 | **0.964** | 348 | 1485 | 6 | 1245 |
| rdt | 21 | 39466041344 | 25 | 2416956928 | **0.942** | 0 | 0 | 0 | 29 |
| rdt2 | 26 | 991854592 | 0 | 0 | **1.000** | 12 | 0 | 0 | 115 |
| smolvla | 302 | 110595843584 | 84 | 483485810688 | **0.186** | 96 | 225 | 2 | 1025 |
| tiny_llama | 30 | 923795456 | 0 | 0 | **0.999** | 12 | 0 | 0 | 120 |
| xr0 | 19 | 1115879424 | 14 | 8699904 | **0.992** | 0 | 0 | 0 | 111 |

**Still erased** (genuinely absent in the flat capture, not merely unparsed): the K-loop trip count, KV state across the decode loop, and packed low-bit layout + scales — needing a loop-preserving (+ KV) capture and a low-bit recapture. Per-workload: `visible_vs_erased_work_table.csv`.

