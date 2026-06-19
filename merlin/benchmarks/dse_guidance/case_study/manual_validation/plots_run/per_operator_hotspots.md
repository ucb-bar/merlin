# Per-operator hotspots (all)

> Which few operators dominate the constraints DSE must size for. Structural quantities (MACs / weight bytes / tile padding waste / avoidable reload) recovered from the capture — no latency, throughput, or performance claim.

Total operators analyzed: **1385**.

**Dominant op (by MACs):** `` in rdt — 4096x4096x2048 = 34,359,738,368 MACs (87% of its workload), class squareish_gemm.

## Top ops by MACs

| workload | op | shape M×N×K | MACs | % of workload | class |
|---|---|---|---|---|---|
| rdt |  | 4096×4096×2048 | 34,359,738,368 | 87% | squareish_gemm |
| pi05 | model.paligemma_with_expert.paligemma.model.language_model.layers.slice(None, 18, None).0.mlp.gate_proj | 968×16384×2048 | 32,480,690,176 | 2% | wide_skinny |
| pi05 | model.paligemma_with_expert.paligemma.model.language_model.layers.slice(None, 18, None).0.mlp.up_proj | 968×16384×2048 | 32,480,690,176 | 2% | wide_skinny |
| pi05 | model.paligemma_with_expert.paligemma.model.language_model.layers.slice(None, 18, None).0.mlp.down_proj | 968×2048×16384 | 32,480,690,176 | 2% | projection_like |
| pi05 | model.paligemma_with_expert.paligemma.model.language_model.layers.slice(None, 18, None).1.mlp.gate_proj | 968×16384×2048 | 32,480,690,176 | 2% | wide_skinny |
| pi05 | model.paligemma_with_expert.paligemma.model.language_model.layers.slice(None, 18, None).1.mlp.up_proj | 968×16384×2048 | 32,480,690,176 | 2% | wide_skinny |
| pi05 | model.paligemma_with_expert.paligemma.model.language_model.layers.slice(None, 18, None).1.mlp.down_proj | 968×2048×16384 | 32,480,690,176 | 2% | projection_like |
| pi05 | model.paligemma_with_expert.paligemma.model.language_model.layers.slice(None, 18, None).2.mlp.gate_proj | 968×16384×2048 | 32,480,690,176 | 2% | wide_skinny |
| pi05 | model.paligemma_with_expert.paligemma.model.language_model.layers.slice(None, 18, None).2.mlp.up_proj | 968×16384×2048 | 32,480,690,176 | 2% | wide_skinny |
| pi05 | model.paligemma_with_expert.paligemma.model.language_model.layers.slice(None, 18, None).2.mlp.down_proj | 968×2048×16384 | 32,480,690,176 | 2% | projection_like |

## Top ops by tile padding waste (tile-hostility)

| workload | op | shape M×N×K | best tile waste | class |
|---|---|---|---|---|
| rdt |  | 1×2048×256 | 7.000 | gemv_like |
| rdt |  | 1×2048×2048 | 7.000 | gemv_like |
| rdt |  | 1×2048×256 | 7.000 | gemv_like |
| rdt |  | 1×2048×2048 | 7.000 | gemv_like |
| openvla | lm_head | 1×512×128 | 7.000 | gemv_like |
| openvla |  | 1×512×128 | 7.000 | gemv_like |
| openvla |  | 1×512×128 | 7.000 | gemv_like |
| openvla |  | 1×512×128 | 7.000 | gemv_like |
| openvla |  | 1×128×512 | 7.000 | gemv_like |
| openvla |  | 1×256×128 | 7.000 | gemv_like |

## Regions by avoidable weight reload (residency target)

| workload | region | avoidable reload (B) | weight bytes (B) |
|---|---|---|---|
| molmoact | repeated_head | 26,512,195,584 | 3,787,456,512 |
| pi05 | repeated_head | 15,479,341,056 | 1,719,926,784 |
| groot_n1d7 | repeated_head | 6,601,310,208 | 2,200,436,736 |
| tiny_llama | repeated_head | 3,686,793,216 | 614,465,536 |
| smolvla | repeated_head | 1,855,134,720 | 206,126,080 |
| rdt | repeated_head | 1,572,864,000 | 393,216,000 |
| rdt2 | repeated_head | 1,238,958,080 | 309,739,520 |
| xr0 | repeated_head | 676,347,904 | 169,086,976 |
| bitvla | repeated_head | 34,603,008 | 5,767,168 |
| openvla | repeated_head | 18,874,368 | 3,145,728 |
