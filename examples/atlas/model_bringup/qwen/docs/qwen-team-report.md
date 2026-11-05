# Atlas MLIR: Qwen kernel results

**Measured November 4, 2026.** Five manually written PyTorch subgraphs matching
DeepSeek-R1-Distill-Qwen-1.5B dimensions now compile through torch-mlir, Merlin and Atlas
and execute successfully on the NPU simulator. Scheduling reduces cycles by
**7–15%**. The current memory system limits performance; numerical precision
and allocation scalability remain the main blockers to larger model sections.

## Performance

Results use the simulator's default VCS-derived memory curve.

| Kernel | Unscheduled cycles | Scheduled cycles | Cycle reduction | Roofline efficiency* |
| --- | ---: | ---: | ---: | ---: |
| Attention | 18,866 | 16,127 | 14.5% | 94.4% |
| SwiGLU partition | 8,833 | 7,691 | 12.9% | 89.9% |
| Projection K=128 | 8,324 | 7,717 | 7.3% | 89.6% |
| Projection K=256 | 15,060 | 13,727 | 8.9% | 90.8% |
| Projection K=512 | 28,532 | 25,745 | 9.8% | 91.5% |

*Roofline efficiency is resource-lower-bound cycles divided by measured cycles;
it measures throughput relative to that bound, rather than peak compute utilization.*

All five are **DRAM-bound**, with only 19–30 matrix FLOP per transferred byte.
With work and traffic unchanged, the optimistic remaining cycle reduction is
approximately **6–10%**. Larger gains require reducing transfers or increasing
data reuse. These comparisons include boundary-DMA changes; scheduling alone
still saves 13.5% for attention and 11.4% for SwiGLU against a boundary-only baseline.

A hypothetical faster-memory test exposes additional limits: SwiGLU's largest
resource charge becomes the VPU, and K=512's becomes VMEM-to-register loads.
Bound efficiency falls to 28–38%, exposing latency/dependency and overlap costs
that need further attribution.

## Existing problems

- **FP8 accuracy:** bounded signed SwiGLU inputs produce an all-zero output
  and 100% relative error. The gated intermediate is too small for the selected
  unit-scale BF16→FP8 policy and is flushed away. Emitted code matches that
  policy; the precision choice is unsuitable for these inputs.
- **Projection accuracy:** relative error against float32 grows from 1.3% at
  K=128 to 9.2% at K=512 on the positive panel. Quantization and accumulation
  error still need separate attribution.
- **Allocation scalability:** the full K=1536 projection times out during
  allocation, including a 300-second attempt. K=128/256/512 compile in roughly
  0.27/1.06/4.37 seconds.
- **Frontend coverage:** full-width RMSNorm/QKV and residual MLP capture
  successfully but fail target lowering. Broader nonlinear-domain qualification,
  RoPE and causal-mask lowering also remain necessary for full-model support.

## Recommended priorities

1. Add activation-aware scaling, scale compensation and higher precision fallbacks.
2. Use streaming schedules with bounded live tiles for long reductions.
3. Keep data resident across kernels/windows and reuse K/V across query heads.
4. Expand Qwen operator coverage and test pretrained activation distributions.

## Validation and scope

All 30 roofline runs passed exact virtual-reference output and memory-guard
checks, with zero runtime errors. Packaging fixes, a two-command demo and CI
checks are on the separate `feat/pip-package` branch; installed wheels passed Python 3.10/3.12 checks, and all
149 CTest entries passed. Hosted CI execution is pending.

These are **simulator measurements with synthetic tensors**. Attention covers
one head's 32-token unmasked window with all 128 head channels; SwiGLU is a
32-channel partition. Full-model accuracy, token throughput and hardware
performance remain unmeasured.

Details: [compilation and accuracy](qwen-results.md),
[roofline analysis](qwen-roofline.md).
