# Atlas MLIR: pretrained Qwen bringup

The workflow now starts from the actual **DeepSeek-R1-Distill-Qwen-1.5B
checkpoint**. A 32-token prefill plus two greedy decode steps through all 28
layers exports to **6,176 ATen operations, 591 projection calls and 84 attention
calls**. This supplies model-derived shapes and invocation counts.

## Completed

- The exported full session matches eager FP32 PyTorch exactly.
- Five PyTorch regions—first-layer norm, attention, MLP, gate projection and
  down projection—are extracted from that graph and independently reproduce
  their captured outputs exactly. Original source nodes and parameters are retained.
- Four distinct projection-launch variants execute on `npu_model`, match
  independent target-arithmetic goldens, and pass Atlas machine-stream and
  timing verification. `atlas-emit` reproduces the assembler's words.
- Gate and down panels also pass those checks with actual pretrained weights
  and captured model activations, beyond the synthetic model-shape benchmark.

## Performance

Output panels contain 128 columns and keep the full source K reduction.
Both prefill and single-row decode execute 32 physical rows.

| Projection panel | Simulator cycles | Launches in the selected session |
| --- | ---: | ---: |
| 32×1536×128 | 358,100 | 17,673 |
| 32×8960×128 | 2,056,340 | 1,008 |

The 591 logical projection calls become **18,681 physical panels**, totaling
**8.401 billion simulator cycles** under this unfused, cold-launch policy.
That is a projection-only estimate: unsupported kernels, host work, launch
overhead and full graph integration are excluded. It is not model latency.
No Qwen-specific traffic-reuse recipe is currently selected.

On two actual first-layer panels, relative L2 error against captured FP32
outputs is **4.40% for gate** and **8.62% for down**. Both remain bit-exact to
the selected target arithmetic. These spot checks do not establish full-model
quantized accuracy or error propagation through a session.

## Existing problems and priorities

1. **Operator coverage:** attention needs 128-channel heads; the existing
   generator supports 64. RMSNorm needs width 1536 and epsilon 1e-6; its current
   standalone generator supports width 2048 with a different epsilon. These
   variants remain explicitly unsupported.
2. **Missing graph work:** QKV biases, SiLU/gating, residuals, RoPE, embedding,
   masks and cache/layout operations remain unmapped in this Qwen placement.
3. **Automatic lowering:** real batched projections emit `linalg.batch_matmul`,
   which the current normalizer cannot parse. Raw Linalg is retained. The
   successful programs above use library schedules with compiler verification
   and encoding, rather than Merlin-generated schedules.
4. **Scheduling and precision:** prioritize streaming allocation and reusable
   Qwen schedules, decode-aware execution, and calibration/error budgets on
   actual activations. Full K=8960 execution demonstrates that long reductions
   are feasible on the simulator despite earlier compiler allocation timeouts.

No full-model Atlas ELF, board execution, perplexity or token-throughput result
is claimed. Raw tensors, checkpoints and images stay local; results are committed
as Markdown. Qwen work is on `feat/qwen-kernels`; packaging and CI remain separate.

Details: [bringup report](qwen-model-bringup.md),
[complete MODEL-BENCH table](../full_model/MODEL-BENCH.md),
[reproduction](../full_model/README.md).
