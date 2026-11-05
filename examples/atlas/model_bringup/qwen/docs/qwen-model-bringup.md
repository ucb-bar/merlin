# Qwen model-driven bringup results

The checkpoint is now loaded and the full model session is exported, rather
than reconstructing model-shaped microbenchmarks. This report covers a 32-token
prefill plus two greedy decode steps through all 28 layers of
`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.

The pinned pretrained BF16 weights are promoted to FP32 for the reference run.
This is a new fixed-input experiment, not the historical TinyLlama/SmolVLA kit.
The prompt and loader are in `examples/atlas/model_bringup/qwen/full_model/loader.py`.

## What now works

- Full-session export: **6,176 ATen operations, 591 projection calls, and 84
  attention calls**, with 197 projections in prefill and 394 in decode.
- The exported session reproduces eager PyTorch outputs exactly, including all
  three greedy token selections.
- Five executable PyTorch regions are extracted directly from that graph and
  reproduce their actual boundary outputs exactly. Source operations and
  parameter names are retained; these regions are not manually reimplemented.
- Four distinct projection-launch variants pass the machine-stream and timing
  verifiers, match assembler/emitter words, match independent target-arithmetic
  goldens, and produce identical functional/timed outputs on `npu_model`.
- Two additional projection panels pass the same checks using **actual captured
  activations and checkpoint weights**, rather than synthetic benchmark data.

## Model-derived kernel benchmark

Every output panel has 128 columns and preserves the full source K reduction.
Decode's single logical row is padded to the target's physical 32 rows.

| Projection product | Logical rows | Physical M×K×N | Model cycles per launch | Count in session | Relative L2 error* |
| --- | ---: | --- | ---: | ---: | ---: |
| Other projections / LM head | 1 | 32×1536×128 | 358,100 | 11,782 | 6.67% |
| Other projections / LM head | 32 | 32×1536×128 | 358,100 | 5,891 | 5.86% |
| FFN down | 1 | 32×8960×128 | 2,056,340 | 672 | 13.12% |
| FFN down | 32 | 32×8960×128 | 2,056,340 | 336 | 11.02% |

*Benchmark errors compare decoded FP8/BF16 results with FP32 products on seeded
synthetic inputs. Bit-exactness is separately checked against target arithmetic.

The 591 logical projection calls expand to **18,681 physical panel launches**.
Their weighted sum is **8,401,492,020 simulator cycles**. This is a cold-launch,
projection-only estimate; it excludes unsupported attention/norm launches,
unmapped operations, host time, launch overhead and cross-launch integration.
It is not end-to-end model latency or token throughput.

All four use the library's default schedule because no dedicated traffic-reuse
recipe covers these Qwen shapes. Prefill and decode have identical per-panel
cycles because both execute 32 physical rows. This exposes opportunities for
Qwen-specific reuse and decode schedules.

## Qualification on real model tensors

The first 128 output columns of the first-layer gate and down projections were
run on their actual captured inputs and pretrained weights. Input and weight
scales are calibrated powers of two using the library's existing policy.

| Captured projection | Full K | Simulator cycles | Relative L2 error vs captured FP32 output | Target golden | Compiler checks |
| --- | ---: | ---: | ---: | --- | --- |
| first_gate_projection | 1,536 | 358,100 | 4.40% | exact | stream + timing pass |
| first_down_projection | 8,960 | 2,056,340 | 8.62% | exact | stream + timing pass |

These are independent checks on native model boundary tensors. Errors are not
propagated through an entire quantized session, and no perplexity or token
agreement claim is made for Atlas execution.

## Graph extraction and compiler gaps

| Actual first-layer region | Source operations | Extracted PyTorch result | Linalg handoff |
| --- | ---: | --- | --- |
| first_layer_norm | 10 | exact | normalized capture produced |
| first_layer_attention | 53 | exact | raw Linalg emitted; batch_matmul parser refusal |
| first_layer_mlp | 5 | exact | raw Linalg emitted; batch_matmul parser refusal |
| first_gate_projection | 1 | exact | raw Linalg emitted; batch_matmul parser refusal |
| first_down_projection | 1 | exact | raw Linalg emitted; batch_matmul parser refusal |

The attention cut retains its cache outputs and external mask/rotary inputs;
closure is checked through actual graph-boundary values. The MLP cut contains
all three full-width projections, SiLU and gating.

The following work remains:

1. **Attention:** all 84 calls require head dimension 128; the existing library
   generator supports 64. Prefill and decode shapes are reported as `no-variant`.
2. **RMSNorm:** all 171 calls require width 1536 and epsilon 1e-6; the standalone
   generator supports width 2048 with a different epsilon. Both physical launch
   variants are `no-variant`. The width hint preserves the original norm width;
   the scalar reciprocal-root result itself has final dimension 1.
3. **Unmapped computation:** 138 aggregated rows remain, including Q/K/V bias,
   SiLU/gating, residual adds, RoPE, embedding, cache/layout operations and mask
   preparation. They have no implemented Qwen host graph or qualified fused
   placement in this experiment. High planned FLOP coverage does not close them.
4. **Automatic compiler path:** actual batched PyTorch projections become
   `linalg.batch_matmul`; the current xDSL normalizer does not register that op.
   Raw Linalg is retained with an explicit diagnostic. Successful library
   assembly execution does not establish semantic-compiler lowering of these
   full regions.

Full K=1536 and K=8960 programs execute successfully with library schedules.
This supports treating the earlier semantic-compiler allocation timeout as a
compiler scalability problem, rather than evidence that those reductions cannot
execute on the simulator. Streaming allocation and schedule templates are a
concrete integration target.

## Reproduction and evidence

```sh
python3 examples/atlas/model_bringup/qwen/full_model/run.py --setup
python3 examples/atlas/model_bringup/qwen/full_model/run.py -o out/qwen-full-model-new
```

The end-to-end driver was exercised with its isolated capture/model environments.
Three graph-extraction tests additionally cover exact boundary execution,
separation of repeated model calls and explicit refusal of missing regions.
The local measured run is `out/qwen-full-model-final`: it holds the inventory,
checkpoint identity, kernel program receipts, tensors, generated PyTorch modules,
raw/normalized Linalg and qualification outputs. Raw data and images are not
committed. [MODEL-BENCH.md](../full_model/MODEL-BENCH.md) contains
all launch results and the unmapped-operation table.

Pinned checkpoint: `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`; atlas-kernels:
`80c5f27d4bcded9c1ca7618a17ffcdbae2be7a39`; npu_model:
`47b8599d8ffbe5ddd96059ffe2b5be2427a410f5`.

The kernel-library path supplies hand-written schedules checked and encoded by
Atlas MLIR. Merlin-generated full-model schedules, host dispatch, a model ELF,
and board/RTL execution are not established by this run. Packaging and CI remain
on `feat/pip-package`; this implementation is confined to `feat/qwen-kernels`.
