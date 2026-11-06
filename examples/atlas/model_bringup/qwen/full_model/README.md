# Pretrained Qwen model bringup

This follows the kernel-library coverage workflow: export a real model session, map its operations to candidate kernel families, run every distinct launch at its model-derived shape, and report unsupported work. It is separate from the older hand-written 32×32 microbenchmarks in the parent directory.

## Authentication limitation

Pretrained execution is currently disabled. The repository has an independently pinned weights SHA-256, but no independently trusted pins for `config.json`, `generation_config.json`, `tokenizer.json`, or `tokenizer_config.json`. The downloader writes observed hashes under `requested_model` and `requested_revision`; those fields describe the request, and do not authenticate the local snapshot. A local `checkpoint-identity.json` is never a trust anchor. Setup rejects the checkpoint after recording observations, and the pretrained driver, loader, and inventory reject it before model/config/tokenizer loading. Restoring this path requires reviewing and adding all missing hashes to `checkpoint_identity.py`; generating hashes from the same local directory cannot establish trust.

The separate `launch_bench.py` random-weight fixture remains supported without a checkpoint. It constructs its architecture explicitly, and does not claim pretrained weights, tokenizer identity, or a pinned Hugging Face snapshot. Historical real-checkpoint results below are prior evidence, not a claim that the current pretrained path can execute with the incomplete trust manifest.

## Run (pending independent metadata pins)

Use a compiler wheel or `--atlas-mlir-root` / `ATLAS_MLIR_ROOT` / a sibling compiler checkout, plus built `atlas-opt` and `atlas-emit` selected by `--atlas-bin` or `ATLAS_OOT_BIN_DIR`. The optional setup action also needs `uv` and Git:

```sh
python3 examples/atlas/model_bringup/qwen/full_model/run.py --setup
python3 examples/atlas/model_bringup/qwen/full_model/run.py
```

Setup prepares isolated CPU capture/model environments, checks out pinned kernel and simulator sources, downloads the pretrained checkpoint (about 3.6 GB), and checks the complete weights SHA-256, records observed file hashes, and fails closed while metadata pins are missing. The default session is a 32-token prefill plus two greedy decode steps through all 28 layers. The checkpoint's BF16 parameters are promoted exactly to FP32 for the reference computation; attention uses the eager implementation from Transformers 4.44.0.

The second command requires a fresh `out/qwen-full-model`; use `-o DIR` for a new run. Defaults, including 128-column output panels, are explicit in `mapping.yaml`. Every panel keeps the full source reduction width. `--plan-only` skips execution and real-value capture. `--skip-real-capture` runs the inventory/bench alone. Existing environments/checkouts can be selected with `--capture-python`, `--model-python`, `--library-root`, `--model-root`, and `--checkpoint`. `ATLAS_KERNELS_ROOT` and `NPU_MODEL_ROOT` are source-root alternatives. Supplied local source roots need not match historical pins; `run.json` records their observed revisions, historical pins, and tool hashes. The default worker count is 1; explicit counts from 1 through 8 are accepted.

## What is checked

1. `loader.py` authenticates every consumed checkpoint file before loading, and
   currently refuses execution because metadata pins are missing. There is no random
   initialization fallback. No model computation is replaced by authored kernels.
2. The kernel library's `coverage/export_ops.py` exports the session. `inventory.py`
   labels prefill/decode and checks complete projection counts: 197 for prefill
   and 394 across the two decode steps. Checkpoint hashes join the inventory.
3. The library's `build_table.py` applies the explicit placement policy. Missing
   bias, nonlinear, residual, layout and cache operations remain unmapped.
4. `bench_model.py` measures the resulting launches on seeded synthetic data.
   It requires the library verified tier, equal assembler/emitter
   words, independent arithmetic goldens where available, and equal functional
   and cycle-model outputs. These schedules come from the kernel library, not
   Merlin's semantic compiler. Missing variants remain failures of coverage.
5. `capture_boundaries.py` exports and executes the real session, requires exact
   agreement with eager PyTorch, and cuts executable PyTorch graph regions for
   first-layer normalization, attention, MLP, gate projection and down projection.
   Each extracted region must exactly reproduce its observed graph outputs from
   actual boundary tensors. Local artifacts include Python modules, inputs,
   goldens, source-node/parameter provenance and Linalg capture diagnostics.
6. `qualify_projections.py` runs first-128-column gate/down panels with the actual
   captured activations and checkpoint weights through the same generator,
   compiler gate, target-arithmetic golden, and simulator. Error is measured
   against the saved full-model FP32 region outputs.

## Historical pinned inputs

| Component | Revision |
| --- | --- |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` |
| `ucb-bar/atlas-kernels` | `80c5f27d4bcded9c1ca7618a17ffcdbae2be7a39` |
| `ucb-ee194-tapeout/npu_model` | `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5` |

The capture stack is Python 3.12, CPU Torch 2.10.0, Transformers 4.44.0 and torch-mlir 20261001. Device simulation uses Python 3.14 and CPU Torch 2.11.0. Tool binaries, checkpoint files, inputs, source graphs and generated programs are hashed in the local artifacts.

## Results and limits

Read [MODEL-BENCH.md](MODEL-BENCH.md) for the measured library launches and [the model bringup report](../docs/qwen-model-bringup.md) for extraction, real-value qualification, current blockers and recommended work.

This establishes real-model coverage and validated graph extraction. The projection benchmark is an unfused candidate placement, and real-value checks cover two projection panels. It does not establish a complete quantized model, generated host dispatch, model ELF, hardware execution, perplexity, or throughput. All datasets, checkpoints, tensor captures, programs, and images remain under ignored local output directories. Source code, mapping policy, regression tests, and Markdown reports are tracked.

## Current integration

The imported reports describe the historical kernel pin above. The current library supports D128 attention and width-1536 RMSNorm with explicit epsilon. The mapping requests Qwen epsilon `1.0e-6` and retains causal masking and export-derived GQA dimensions. New numerical qualification is reported separately from the historical evidence.

`capture_boundaries.py --skip-linalg` retains graph extraction and exact eager-versus-export checks when torch-mlir is unavailable. `--loader PATH` selects a local module exporting `get_model_and_inputs`; the default remains the pinned pretrained loader without a random-initialization fallback. Capture reports mark skipped Linalg diagnostics explicitly.

## Opt-in projection epilogues

`run.py --projection-fusions` and `launch_bench.py --projection-fusions` additionally execute the kernel library's `kernels/projection_gemm/qualify_qwen_fusions.py --timed` and retain `projection-fusions/summary.json`. This qualifies exact BF16 round-to-nearest-even channel bias for Q/K/V and residual addition for o/down panels on seeded synthetic inputs, with full reduction widths, independent arithmetic goldens, verified compiler encoding, and measured isolated device costs. Plan-only runs record the qualification as PENDING. The option requires a library checkout containing that runner and preserves the default coverage mapping and census.

The candidate uses explicit tiled BF16 operands and fuses the add into an existing projection launch. K1536 retains the admitted resident-A option; K8960 uses the base schedule. This is a bringup qualification option: Qwen has no connected Atlas host graph, and the flag does not claim one. Exactness is relative to the quantized Atlas projection followed by a BF16 RNE sum; it does not reproduce the full FP32 reference linear layer. Host time saved, model accuracy, real captured bias/residual qualification, and production staging remain unmeasured.
