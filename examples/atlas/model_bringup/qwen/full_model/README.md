# Pretrained Qwen model bringup

This follows the kernel-library coverage workflow: export a real model session,
map its operations to candidate kernel families, run every distinct launch at
its model-derived shape, and report unsupported work. It is separate from the
older hand-written 32×32 microbenchmarks in the parent directory.

## Run

With `uv`, Git, and built `build/bin/atlas-opt` / `atlas-emit`:

```sh
python3 examples/atlas/model_bringup/qwen/full_model/run.py --setup
python3 examples/atlas/model_bringup/qwen/full_model/run.py
```

Setup prepares isolated CPU capture/model environments, checks out pinned kernel
and simulator sources, downloads the pretrained checkpoint (about 3.6 GB), and
checks the complete weights SHA-256. The default session is a 32-token prefill
plus two greedy decode steps through all 28 layers. The checkpoint's BF16
parameters are promoted exactly to FP32 for the reference computation; attention
uses the eager implementation from Transformers 4.44.0.

The second command requires a fresh `out/qwen-full-model`; use `-o DIR` for a new
run. Defaults, including 128-column output panels, are explicit in `mapping.yaml`.
Every panel keeps the full source reduction width. `--plan-only` skips execution
and real-value capture. `--skip-real-capture` runs the inventory/bench alone.
Existing environments/checkouts can be selected with `--capture-python`,
`--model-python`, `--library-root`, `--model-root`, and `--checkpoint`.

## What is checked

1. `loader.py` loads the actual pinned Hugging Face checkpoint with no random
   initialization fallback. No model computation is replaced by authored kernels.
2. The kernel library's `coverage/export_ops.py` exports the session. `inventory.py`
   labels prefill/decode and checks complete projection counts: 197 for prefill
   and 394 across the two decode steps. Checkpoint hashes join the inventory.
3. The library's `build_table.py` applies the explicit placement policy. Missing
   bias, nonlinear, residual, layout and cache operations remain unmapped.
4. `bench_model.py` measures the resulting launches on seeded synthetic data.
   It requires at least machine-stream verification, equal assembler/emitter
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

## Pinned inputs

| Component | Revision |
| --- | --- |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` |
| `ucb-bar/atlas-kernels` | `80c5f27d4bcded9c1ca7618a17ffcdbae2be7a39` |
| `ucb-ee194-tapeout/npu_model` | `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5` |

The capture stack is Python 3.12, CPU Torch 2.10.0, Transformers 4.44.0 and
torch-mlir 20261001. Device simulation uses Python 3.14 and CPU Torch 2.11.0.
Tool binaries, checkpoint files, inputs, source graphs and generated programs
are hashed in the local artifacts.

## Results and limits

Read [MODEL-BENCH.md](MODEL-BENCH.md) for the measured library launches and
[the model bringup report](../docs/qwen-model-bringup.md) for extraction,
real-value qualification, current blockers and recommended work.

This establishes real-model coverage and validated graph extraction. The
projection benchmark is an unfused candidate placement, and real-value checks
cover two projection panels. It does not establish a complete quantized model,
generated host dispatch, model ELF, hardware execution, perplexity, or throughput.
All datasets, checkpoints, tensor captures, programs and images remain under
ignored local output directories. Only source code, mapping policy and Markdown
reports are tracked; this branch contains no CI changes.
