# Multi-Task DiT bringup

Capture LeRobot Multi-Task DiT through Merlin's capture worker, inventory the complete exported forward, and qualify isolated Atlas launches using real captured operands. The float export and the FP8/BF16 device launch experiments are separate precision boundaries. Neither establishes connected policy execution or trained task accuracy.

The pretrained input is the community checkpoint `NONHUMAN-RESEARCH/multi-task-dit-training-fruits`, revision `d8e9ce7000808c3a8c52d291f5d7f7899713cd27`. It is not an official LeRobot checkpoint. Its `config.json` SHA-256 is `cb76a7924ce2aee52ae699425a01491dc0c9da2fd781245bd99ccd46a1718f22`, and its `model.safetensors` SHA-256 is `b2bb33c5b137abf32ace687b06fff60cb4c25dc08696f61a0ce19115a189c698`. The loader verifies every entry in its supplied `SHA256SUMS`; capture receipts retain configuration, processor, tokenizer, and weight identities.

The random workload retains the earlier two-camera, ten-step DDIM setup with random DiT and text-projection weights and pretrained CLIP encoders. The checkpoint workload uses its saved three-camera architecture, 100 Euler flow-matching steps, observation normalization, and action unnormalization. Its raw image input is `[1,2,3,3,376,672]`; eval-mode crops produce 224 x 224 images. Both workloads use horizon 32, six DiT blocks of width 512, eight 64-channel heads, and output `[1,24,14]`. Inputs are synthetic, and initial noise is explicit. Tokenization and policy queues remain outside the tensor ABI.

## Capture

Use Python 3.12, torch 2.10.0+cpu, torch-mlir 20260531.828, xDSL 0.68.0, LeRobot 0.6.1, transformers 5.5.4, diffusers 0.39.0, and Model2MLIR `8893a37480b4560f701089e5bf408644573a906f`. Install from local wheels with network access disabled. Capture receipts record observed versions. All inspected Merlin lock versions match except Pillow: the local CPython 3.12 environment has 12.3.0, while the lock requires 12.2.0. The exact cached Pillow wheel targets CPython 3.14, and the completed local source-cache audit found no compatible source. Full lock matching remains pending. `HF_HOME` must contain CLIP revision `57c216476eefef5ab752ec549e440a49ae4ae5f3`. The explicit torchvision compatibility flag supplies only a missing NMS schema for import; this model never invokes NMS or compiled image operators.

```bash
export TMPDIR="$PWD/tmp" TEMP="$PWD/tmp" TMP="$PWD/tmp"
mkdir -p "$TMPDIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MTDIT_TORCHVISION_SCHEMA_COMPAT=1
EXAMPLE=merlin/examples/atlas/model_bringup/multi_task_dit
nice -n 10 venv/bin/python "$EXAMPLE/capture.py" --m2m-dir "$M2M" \
  --out "$PWD/out/capture-random"
nice -n 10 venv/bin/python "$EXAMPLE/capture.py" --m2m-dir "$M2M" \
  --checkpoint "$CHECKPOINT" --out "$PWD/out/capture-checkpoint"
```

The pinned xDSL 0.68.0 resolves the earlier printer API mismatch. The original DDIM scheduler still leaves 38 mangled `aten_select_int` calls, so that failure is not a version artifact. The static lookup specialization removes them without changing eager output. A parsed census counts calls to external declarations, including sanitized symbols; the worker tail text alone is insufficient.

The default backend invokes Merlin's own `_m2m_capture_worker`, preserves its bundle, parses opaque calls, and checks exported and Model2MLIR-prepared programs against eager exactly. Fresh output directories are required. Set `MTDIT_STATIC_SCHEDULER=0` for the original DDIM table-indexing control. The default freezes scalar lookup addressing while retaining scheduler arithmetic. `--backend torch_mlir` retains the earlier public Model2MLIR companion and `--region block|vision|text|dit`; it is a fallback capture path, and receipts distinguish it from Merlin's worker.

The public companion for the checkpoint uses the same loader through an explicit environment setting:

```bash
MTDIT_CHECKPOINT="$CHECKPOINT" nice -n 10 venv/bin/python "$EXAMPLE/capture.py" \
  --backend torch_mlir --m2m-dir "$M2M" --threads 2 \
  --out "$PWD/out/capture-checkpoint-torch-mlir"
```

Its trained prepared-export check also fails exactly; the complete MLIR and bytecode remain retained. The native torch-mlir module verifies, while the xDSL 0.68 bridge cannot parse the custom `tensor.extract_slice` format. These are separate diagnostics, and this path does not use an FX fallback.

The trained checkpoint raw export equals its original eager objective exactly. The stock worker bundle completed with zero opaque calls, but its prepared/decomposed replay differs from eager by up to `3.0517578125e-05`; that capture is marked failed and retained for diagnosis. Random-policy capture passes original, raw-export, and prepared-export equality exactly. The trained prepared-export gate remains pending.

## Inventory and launch qualification

The checkpoint forward inventory contains 32,525 operations across 41 kinds, 3,545 linear calls, one patch convolution, and 624 attention calls. Candidate linear and attention placement accounts for 359,531,242,176 of 361,690,282,525 semantic FLOPs (99.4031%). The remaining 2,159,040,349 semantic FLOPs and movement obligations stay uncovered. This percentage is semantic work, not measured runtime share.

`launch_inventory.py` reads the full exported program, uses atlas-kernels' operation/FLOP conventions, and retains actual projection and attention boundaries with hashes. It records every denoising iteration and separates projection bias from GEMM. Candidate geometric coverage is distinct from directly supported FP32 coverage. LayerNorm is not RMSNorm; patch gathering, embeddings, QuickGELU, adaLN modulation, rotary layout, scheduler arithmetic, residual glue, and movement require explicit accounting.

```bash
nice -n 10 venv/bin/python "$EXAMPLE/launch_inventory.py" \
  --export out/capture-checkpoint/exported.pt2 \
  --inputs out/capture-checkpoint/inputs.npz --expected out/capture-checkpoint/eager.npy \
  --receipt out/capture-checkpoint/receipt.json --library "$KERNELS" \
  --out out/inventory-checkpoint
```

`qualify.py` uses the shared kernel library, full-K projection panels split at 128 with tails padded to a multiple of 32, supplied real Q/K/V, and captured attention masks. Independent target-arithmetic goldens apply to projections. Attention retains FP32 fidelity metrics and exact timed/functional equality without an independent byte golden. The driver requires compiler tier `verified` and assembler/emitter word equality, then runs functional and timed `npu_model` at `47b8599` with `numerics="rtl"`. The official performance pass uses unmodified `74982e7` speed-of-light and roofline APIs in a separate process. See the driver's `--help` for preparation, worker, report, and run modes.

Use a separate Python 3.14 simulator environment with NumPy, PyTorch, PyYAML, and tabulate. Set `NPU_RTL` to unmodified `47b8599`, `NPU_PERF` to unmodified `74982e7`, and `ATLAS_BIN` to the verified compiler tools. Preparation deliberately opts into the library's d64 synchronous-DMA attention variant; the default attention stream is unchanged.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nice -n 10 "$MODEL_PYTHON" "$EXAMPLE/qualify.py" --prepare \
  --inventory out/inventory-checkpoint/boundaries.json \
  --out out/qualification-checkpoint --library "$KERNELS" \
  --model-root "$NPU_RTL" --perf-root "$NPU_PERF" --atlas-bin "$ATLAS_BIN" \
  --attention-sync-dma
nice -n 10 "$MODEL_PYTHON" "$EXAMPLE/qualify.py" --run --mode rtl \
  --out out/qualification-checkpoint --workers 2
nice -n 10 "$MODEL_PYTHON" "$EXAMPLE/qualify.py" --run --mode official \
  --out out/qualification-checkpoint --workers 2
nice -n 10 "$MODEL_PYTHON" "$EXAMPLE/qualify.py" --report \
  --out out/qualification-checkpoint
```

All 22 launch kinds pass compiler tier `verified`, assembler/emitter equality, exact RTL timed/functional equality, and exact official/RTL output and cycle agreement. All 19 projection kinds also pass independent target-arithmetic goldens. The 44,640 launches per checkpoint forward comprise 44,016 projection panels and 624 attention calls.

| Count-weighted device cycles | Official SOL cycles | SOL fraction | Modeled headroom |
| ---: | ---: | ---: | ---: |
| 29,428,888,176 | 22,248,586,752 | 75.6012% | 7,180,301,424 |

Artifacts, arenas, programs, captures, and hash receipts belong in `out/`, outside Git. Cycle totals are count-weighted isolated-launch estimates; they exclude host dispatch, connected transfers, uncovered work, and hardware timing. The companion atlas-kernels `coverage/ROOFLINE-MTDIT.md` records the measured scope and pending gates.

## Offline tests

```bash
nice -n 10 venv/bin/python -m pytest -q -p no:cacheprovider "$EXAMPLE"
nice -n 10 venv/bin/python -m pytest -q -p no:cacheprovider \
  merlin/examples/atlas/model_bringup/qwen/tests
```

`inventory.py` retains the earlier verified float-MLIR census. The float path does not quantize tensors or compile a connected Atlas policy. No task accuracy, host executable, hardware RTL, QEMU, or FireSim result is implied by isolated launch qualification.
