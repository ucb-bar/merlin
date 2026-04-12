# SaturnNPU — SmolVLA Kernel Decomposition & Test Framework

Scripts and tooling for analyzing the SmolVLA model's compute graph across
MLIR compilation levels and generating golden test data for NPU kernel
development.

## Quick Start

From the Merlin repo root, with `merlin-dev` conda environment:

```bash
# 0. Compile (if not done) — vanilla target, no accelerator plugins
conda run -n merlin-dev uv run tools/merlin.py compile \
  models/smolVLA/smolVLA.q.fp8po2.mlir \
  --target spacemit_x60 --quantized \
  --compile-to global-optimization --dump-phases

# 1. Strip weight blobs
conda run -n merlin-dev uv run tools/strip_mlir_weights.py \
  build/compiled_models/smolVLA/spacemit_x60_RVV_smolVLA.q.fp8po2/ --in-place

# 2. Run analysis
conda run -n merlin-dev uv run benchmarks/SaturnNPU/scripts/analyze_npu_graph.py \
  --torch-mlir   build/compiled_models/smolVLA/spacemit_x60_RVV_smolVLA.q.fp8po2/smolVLA.q.fp8po2.mlir \
  --linalg-input build/compiled_models/smolVLA/spacemit_x60_RVV_smolVLA.q.fp8po2/phases/module.1.input.mlir \
  --global-opt   build/compiled_models/smolVLA/spacemit_x60_RVV_smolVLA.q.fp8po2/phases/module.4.global-optimization.mlir \
  --output-dir   benchmarks/SaturnNPU/ --assert-counts

# 3. Layer decomposition trace (uses MLIR Python bindings)
conda run -n merlin-dev uv run benchmarks/SaturnNPU/scripts/trace_layer_decomposition.py \
  --linalg-input build/compiled_models/smolVLA/spacemit_x60_RVV_smolVLA.q.fp8po2/phases/module.1.input.mlir \
  --global-opt   build/compiled_models/smolVLA/spacemit_x60_RVV_smolVLA.q.fp8po2/phases/module.4.global-optimization.mlir

# 4. Plots
conda run -n merlin-dev uv run benchmarks/SaturnNPU/scripts/plot_npu_coverage.py \
  benchmarks/SaturnNPU/smolvla_graph_manifest.json
conda run -n merlin-dev python3 benchmarks/SaturnNPU/scripts/plot_sankey.py

# 5. Kernel catalog (MLIR snippets per kernel type, resolved affine maps)
conda run -n merlin-dev uv run benchmarks/SaturnNPU/scripts/extract_all_kernel_variants.py

# 6. Golden data
conda run -n merlin-dev uv run benchmarks/SaturnNPU/scripts/generate_npu_golden_tests.py \
  --output-dir benchmarks/SaturnNPU/golden_data/ --scale both
conda run -n merlin-dev uv run benchmarks/SaturnNPU/scripts/generate_mlir_golden_data.py \
  --output-dir benchmarks/SaturnNPU/golden_data/mlir_level/
conda run -n merlin-dev uv run benchmarks/SaturnNPU/scripts/export_golden_data.py \
  benchmarks/SaturnNPU/golden_data/ --formats numpy
```

## Quantization Status

SmolVLA uses **per-tensor FP8 E4M3 weight quantization** with power-of-two
(po2) scaling — no int8 fallback:

- **302 linears**: FP8 po2 weights (`f8E4M3FN`) — all components
- **1 linear**: unquantized (lm_head, intentionally skipped)

All linears, including Gemma expert (`hidden_dim=720`), are quantized to FP8.
The per-tensor po2 scheme has no `in_features % block_size` constraint (unlike
MX block quantization). Scale = `2^floor(log2(amax / 256))`, which fits in
the NPU's E8M0 scale registers.

After global-optimization + the `fold-fp8-scales-around-contractions` pass,
kernel writers see:
1. `quantized_matmul_fp8`: fp8 activation × bf16 weight → bf16 accum (446 instances, 82% compute)
2. `linalg.batch_matmul`: bf16 × bf16 → f32 accum (46 instances, SigLIP attention only)
3. `iree_linalg_ext.attention`: fused SDPA (36 instances)
4. All elementwise/softmax/norm: bf16

## Kernel Developer Walkthrough

### 1. Pick your kernel

Run Step 2. The Pareto output shows what to implement first:
```
#1  quantized_matmul_fp8     446 instances   82.2%  (fp8 act, bf16 weight, bf16 accum)
#2  fused_attention           36 instances   16.1%
#3  batch_matmul_bf16         46 instances    0.6%  (SigLIP attention only)
```

Matmul inputs are fp8 (activation) with bf16 accumulation.
All vector ops (softmax, norm, silu, elementwise) are bf16.
No int8 operations remain in the model.

### 2. See the MLIR

Open `kernels/<type>/`. Each shape variant is a standalone `.mlir` file with
**fully resolved affine maps** (no `#mapN` references — extracted using MLIR Python bindings).

### 3. See how it fits in a layer

Open `LAYER_DECOMPOSITION_TRACE.md` — shows how each PyTorch layer
(SiglipAttention, GemmaMLP, etc.) decomposes into MLIR ops at input and
global-opt levels.

### 4. Get golden data

`golden_data/small/<layer>/operators/NN_<op>/` has input/output `.pt` and `.npy` files.
Composition verified: chaining all operator outputs = layer output.

### 5. Compile an op yourself

```python
import iree.compiler as compiler
import iree.runtime as runtime
vmfb = compiler.compile_str(open("kernels/silu/variant_0_....mlir").read(),
                            target_backends=["llvm-cpu"])
# No Merlin build needed — just iree.compiler + iree.runtime
```

## MLIR Reference

| File | Level | Use for |
|------|-------|---------|
| `smolVLA.q.fp8po2.mlir` | Torch-MLIR | PyTorch op structure |
| `module.1.input.mlir` | Linalg/Input | Full decomposition with named ops |
| `module.4.global-optimization.mlir` | Global-Opt | **Implement against this** |

All from the **spacemit_x60** target (vanilla IREE, no accelerator plugins).
Weights are FP8 E4M3 with per-tensor po2 scaling (no int8 fallback).

## Scripts

| Script | Purpose |
|--------|---------|
| `analyze_npu_graph.py` | Multi-level analysis, Pareto, composite patterns |
| `trace_layer_decomposition.py` | MLIR-bindings per-layer trace |
| `plot_npu_coverage.py` | Pareto + layer decomposition plots |
| `plot_sankey.py` | Interactive Sankey diagram (plotly HTML) |
| `extract_all_kernel_variants.py` | All shape variants + fused patterns |
| `generate_npu_golden_tests.py` | PyTorch-level golden data |
| `generate_mlir_golden_data.py` | MLIR-level golden data via IREE AOT |
| `export_golden_data.py` | Export to .npy / .bin |
