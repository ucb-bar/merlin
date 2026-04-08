# SaturnNPU — SmolVLA Kernel Decomposition & Test Framework

Scripts and tooling for analyzing the SmolVLA model's compute graph across
MLIR compilation levels and generating golden test data for NPU kernel
development.

## Quick Start

From the Merlin repo root, with `merlin-dev` conda environment:

```bash
# 0. Compile (if not done) — vanilla target, no accelerator plugins
conda run -n merlin-dev uv run tools/merlin.py compile \
  models/smolVLA/smolVLA.q.fp8.mlir \
  --target spacemit_x60 --quantized \
  --compile-to global-optimization --dump-phases

# 1. Strip weight blobs
conda run -n merlin-dev uv run tools/strip_mlir_weights.py \
  build/compiled_models/smolVLA/spacemit_x60_RVV_smolVLA.q.fp8/ --in-place

# 2. Run analysis
conda run -n merlin-dev uv run benchmarks/SaturnNPU/scripts/analyze_npu_graph.py \
  --torch-mlir   build/compiled_models/smolVLA/spacemit_x60_RVV_smolVLA.q.fp8/smolVLA.q.fp8.mlir \
  --linalg-input build/compiled_models/smolVLA/spacemit_x60_RVV_smolVLA.q.fp8/phases/module.1.input.mlir \
  --global-opt   build/compiled_models/smolVLA/spacemit_x60_RVV_smolVLA.q.fp8/phases/module.4.global-optimization.mlir \
  --output-dir   benchmarks/SaturnNPU/ --assert-counts

# 3. Layer decomposition trace (uses MLIR Python bindings)
conda run -n merlin-dev uv run benchmarks/SaturnNPU/scripts/trace_layer_decomposition.py \
  --linalg-input build/compiled_models/smolVLA/spacemit_x60_RVV_smolVLA.q.fp8/phases/module.1.input.mlir \
  --global-opt   build/compiled_models/smolVLA/spacemit_x60_RVV_smolVLA.q.fp8/phases/module.4.global-optimization.mlir

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

SmolVLA uses **MX fp8 weight quantization** (block_size=32) with int8 fallback:

- **236 linears**: MX fp8 weights (`f8E4M3FN`) — SigLIP + Gemma main
- **66 linears**: int8 weights — Gemma expert (hidden_dim=720, not divisible by 32)
- **1 linear**: unquantized (lm_head)

The int8 fallback exists because TorchAO's MX format requires `in_features % 32 == 0`,
and SmolVLA_base's Gemma expert has `hidden_dim=720` (`720 % 32 = 16`).

Kernel writers need **two matmul types**:
1. `quantized_matmul_fp8`: bf16 activation × f8E4M3FN weight + block scaling (82% of compute)
2. `linalg.matmul i8`: i8 activation × i8 weight → i32 accumulator (0.8% of compute)

## Kernel Developer Walkthrough

### 1. Pick your kernel

Run Step 2. The Pareto output shows what to implement first:
```
#1  quantized_matmul_fp8     379 instances   82.2%
#2  fused_attention           36 instances   16.1%
#3  matmul_i8                 67 instances    0.8%
#4  batch_matmul_bf16         46 instances    0.6%
```

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
| `smolVLA.q.fp8.mlir` | Torch-MLIR | PyTorch op structure |
| `module.1.input.mlir` | Linalg/Input | Full decomposition with named ops |
| `module.4.global-optimization.mlir` | Global-Opt | **Implement against this** |

All from the **spacemit_x60** target (vanilla IREE, no accelerator plugins).

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
