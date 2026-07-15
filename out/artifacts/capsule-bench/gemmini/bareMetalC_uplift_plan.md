# bareMetalC uplift plan (capsule_bench_v0)

## Principle

`gemmini-rocc-tests/bareMetalC/` (57 specimens at
`tmp/dialects/gemmini/software/gemmini-rocc-tests/bareMetalC/`) is used **only as a behavioral &
coverage corpus**. For each selected specimen we:

1. read what Gemmini operation / shape / dtype / layout / instruction classes it exercises,
2. author an **equivalent compiler-input capsule** (interface MLIR + capsule.yaml),
3. author a numpy/`Tensor`-engine golden + an expected-instruction-coverage spec,
4. compile it through the backend-under-test and run the full capsule verification.

**Strict rule (enforced by the integrity scan + the ABI boundary):** the uplifted test is an *input
to the compiler*, never a copied or called kernel. We do **not** copy bareMetalC C, call its kernels,
or call any high-level Gemmini C library function as the generated implementation. The device code is
MLIR-lowered RoCC only.

## Specimen → capsule mapping (selected v0 set)

| bareMetalC specimen | op class exercised | uplifted capsule | notes |
|---|---|---|---|
| `matmul_ws.c` | single-tile WS matmul i8→i32 | `isa/A2_single_tile_matmul` | 16×16×16 |
| `tiled_matmul_ws.c` | tiled WS matmul, K-accumulation | `isa/A3_k_accumulation` | K=32 ⇒ accumulate-onto PRELOAD |
| `transpose_scale.c` | acc_scale (f32) → i8 readout | `isa/A4_acc_scale_i8` | scale 0.0625, saturating i8 |
| (relu activation, ubiquitous) | relu epilogue | `isa/A5_relu_epilogue` | activation bits in CONFIG_ST |
| `matmul_ws.c` (resident reuse) | one resident W, multiple matmuls | `isa/A6_resident_reuse` | 2 matmuls, no weight reload |
| `padded.c` | non-16-multiple dims / edge | `isa/A7_edge_padding` | 20×24×12, zero-pad tiles |
| `mvin_mvout.c` | MVIN→scratchpad→MVOUT, no compute | `isa/A1_mvin_mvout` | movement-only; forbids compute |
| `conv.c` | conv2d (im2col) i8 | `layers/B3_conv2d_im2col_i8` | 1×8×8×4, 3×3, Co=8 |
| `conv_with_pool.c` (conv core) | conv2d + relu | `layers/B4_conv2d_relu_i8` | conv core + relu epilogue (no pool) |
| (quantized linear, NN MLPs) | linear + requant → i8 | `layers/B0_quantized_linear_i8` | nn.Linear analog |
| (linear + relu) | linear + relu | `layers/B1_linear_relu_i8` | |
| (linear + requant + relu) | linear + acc_scale + relu → i8 | `layers/B2_linear_acc_scale_relu_i8` | |

Model-slice capsules (`model_slices/C0..C6`) are derived from PyTorch MLP/attention slices via
`model_slice_export.py` (Q/K/V projections, QK^T with K provided pre-transposed as the resident
weight leaf, PV) — also single weight-stationary matmuls, no softmax.

## Coverage rationale

The selected specimens span every instruction class the backend emits (CONFIG_EX/LD/ST, MVIN, MVOUT,
PRELOAD, COMPUTE_PRELOADED, FLUSH, FENCE) and the K-accumulation idiom (accumulate-onto PRELOAD).
Specimens exercising features the backend does not model (output-stationary dataflow, LOOP_WS/
LOOP_CONV hardware loops, igelu/layernorm/softmax activations, depthwise/dilated/transposed conv,
residual-add, pooling, transpose) are **not** uplifted in v0 and are listed as explicit
"not covered" rows in `isa_coverage_report.md` — they are out of the v0 scope, not silently omitted.
