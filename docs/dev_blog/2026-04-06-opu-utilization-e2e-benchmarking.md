# Saturn OPU: Int8 QDQ Model Compilation & VOPACC Coverage

**Date:** 2026-04-06 → 2026-04-08

## Summary

End-to-end int8 QDQ model compilation with Saturn OPU VOPACC acceleration.
Four models compiled with 100% matmul OPU coverage (74-87% compute-weighted).
Includes fused matmul+QDQ ukernel, im2col for CNNs, multi-head collapse
preprocessing for LLMs, and per-dispatch coverage analysis tooling.

## Final OPU Coverage

| Model | hw Target | Matmuls | Compute % | Key Technique |
|-------|-----------|---------|-----------|---------------|
| MLP (3 FC) | OPU | 3/3 (100%) | 74% | Standard encoding pipeline |
| DroNet (CNN) | OPU_IM2COL | 10/10 (100%) | 82% | Im2col conv→matmul |
| YOLOv8-nano | OPU_IM2COL | 64/64 (100%) | 87% | Im2col + no opt-level |
| TinyLlama 1.1B | OPU_LLM | 135/135 (100%) | 86% | Multi-N collapse pass |

Non-OPU dispatches are elementwise (dequant/requant), softmax, reductions, and
transposes — all use RVV vector instructions. These inherently cannot use VOPACC.

## Compilation Targets

Three targets in `saturn_opu.yaml` for different model architectures:

**OPU** (`+xopu`, O3) — Dense matmul models. Standard data-tiling pipeline:
encoding → pack → mmt4d → `iree_uk_opu_matmul` ukernel.

**OPU_LLM** (`+xopu`, no opt-level) — LLM models with multi-head attention
projections. Runs `CollapseMultiNContractions` preprocessing to flatten
`[K,H,D] → [K,H*D]` before encoding. No O3: IREE's SinkReshapes at O3
re-fuses the collapse, recreating multi-N contractions that encoding rejects.

**OPU_IM2COL** (no `+xopu`, no opt-level) — CNN models. Im2col converts
convolutions to matmuls. No `+xopu`: OPU encoding materializer doesn't handle
im2col dispatch shapes. OPU VOPACC activates at runtime via `iree_uk_mmt4d`
hardware detection (`cpu_data[0] & XOPU_BIT`). No opt-level: O2/O3 crash with
im2col on RISC-V (pre-existing IREE bug).

## Compiler Changes

### Fused OPU+QDQ Ukernel

QDQ models wrap every matmul in dequant/requant ops. Without fusion, the i32
accumulator is written to memory and read back. The fused ukernel keeps it in
the OPU matrix register:

```
matmul_i8 → i32 (in m0)
  → vfcvt.f.x.v          (i32 → f32)
  → vfmadd.vf(scale,bias) (dequant + bias, one FMA)
  → vfmul.vf(inv_scale)   (requant)
  → fsrmi 0 + vfcvt.x.f.v (hardware RNE roundeven)
  → vnsrl × 2             (i32 → i16 → i8)
  → vmax(0)               (ReLU)
  → vse8.v                (store i8)
```

~13 instructions per row vs ~165 for split path. Hardware RNE via `fsrmi 0`
replaces ~40-instruction software roundeven.

**Runtime:** `mmt4d_riscv_64_xopu.c` — `OPU_STORE_SUBTILE_QDQ_I8` macro +
`iree_uk_opu_matmul_qdq()`.

**Compiler:** `CPULowerToUKernels.cpp` — `FusedOPUMatmulQDQPattern` matches
`mmt4d_ukernel → generic(dequant+bias+requant)` chain, guarded by `+xopu`.

**Prerequisite:** `ArithOps.cpp` fold: `divf(mulf(x,c),c) → x` eliminates
redundant scale cancellation in QDQ models.

### Batch Matmul Fix

`lowerContractionOpWithEncoding` (CPUEncodingExternalModels.cpp:338):
`isNarrowNResult` caused `std::swap(newLhs, newRhs)` for batch attention
matmuls, breaking `batch_mmt4d` dimension verification when M != N.
Fix: disable transpose for batch matmuls.

### Multi-N Collapse Preprocessing

**New pass:** `CollapseMultiNContractions` in `Preprocessing/Common/`.

TinyLlama Q/K/V projections have multi-head weights `[K,H,D]` producing
multi-N contractions (`n.size() == 2`) that IREE's encoding system rejects.
The pass collapses them:

```
linalg.generic(out[H,M,D] = Σ_k lhs[M,K] × rhs[K,H,D])
  → collapse_shape rhs [K,H*D]
  → standard 2D matmul [M,K] × [K,N] → [M,N]
  → expand_shape + transpose → [H,M,D]
```

The 2D matmul flows through encoding → mmt4d → OPU ukernel naturally. The
expand+transpose becomes a separate cheap dispatch.

### Multi-N Vector Size Guard

`getMatmulOPUVectorSizes` (KernelDispatch.cpp) used K0=128 for multi-N
contractions with identity encoding, creating 8192-byte vectors on V128.
Fix: skip multi-N (`n.size() != 1`) so they fall through to generic RVV.

### Resilient OPU Materializer

`lowerOp` now has a full fallback chain: OPU ukernel → CPU mmt4d →
`dropEncodingAndCloneOp` → `dropAllEncodingsAndClone` (via `mlir::clone`).
Never returns nullptr.

### Quantized Conv Channels-Last

`ConvertConvToChannelsLast.cpp`: Added `Conv2DNchwFchwQOp → Conv2DNhwcHwcfQOp`
conversion using DPS operand access for quantized convolutions with scalar
zero-point operands.

## Code Organization

All OPU-specific code is guarded by `hasFeature(config, "+xopu")` and marked
with `===== Saturn OPU (+xopu) BEGIN/END =====` comment blocks.

| File | Lines | Type | Guard |
|------|-------|------|-------|
| `CPUEncodingExternalModels.cpp` | ~200 | OPU resolver + materializer | External interface structs |
| `CPULowerToUKernels.cpp` | ~175 | Fused QDQ pattern | `+xopu` feature check |
| `KernelDispatch.cpp` | ~16 | Vector sizes + UKernel guard | `+xopu` + generic |
| `mmt4d_riscv_64_xopu.c` | ~143 | Runtime kernels | OPU-only file |
| `ConvertConvToChannelsLast.cpp` | ~56 | Quantized conv (generic) | All targets |
| `CollapseMultiNContractions.cpp` | ~260 | Multi-N preprocessing | All targets |
| `ArithOps.cpp` (llvm-project) | ~10 | divf/mulf fold (generic) | All targets |
| `mmt4d.c` | ~49 | Cycle instrumentation | `#ifdef` gated |

## Analysis Tooling

- `benchmarks/SaturnOPU/analyze_opu_coverage.py` — Parses compiled assembly,
  classifies dispatches, checks for VOPACC / mmt4d ukernel calls, reports
  per-dispatch OPU status and compute-weighted coverage.
- `benchmarks/SaturnOPU/plot_opu_coverage.py` — Paper-quality charts.

## Isolated Matmul Utilization (FireSim V128-D64)

| Size | Ops/Cycle | Utilization | Notes |
|------|-----------|-------------|-------|
| 64×64 | 3.95 | 3% | Overhead-dominated |
| 256×256 | 29.49 | 23% | |
| 1024×1024 | 57.94 | 45% | |
| 2048×2048 | 65.62 | 51% | Approaching peak (128 ops/cycle) |

Kernel-only cycles (via `IREE_UK_BENCHMARK_CYCLES`) confirm overhead is in VM
dispatch, not the OPU kernel.

## Known Limitations

- **O3 + multi-N collapse:** IREE's `SinkReshapes` at O3 re-fuses
  `collapse_shape` back into the matmul. LLM models use `OPU_LLM` (no O3).
- **Im2col + `+xopu`:** OPU encoding materializer crashes on im2col dispatch
  shapes. CNN models use `OPU_IM2COL` (no `+xopu`, runtime OPU detection).
- **Im2col + opt-level:** Pre-existing IREE crash with O2/O3 on RISC-V im2col.
- **FP8:** Compiles but not hardware-accelerated (see mmt4d workstream blog).
