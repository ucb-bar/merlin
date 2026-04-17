# 2026-04-06: Saturn OPU — Int8 QDQ Model Compilation & VOPACC Coverage

> **Repro pin:** merlin@[`57529095`](https://github.com/ucb-bar/merlin/commit/57529095a81ea8df1258170c961da02f73423858) · iree_bar@[`b05497ea75`](https://github.com/ucb-bar/iree_bar/commit/b05497ea75)
> **Status:** Active

Related entries:

- [2026-03-13 RISC-V MMT4D ukernel workstream](2026-03-13-riscv-mmt4d-ukernel-workstream.md) — earlier "TODO: Evaluate K0=128" thread that anticipated narrow-M problems.
- [2026-04-14 f32-reduction hang findings](2026-04-14-f32-reduction-hang-findings.md) — separate transformer-only hang class on the same FireSim build.

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

## 2026-04-09 — MLP narrow-M FireSim hang

### What we observed

DroNet OPU on FireSim ran successfully (32.5M cycles per inference, 6.13× over
the RVV baseline at 199.8M cycles). MLP OPU compiled with the same toolchain
hung indefinitely on the **first** warmup invocation. The serial log stopped
after `Warmup (2 iterations)...` and never recovered after 12+ hours.

The original "100% MLP OPU coverage" claim above was based purely on **static
analysis** of the linked binary (counting `.insn r 87 ...` opcodes). MLP had
never actually been executed end-to-end on FireSim until the new
`bench_model_*` runner was wired up — at which point the runtime bug surfaced.

### Why MLP is special

All three MLP matmuls are **vecmat** (1D × 2D → 1D), not matvec (2D × 2D → 2D):

| Dispatch | Shape | After Torch lowering |
|----------|-------|----------------------|
| `dispatch_1` | `[1,10]` × `[32,10]` → `[1,32]` | `tensor<10>` × `tensor<32x10>` → `tensor<32>` |
| `dispatch_2` | `[1,32]` × `[32,32]` → `[1,32]` | `tensor<32>` × `tensor<32x32>` → `tensor<32>` |
| `dispatch_3` | `[1,32]` × `[2,32]` → `[1,2]`   | `tensor<32>` × `tensor<2x32>` → `tensor<2>` |

The LHS is rank-1 — there is **no M dimension** in the linalg encoding (the
indexing maps are `(d0, d1) -> (d1)`, `(d0, d1) -> (d0, d1)`,
`(d0, d1) -> (d0)`). When `getEncodingContractionLikeDims` (in
`Codegen/Dialect/Codegen/Utils/Utils.cpp`) processes this, `mDim.operandIdx`
is `std::nullopt`. `getEncodingInfoForMatmul` then skips the M dimension
entirely; `getEncodingInfoForMatmul` plus the dispatch builder ends up
producing a packed mmt4d with **M0=1, N0=16, K0=1** for the LHS panel
(`tensor<1x10x1x1xi8>` for dispatch_1).

DroNet does not hit this because im2col converts every conv into a matmul
with M ≥ 16. Its dispatches use M0=16 cleanly.

### The actual bug — `iree_uk_mmt4d_opu_full_loop`, not the tile function

We initially suspected the per-M0 tile dispatcher
(`iree_uk_mmt4d_tile_s8s8s32_NxXXx1_riscv_64_xopu`) and PATH B inside its
generic backing function. That was a red herring. **MLP never reaches the
tile dispatcher.**

The actual code path is the early handler `iree_uk_mmt4d_early_riscv_64_xopu`
at `runtime/src/iree/builtins/ukernel/arch/riscv_64/mmt4d_riscv_64_xopu.c:304`,
which delegates to `iree_uk_mmt4d_opu_full_loop` (line 116). That function
processes the entire M×N tile structure itself with its own 2×2 sub-tiling
across `m0`/`m1`/`m2`/`m3` matrix registers. The early handler intercepts
**before** the standard mmt4d tile dispatcher runs, so all narrow-M mitigations
in the tile-side PATH B are unreachable.

For narrow-M cases (M0 < HW=16) the full-loop function had two compounding
problems in its inner K loop (lines 222–268):

1. The outer `vsetvli zero, %0, e8, m1, ta, ma : : "r"(HW)` set vl=16. The
   inner `vle8.v v16, (lhs)` therefore loaded **16 bytes** for the LHS,
   when the encoding only has `m_hw0` valid bytes (1 for vecmat). It read
   15 bytes of garbage past the end of the LHS panel.
2. Even though it then computed `m_hw0 = min(M0 - m_sub, HW)` correctly and
   wrote out only `m_hw0` rows via `OPU_STORE_SUBTILE_2D`, the OPU outer
   product had already accumulated the wrong rows because the LHS register
   v16 contained 15 garbage lanes instead of zeros. This produced wrong
   outputs and likely stalled the OPU state machine on real hardware.

### Why naive `asm volatile vsetvli` inside the loop doesn't fix it

The seemingly obvious fix — set `vl = m_hw0` for the LHS load and
`vl = HW` for the RHS load — was already present in the per-M0 tile path
(`mmt4d_riscv_64_xopu.c` PATH B, ~line 631):

```c
asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(ml));
asm volatile("vle8.v v4, (%0)" : : "r"(&lhs_ptr[k * M0]) : "memory");
asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(vl));
asm volatile("vle8.v v5, (%0)" : : "r"(&sub_rhs[k * N0]) : "memory");
asm volatile(".insn r 0x57, 0x2, 0x51, x0, x5, x4" : : : "memory");
```

**LLVM's `RISCVInsertVSETVLI` pass strips standalone `asm volatile vsetvli`
instructions.** It inspects `vle8.v` / vector intrinsics, computes the
required vl from the operand types and live ranges, and inserts its own
`vsetvli` — usually hoisted out of the loop. The standalone hand-written
`vsetvli` survives in the IR but is treated as a no-op for vl tracking,
so both `vle8.v`s end up using `vl = HW = 16`. This is the same way the
standalone vsetvli's get DCE'd in `vmt4d_riscv_64_xopu.c` PATH B too.

Disassembling the bad MLP build confirms it:

```asm
.LBB6_2:                              ; (broken)
    vle8.v v16, (a3)                  ; LHS load — vl=16 (only 1 byte valid)
    vle8.v v17, (a1)                  ; RHS load — vl=16
    .insn r 87, 2, 81, zero, a7, a6   ; VOPACC (consumes garbage from v16)
    addi a3, a3, 1                    ; LHS += M0=1
    addi s1, s1, -16
    addi a1, a1, 16                   ; RHS += N0=16
    bnez s1, .LBB6_2
```

Only **one** `vsetvli` is left (and it's outside the loop, with vl=16).

### The fix

Two parts in `mmt4d_riscv_64_xopu.c`:

1. **Pre-zero `v16` and `v19` once** with `vmv.v.i v_, 0` while vl=HW. The
   tail-undisturbed loads inside the loop will preserve those zero lanes.
2. **Combine vsetvli + vle + vsetvli + vle + vsetvli + VOPACC into a single
   `asm volatile` block per K iteration.** Inside one asm string, LLVM treats
   the whole sequence as opaque and emits the instructions verbatim — the
   inner-loop `vsetvli`s survive.

Resulting inner loop in `iree_uk_mmt4d_opu_full_loop`:

```c
asm volatile(
    "vsetvli zero, %2, e8, m1, tu, ma\n\t"   // vl = m_hw0 (e.g. 1)
    "vle8.v v16, (%0)\n\t"                    // load m_hw0 bytes; lanes [m_hw0..15] preserved
    "vsetvli zero, %3, e8, m1, tu, ma\n\t"   // vl = n_hw0 (e.g. 16)
    "vle8.v v17, (%1)\n\t"                    // load n_hw0 RHS bytes
    "vsetvli zero, %4, e8, m1, ta, ma\n\t"   // vl = HW for VOPACC
    ".insn r 0x57, 0x2, 0x51, x0, x17, x16\n\t"
    :
    : "r"(lhs_kk + k0 * M0), "r"(rhs_kk0 + k0 * N0),
      "r"((size_t)m_hw0), "r"((size_t)n_hw0), "r"((size_t)HW)
    : "memory");
```

Symmetric fixes apply to the `n_hw1 > 0` (narrow-N right-half) and
`m_hw1 > 0` (narrow-M lower-half) branches for completeness — those branches
only fire for cases like M0=17..31 / N0=17..31 which we don't actually
encounter today, but the same pattern would have hit them.

### Verification (post-fix MLP inner loop)

```asm
.LBB6_2:                              ; (fixed)
    vsetvli zero, a3, e8, m1, tu, ma  ; vl = m_hw0 = 1 (a3)
    vle8.v v16, (a5)                  ; load 1 byte (lanes [1..15] stay 0)
    vsetvli zero, a1, e8, m1, tu, ma  ; vl = 16 (a1)
    vle8.v v17, (a4)                  ; load 16 RHS bytes
    vsetvli zero, a1, e8, m1, ta, ma  ; vl = 16 for VOPACC
    .insn r 87, 2, 81, zero, a7, a6   ; VOPACC m0
    addi a5, a5, 1                    ; LHS += M0
    addi s1, s1, -16
    addi a4, a4, 16                   ; RHS += N0
    bnez s1, .LBB6_2
```

All three `vsetvli`s are now inside the loop. `RISCVInsertVSETVLI` left them
alone because they live inside an opaque `asm volatile` block.

### Debug instrumentation (kept in tree)

For future narrow-M debugging:

- `samples/SaturnOPU/simple_embedding_ukernel/model_benchmark.c` now prints
  `Warmup iter X enter / done` and `Bench iter X` so the FireSim serial log
  pinpoints which invocation of the model hung.
- `mmt4d_riscv_64_xopu.c` has `IREE_UK_DEBUG_OPU_NARROW_M_PRINT(...)` macros
  at the entry/exit of `iree_uk_mmt4d_opu_full_loop` and the per-M0 tile
  function, gated by `#ifdef IREE_UK_DEBUG_OPU_NARROW_M`. Define it via
  `target_compile_definitions` on the bare-metal MLP target to surface the
  M, N, K, M0, N0, K0 of every dispatch.

### Lessons

1. **Static binary analysis is not enough.** "100% OPU coverage" must mean
   "the model finishes on real hardware producing correct outputs", not
   "the binary contains OPU opcodes".
2. **`asm volatile vsetvli` is fragile.** LLVM's RISC-V backend treats it as
   a no-op for vl tracking. Any time you need a specific vl for a single
   load/store, fuse the `vsetvli` and the vector instruction into the same
   asm block, or stop touching vl yourself and use intrinsics.
3. **Watch for early-handler hijacks.** `iree_uk_mmt4d_early_riscv_64_xopu`
   intercepts before the per-M0 tile dispatcher runs. Fixes that target the
   tile dispatcher are dead code for any case the early handler claims.
