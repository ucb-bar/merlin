# 2026-03-13: RISC-V MMT4D Ukernel Workstream

## Context and Goal

This workstream focused on moving the RISC-V matmul effort onto the `mmt4d`
ukernel path instead of the vector-contract custom-kernel path.

The concrete goals were:

- keep regular RVV int8 on an efficient `mmt4d` path
- enable SpacemiT `+xsmtvdot` int8 through `mmt4d`
- enable Saturn OPU `+xopu` int8 through `mmt4d` with high-K tiling
- make FP8 compilation work for the same lowering flow
- validate the generated assembly with `--dump-artifacts`, not just the IR

The main design constraint was to follow IREE's existing `mmt4d` structure and
tile-selection conventions as closely as possible, while still exposing target-
specific tile families where the hardware clearly wants them.

## Implementation Changes

### 1. RISC-V ukernel-side `mmt4d` work

The i8 path was moved onto explicit RISC-V ukernel implementations in:

- `third_party/iree_bar/runtime/src/iree/builtins/ukernel/arch/riscv_64/mmt4d_riscv_64_v_i8.c`
- `third_party/iree_bar/runtime/src/iree/builtins/ukernel/arch/riscv_64/mmt4d_riscv_64_tiles.inl`
- `third_party/iree_bar/runtime/src/iree/builtins/ukernel/arch/riscv_64/query_tile_sizes_riscv_64_entry_point.c`

Implemented tile families:

- SpacemiT `+xsmtvdot`: `4x4x8`
- OPU `+xopu` int8: `16x16x128` with narrow-M truncations

The OPU int8 path keeps the high-K tile because that is where the hardware
extension actually makes sense. This matches the intent from the Saturn sample
kernels and avoids falling back to a low-K generic RVV-style shape.

### 2. Compiler tile selection and lowering configuration

Compiler-side tile selection was updated so encoding/materialization and
lowering strategy agree on the same tile families:

- `third_party/iree_bar/compiler/src/iree/compiler/Codegen/ExternalInterfaces/CPUEncodingExternalModels.cpp`
- `third_party/iree_bar/compiler/src/iree/compiler/Codegen/LLVMCPU/KernelDispatch.cpp`

Changes made:

- added `+xsmtvdot` FP8/int8 tile families using `4x4x8`
- added `+xopu` int8 tile family using `16x16x128`
- added `+xopu` FP8 tile family using `16x16x8`

The OPU FP8 case intentionally does not use `K=128` yet. That tile shape
caused oversized vector contracts during lowering, so for now the compiler uses
a smaller `K=8` shape to keep the path compiling while native FP8 OPU lowering
is still missing.

### 3. FP8 legalization fix

The first FP8 blocker was not OPU- or SpacemiT-specific. The CPU backend left
`arith.extf` illegal for cases like:

```mlir
vector<4x8xf8E4M3FN> -> vector<4x8xf16>
```

The fix was made in:

- `third_party/iree_bar/compiler/src/iree/compiler/Codegen/Common/ConvertUnsupportedFloatArithPass.cpp`

The pass already knew how to emulate small-float `extf` to `f32`. It now
reconstructs `f32` first and then emits a final cast to the requested wider
destination type such as `f16`.

Regression coverage was added in:

- `third_party/iree_bar/compiler/src/iree/compiler/Codegen/Common/test/convert_unsupported_float_arith.mlir`

### 4. Lowering-strategy tests

Test coverage was extended so target-specific lowering selection is visible in
MLIR before looking at assembly:

- `third_party/iree_bar/compiler/src/iree/compiler/Codegen/LLVMCPU/test/select_riscv_spacemit_lowering_strategy.mlir`
- `third_party/iree_bar/compiler/src/iree/compiler/Codegen/LLVMCPU/test/select_riscv_opu_lowering_strategy.mlir`

The new OPU test now covers both:

- int8 -> i32
- fp8 -> f16

### 5. Benchmark / artifact scripts

Artifact-driven validation scripts were added or updated in Merlin:

- `benchmarks/SpacemiTX60/compile_matmul_xsmt_i8_ukernel_all.sh`
- `benchmarks/SpacemiTX60/compile_matmul_xsmt_fp8.sh`
- `benchmarks/SaturnOPU/compile_matmul_opu_i8_ukernel_all.sh`
- `benchmarks/SaturnOPU/compile_matmul_opu_fp8_ukernel_all.sh`

These scripts compile with:

- `--iree-llvmcpu-enable-ukernels=all`
- `--iree-llvmcpu-link-ukernel-bitcode=true`
- `--iree-opt-data-tiling=true`
- `--iree-dispatch-creation-data-tiling=true`
- `--dump-artifacts`

For i8, the scripts also verify the hot loop by checking the dumped `.s` file
for the expected target instructions.

## What Worked

### 1. XSMT int8 `mmt4d` path

The generated hot loop for SpacemiT int8 is structurally good:

- two vector loads
- one `vmadot` encoded as `.insn`
- pointer increments
- loop branch

Relevant assembly:

- `/scratch2/agustin/merlin/build/compiled_models/SpacemiT/spacemit_x60_RVV_matmul_i8_2048/files/module_matmul_i8_2048_linked_embedded_elf_riscv_64.s`

The hot loop is:

```asm
.LBB0_3:
    vle8.v  v0, (a1)
    vle8.v  v1, (a2)
    .insn r 43, 3, 113, t3, zero, ra
    addi    a2, a2, 32
    addi    a0, a0, 1
    addi    a1, a1, 32
    bnez    a0, .LBB0_3
```

This is the right kind of loop to optimize: no obvious in-loop spills, no
extra scalar unpacking, and the target instruction is in the inner loop.

### 2. OPU int8 `mmt4d` path

The OPU int8 loop is also structurally good:

- strided vector load for A
- strided vector load for B
- one `vopacc` encoded as `.insn`
- pointer increments
- loop branch

Relevant assembly:

- `/scratch2/agustin/merlin/build/compiled_models/SpacemiT/saturn_opu_OPU_matmul_i8_2048/files/module_matmul_i8_2048_linked_embedded_elf_riscv_64.s`

Hot loop:

```asm
.LBB0_4:
    vlse8.v v16, (a4), a0
    vlse8.v v18, (a1), a0
    .insn r 87, 2, 81, zero, s2, a6
    addi    a1, a1, 1
    addi    a3, a3, 1
    addi    a4, a4, 1
    bnez    a3, .LBB0_4
```

This is the expected shape for the OPU int8 path and is much closer to the
hardware samples than a generic vector-contract lowering.

### 3. FP8 compilation

After the `arith.extf` legalization fix, both FP8 targets compile:

- SpacemiT FP8 lowers to `linalg.mmt4d` with `4x4x8`
- OPU FP8 lowers to `linalg.mmt4d` with `16x16x8`

Configured dispatch artifacts:

- `/scratch2/agustin/merlin/build/compiled_models/SpacemiT/spacemit_x60_RVV_matmul_fp8_2048/configs/configured_module_matmul_fp8_2048_dispatch_0.mlir`
- `/scratch2/agustin/merlin/build/compiled_models/SpacemiT/saturn_opu_OPU_matmul_fp8_2048/configs/configured_module_matmul_fp8_2048_dispatch_0.mlir`

Both now show `linalg.mmt4d` instead of failing during LLVMCPU lowering.

## What Did Not Work (and Why)

### 1. FP8 is not efficient yet

FP8 now compiles, but the assembly is not good enough yet.

The SpacemiT FP8 assembly contains:

- repeated temporary vector stores/loads to stack
- scalar extraction patterns
- many calls to `__truncsfhf2`
- no `vmadot` FP8-like inner-product instruction in the hot loop

Relevant assembly:

- `/scratch2/agustin/merlin/build/compiled_models/SpacemiT/spacemit_x60_RVV_matmul_fp8_2048/files/module_matmul_fp8_2048_linked_embedded_elf_riscv_64.s`

This means the path is compiling through generic legalized vector/scalar code,
not a target-native FP8 kernel.

The same issue exists for OPU FP8:

- it lowers through `mmt4d`
- but the dumped `.s` does not show `vopacc`-style FP8 hardware usage
- it contains extensive software conversion/truncation traffic

So the current FP8 result is:

- compiler path fixed
- code generation path still not hardware-accelerated

### 2. OPU FP8 high-K tile was too aggressive

Trying to force OPU FP8 into `16x16x128` immediately failed legality checks due
to enormous intermediate vector contracts.

That was reduced to `K=8` as a temporary compiler-side compromise. The int8 OPU
path keeps `K=128`; the FP8 OPU path does not yet have the dedicated lowering
needed to support that shape efficiently.

## Debugging Notes

The debugging loop that worked best here was:

1. confirm the selected lowering config in `configured_module_*.mlir`
2. confirm that the op is really `linalg.mmt4d`
3. compile with `--dump-artifacts`
4. inspect the first hot loop in the dumped `.s`

This was important because simply seeing the right tile sizes in MLIR was not
enough. FP8 is the main example: the IR shape was acceptable, but the final
assembly made it obvious that the path was still software-heavy.

The key compiler bug found during this workstream was the illegal
`arith.extf` from FP8 vectors to `f16`, which was only visible once FP8 matmul
started reaching the LLVMCPU lowering pipeline.

## Test Coverage and Exact Commands

Build:

```bash
conda run -n merlin-dev uv run tools/merlin.py build \
  --profile full-plugin \
  --config release \
  --cmake-target iree-compile
```

Unsupported-float regression:

```bash
cd third_party/iree_bar

/scratch2/agustin/merlin/build/host-merlin-release/tools/iree-opt \
  --split-input-file \
  --pass-pipeline="builtin.module(func.func(iree-convert-unsupported-float-arith))" \
  compiler/src/iree/compiler/Codegen/Common/test/convert_unsupported_float_arith.mlir \
  | /scratch2/agustin/merlin/build/host-merlin-release/llvm-project/bin/FileCheck \
      compiler/src/iree/compiler/Codegen/Common/test/convert_unsupported_float_arith.mlir
```

Lowering-strategy tests:

```bash
cd third_party/iree_bar

/scratch2/agustin/merlin/build/host-merlin-release/tools/iree-opt \
  --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' \
  --split-input-file \
  compiler/src/iree/compiler/Codegen/LLVMCPU/test/select_riscv_spacemit_lowering_strategy.mlir \
  | /scratch2/agustin/merlin/build/host-merlin-release/llvm-project/bin/FileCheck \
      compiler/src/iree/compiler/Codegen/LLVMCPU/test/select_riscv_spacemit_lowering_strategy.mlir

/scratch2/agustin/merlin/build/host-merlin-release/tools/iree-opt \
  --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' \
  --split-input-file \
  compiler/src/iree/compiler/Codegen/LLVMCPU/test/select_riscv_opu_lowering_strategy.mlir \
  | /scratch2/agustin/merlin/build/host-merlin-release/llvm-project/bin/FileCheck \
      compiler/src/iree/compiler/Codegen/LLVMCPU/test/select_riscv_opu_lowering_strategy.mlir
```

Artifact-driven compiles:

```bash
benchmarks/SpacemiTX60/compile_matmul_xsmt_i8_ukernel_all.sh
benchmarks/SpacemiTX60/compile_matmul_xsmt_fp8.sh
benchmarks/SaturnOPU/compile_matmul_opu_i8_ukernel_all.sh
benchmarks/SaturnOPU/compile_matmul_opu_fp8_ukernel_all.sh
```

## Critical Bug Fix: vmadot VL Not Set (2026-03-18)

### Root cause

The vmadot kernel (`iree_uk_mmt4d_tile_s8s8s32_4x4x8_riscv_64_xsmtvdot_native`
in `mmt4d_riscv_64_v_i8.c`) triggered SIGILL on larger matrices (256x256+)
while working on small ones (32x64). Two bugs:

**Bug 1: VL=16 instead of VL=32.** The accumulator init used
`vsetvli zero, zero, e32, m2` which set VL=16 (for 16 x i32). When switching
to `e8, m1` for the input loads, the old code used `vsetvli zero, zero` which
keeps VL=16. But vmadot requires VL=32 (4x4x8 = 32 bytes per operand).
With VL=16, only half the data was loaded and the hardware trapped.

**Bug 2: LLVM bitcode pipeline strips vsetvli.** Even after fixing the source
to use `vsetvli zero, %0, e8, m1, ta, ma` with `%0=32`, LLVM's RISC-V backend
(in the bitcode link path) merges consecutive `vsetvli` instructions with the
same type config, converting our explicit VL=32 back to `vsetvli zero, zero`.

### Fix

Use `.word 0x0c02f057` (raw encoding of `vsetvli zero, t0, e8, m1, ta, ma`)
inside the inline asm block, preceded by `li t0, 32`. This encoding is opaque
to LLVM's bitcode optimizer, so it survives the bitcode pipeline intact.

The fixed hot loop (`mmt4d_riscv_64_v_i8.c`):

```c
for (int k = 0; k < params->K; ++k) {
    asm volatile(
        "li t0, 32\n\t"
        ".word 0x0c02f057\n\t"           // vsetvli zero, t0, e8, m1, ta, ma
        "vle8.v v0, (%0)\n\t"            // Load LHS (32 bytes)
        "vle8.v v4, (%1)\n\t"            // Load RHS (32 bytes)
        ".insn r 0x2b, 3, 0x71, v8, v0, v4\n\t"  // vmadot v8, v0, v4
        :
        : "r"(lhs_ptr), "r"(rhs_ptr)
        : "memory", "t0");
    lhs_ptr += 32;
    rhs_ptr += 32;
}
```

Key register assignments:
- `v8` accumulator (VRM2, even-aligned: v8-v9, holds 16 x i32 = 4x4 tile)
- `v0` LHS input (32 bytes = 4 rows x 8 cols of i8)
- `v4` RHS input (32 bytes, does NOT overlap with acc v8-v9)

**Critical build step**: after changing the kernel, BOTH the host compiler
(`iree-compile`) AND the cross-compiled runtime must be rebuilt:

```bash
# 1. Rebuild host ukernel bitcode + iree-compile
conda run -n merlin-dev uv run tools/merlin.py build \
  --profile full-plugin --config release \
  --cmake-target iree-compile

# 2. Rebuild cross-compiled runtime
conda run -n merlin-dev uv run tools/merlin.py build \
  --profile spacemit \
  --cmake-target iree-run-module
```

The host `iree-compile` embeds the ukernel bitcode at link time. If only the
spacemit runtime is rebuilt, the old bitcode remains embedded in `iree-compile`
and the VMFB still contains the broken kernel.

### Verified assembly output

After the fix, the VMFB assembly shows:

```asm
.LBB0_3:                              ; hot loop
    li      t0, 32                    ; AVL = 32
    .word   201519191                 ; vsetvli zero, t0, e8, m1, ta, ma
    vle8.v  v0, (a0)                  ; load LHS
    vle8.v  v4, (a2)                  ; load RHS
    .insn r 43, 3, 113, v8, v0, v4   ; vmadot v8, v0, v4
    addi    a0, a0, 32
    addi    a2, a2, 32
    ...
    bnez    ..., .LBB0_3
```

## On-Board Benchmark Results (2026-03-18)

Tested on SpacemiT X60 (8-core RISC-V, VLEN=256, Linux 6.1.15).
Board: `root@10.44.86.251`.

### Correctness

Both paths produce correct results for `matmul_q_i8_256.mlir`
(256x256 quantized matmul, input=1):

- RVV i8: `256x256xi32` = all 256. Correct.
- xsmtvdot i8: `256x256xi32` = all 256. Correct.

### Benchmark: 1024x1024 quantized matmul (i8xi8->i32)

Using `iree-benchmark-module` with multiple iterations:

| Path | Tile | Median | Speedup |
|------|------|-------:|--------:|
| RVV i8 (`vwmul.vx` + `vwadd.wv`) | 8x16x1 | **312 ms** | 1.0x |
| xsmtvdot i8 (`vmadot` NPU) | 4x4x8 | **34.8 ms** | **9.0x** |

```bash
# RVV i8
iree-benchmark-module --device=local-task \
  --module=bench_1024_rvv.vmfb \
  --function=matmul_i8_quantized \
  --input="1024x1024xi8=1" --input="1024x1024xi8=1" \
  --benchmark_repetitions=5

# xsmtvdot (NPU)
iree-benchmark-module --device=local-task \
  --module=bench_1024_xsmtvdot.vmfb \
  --function=matmul_i8_quantized \
  --input="1024x1024xi8=1" --input="1024x1024xi8=1" \
  --benchmark_repetitions=5
```

The vmadot IME instruction provides a **9x speedup** over standard RVV
widening multiply-accumulate for int8 quantized matmul.

### RVV i8 assembly (reference hot loop)

From `tests/e2e/SpacemiT/tmp/matmul_q_i8_rvv/`:

```asm
.LBB0_1:
    vsetvli  zero, zero, e8, mf2, ta, ma
    vle8.v   v24, (t1)              ; load RHS (16 i8 elements)
    lbu      a1, -3(a3)             ; load 8 scalar LHS bytes
    lbu      a0, -2(a3)
    ...
    vwmul.vx v25, v24, a1           ; widening mul: i8*i8 -> i16
    vwmul.vx v26, v24, a0
    ...
    vsetvli  zero, zero, e16, m1, tu, ma
    vwadd.wv v8, v8, v25            ; widening add: i16+i32 -> i32
    vwadd.wv v22, v22, v26
    ...
    bnez     a5, .LBB0_1
```

### Upstream patch

Patches stored at `patches/upstream/`:
- `riscv64-mmt4d-i8-rvv-kernel-only.patch` (606 lines) — just the RVV kernel
- `riscv64-mmt4d-i8-full.patch` (865 lines) — all files
- `README.md` — extraction guide

The RVV i8 kernel uses only `<riscv_vector.h>` intrinsics (no inline assembly,
no vendor extensions). Fully extractable from `mmt4d_riscv_64_v_i8.c` lines
1-201.

## FP8 vfmadot Progress (2026-03-18)

### Runtime side: complete

All runtime infrastructure for FP8 `f8E4M3FN x f8E4M3FN -> f16` is in place:

| File | Change |
|------|--------|
| `exported_bits.h` | Added `IREE_UK_FLAG_MMT4D_TYPE_F8E4M3F8E4M3F16 = 0x0B` |
| `common.h` | Added `IREE_UK_TYPE_FLOAT_8 = FLOAT_IEEE \| 3` |
| `mmt4d_internal.h` | Added `iree_uk_mmt4d_type_f8e4m3f8e4m3f16` enum + routing |
| `exported_bits.h` | Added `QUERY_TILE_SIZES_OPERATION_MATMUL_F8E4M3F8E4M3F16 = 0x0700` |
| `query_tile_sizes_riscv_64_entry_point.c` | Returns `{M=4, K=8, N=4}` for xsmtvdot |
| `mmt4d_riscv_64_tiles.inl` | Registered `f8e4m3, f8e4m3, f16, 4, 8, _xsmtvdot` |
| `mmt4d_riscv_64_v_i8.c` | `iree_uk_mmt4d_tile_f8e4m3f8e4m3f16_4x4x8_riscv_64_xsmtvdot_native()` |

The FP8 vfmadot kernel:
```c
// vfmadot: Opcode 0x2b, Funct3 0, Funct7 0x75 (OPFMMA).
// Accumulator: VR (single register, 16 x fp16 = 256 bits).
// Inputs: 32 bytes of f8E4M3FN packed as i8.
asm volatile(
    "li t0, 32\n\t"
    ".word 0x0c02f057\n\t"              // vsetvli zero, t0, e8, m1, ta, ma
    "vle8.v v0, (%0)\n\t"              // Load LHS (32 bytes f8)
    "vle8.v v4, (%1)\n\t"              // Load RHS (32 bytes f8)
    ".insn r 0x2b, 0, 0x75, v8, v0, v4\n\t"  // vfmadot v8, v0, v4
    :
    : "r"(lhs_ptr), "r"(rhs_ptr)
    : "memory", "t0");
```

Key differences from int8 vmadot:
- Accumulator: `e16, m1` (fp16) not `e32, m2` (i32)
- Load/store: `vle16.v`/`vse16.v` for accumulator
- Encoding: `funct7=0x75` (OPFMMA) and `funct3=0` (standard FP)

### Compiler side: blocking

### Compiler side: complete (2026-03-18)

Added FP8 ukernel routing in `CPULowerToUKernels.cpp`:

```cpp
// In the mmt4d type matching (line ~215):
} else if (isa<FloatType>(lhsElemType) &&
           lhsElemType.getIntOrFloatBitWidth() == 8 &&
           isa<FloatType>(rhsElemType) &&
           rhsElemType.getIntOrFloatBitWidth() == 8 &&
           outElemType.isF16()) {
  flags = IREE_UK_FLAG_MMT4D_TYPE_F8E4M3F8E4M3F16;

// In the query tile sizes (line ~507):
} else if (isa<FloatType>(lhs) && lhs.getIntOrFloatBitWidth() == 8 &&
           isa<FloatType>(rhs) && rhs.getIntOrFloatBitWidth() == 8 &&
           out.isF16()) {
  return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_F8E4M3F8E4M3F16;
```

### Assembly verification: vfmadot present

After rebuilding `iree-compile`, the FP8 matmul hot loop now contains:

```asm
.LBB0_2:
    li      t0, 32                    ; AVL = 32
    .word   201519191                 ; vsetvli zero, t0, e8, m1, ta, ma
    vle8.v  v0, (a0)                  ; load LHS (32 bytes f8E4M3FN)
    vle8.v  v4, (a1)                  ; load RHS (32 bytes f8E4M3FN)
    .insn r 43, 0, 117, v8, v0, v4   ; vfmadot v8, v0, v4 (fp8->fp16)
    addi    a0, a0, 32
    addi    a1, a1, 32
```

Only 12 remaining software conversions (for pack/unpack boundary ops),
down from 652 in the baseline.

### Board test: vfmadot SIGILL

The SpacemiT X60 board has `vmadot` (OPMMA, integer) but **not** `vfmadot`
(OPFMMA, floating-point). The FP8 VMFB correctly generates vfmadot
instructions but they trap with SIGILL on the current hardware revision.

The int8 vmadot works and gives 9x speedup. The FP8 vfmadot will work on
future hardware that implements the full OPFMMA instruction set.

## Saturn OPU Per-Operand Encoding Optimization (2026-03-18)

### Problem: strided loads dominate OPU kernel throughput

The original OPU kernel used `vlse8.v` (strided vector loads) because IREE's
mmt4d packs tiles as `[M0, K0]` row-major. To extract a column of M0=16
elements for one k0 value, the kernel needs stride=K0 access:

```asm
; BEFORE: strided loads (1 element/cycle on Saturn VLSU)
vlse8.v v16, (lhs_ptr + k0), stride    ; 16 cycles
vlse8.v v18, (rhs_ptr + k0), stride    ; 16 cycles
.insn r 87, 2, 81, zero, v16, v18      ; VOPACC ~4 cycles
; Total per k0: ~36 cycles (load-bound, 16x overhead)
```

Saturn's documentation (Section 4.6) confirms: "Saturn's VLSU is designed
towards deployment as a DSP system, and thus fundamentally has limited
performance on indexed or strided accesses, as it can only generate one
element's address per cycle." Contiguous loads run at full dLen bandwidth
(16 elements/cycle for dLen=128).

The Saturn OPU benchmarks (`third_party/saturn-vectors/benchmarks/opu-gemm/`)
avoid this entirely by pre-transposing the A matrix so M is innermost,
enabling `vle8.v`.

### Solution: per-operand encoding for xopu targets only

IREE's `getEncodingInfoImpl()` is called **once per operand** (LHS, RHS,
result), and the encoding attribute carries `operandIdx`. We exploit this to
swap the inner dimension order for LHS/RHS when `+xopu` is present, without
affecting the result operand or any other target.

**Compiler change** (`CPUEncodingExternalModels.cpp`, ~10 lines):

```cpp
// In getEncodingInfoImpl(), after getEncodingInfoForMatmul():
if (hasFeature(layoutAttr.getConfiguration(), "+xopu")) {
  int64_t operandIdx = encoding.getOperandIndex().getInt();
  if (operandIdx != IREE::Encoding::MATMUL_RESULT &&
      info.innerDimsPos.size() >= 2) {
    size_t sz = info.innerDimsPos.size();
    std::swap(info.innerDimsPos[sz - 2], info.innerDimsPos[sz - 1]);
    std::swap(info.innerTileSizes[sz - 2], info.innerTileSizes[sz - 1]);
  }
}
```

This changes:
- LHS tile: `[M0, K0]` → `[K0, M0]` (M innermost = contiguous)
- RHS tile: `[N0, K0]` → `[K0, N0]` (N innermost = contiguous)
- Result: unchanged (`[M0, N0]`, has no K dim, skipped by condition)
- `outerDimsPerm`: unchanged (mmt4d outer iteration order preserved)

**What's NOT affected:**
- RVV `_v` path (K0=1, stride=1, already contiguous)
- SpacemiT `_xsmtvdot` path (no `+xopu` feature)
- ARM, x86, or any other architecture
- The result operand encoding

**Kernel change** (`mmt4d_riscv_64_v_i8.c`): replace `vlse8.v` with `vle8.v`:

```asm
; AFTER: contiguous loads (16 elements/cycle on Saturn VLSU)
vle8.v v16, (lhs_ptr + k0*16)          ; 1 cycle
vle8.v v18, (rhs_ptr + k0*16)          ; 1 cycle
.insn r 87, 2, 81, zero, v16, v18      ; VOPACC ~4 cycles
; Total per k0: ~6 cycles (compute-bound, optimal)
```

The transpose cost is absorbed into the `linalg.pack` operation that runs
once before the mmt4d kernel. The pack already copies and tiles the data;
with the encoding swap it simply writes [K0, M0] order instead of [M0, K0].
This is a one-time cost amortized over all K iterations.

### Verified assembly

Compiled `matmul_i8.mlir` with `+xopu`:

```asm
.LBB0_2:                              ; hot loop (k0 unrolled by 2)
    vle8.v  v16, (a1)                 ; contiguous LHS load
    vle8.v  v18, (a0)                 ; contiguous RHS load
    .insn r 87, 2, 81, zero, a6, s2   ; VOPACC m0, v16, v18
    vle8.v  v20, (a5)                 ; LHS (k0+1)
    vle8.v  v22, (a3)                 ; RHS (k0+1)
    .insn r 87, 2, 81, zero, s4, s6   ; VOPACC m0, v20, v22
    addi    a0, a0, 32
    addi    a1, a1, 32
    bltu    a2, a4, .LBB0_2
```

No `vlse8.v` (strided) in the hot loop. All loads are `vle8.v` (contiguous).
No stack spills. **6x improvement on the inner loop** (36 → 6 cycles per k0).

### Additional fixes in this session

1. **VOPACC/OPFMACC operand order bug**: rs1 and rs2 were swapped, producing
   transposed output. Cross-referenced with `bme.h`: rs1=LHS(rows),
   rs2=RHS(cols). Fixed for both int8 VOPACC and fp8 OPFMACC.

2. **OPMVINBCAST register**: changed from mc0 (x16, column-broadcast) to m0
   (x0, row-broadcast) to match the Saturn benchmark pattern.

3. **Removed duplicate function definitions**: cleaned up broken sed-edit
   remnants (K0=1 fallback functions conflicting with K0=16 tile functions).

### Why K0=16 (and the case for K0=128)

With contiguous loads, the inner compute per k0 is identical regardless of
K0. The total VOPACC calls and loads are always `2 * total_K`. However, K0
controls the split between the inner k0 loop and the outer K-tile loop, and
the outer loop has real overhead per iteration:

```
Per K-tile overhead (~5 cycles):
  - 2x pointer arithmetic (lhs_k_ptr, rhs_k_ptr)
  - 1x vsetvli e8,m1 (may stall Shuttle pipeline on vtype transition)
  - 1x loop branch + counter increment
  - potential branch misprediction on small trip counts
```

For total K=128:

| K0 | K-tiles | Overhead (cycles) | Compute (cycles) | Overhead % |
|----|---------|------------------:|------------------:|-----------:|
| 16 | 8 | 40 | 768 | **5.2%** |
| 32 | 4 | 20 | 768 | 2.6% |
| 64 | 2 | 10 | 768 | 1.3% |
| 128 | 1 | 5 | 768 | **0.7%** |

The overhead scales linearly with K-tiles, so the ~5% penalty is consistent
regardless of total K. For large matrices (K=1024): K0=16 → 64 K-tiles ×
5 = 320 overhead cycles vs K0=128 → 8 K-tiles × 5 = 40 cycles.

Additional costs beyond raw cycle count:
- **vsetvli pipeline stall**: each K-tile iteration transitions from e32
  (accumulator) to e8 (loads). Shuttle may bubble on this vtype change.
  With K0=16, this happens 8x per tile; with K0=128, only 1x.
- **Branch prediction**: the K-tile loop has a small trip count that may
  not predict well on Shuttle's in-order pipeline.
- **Icache pressure**: more outer iterations = more fetch cycles for the
  loop preamble.

**Current choice: K0=16.** This was originally motivated by strided loads
(stride=K0=16 = 1 cache line). With contiguous loads that constraint is
gone, but K0=16 still has practical advantages:
- Divides all common K dimensions (64, 128, 256, 512) cleanly
- Pack tiles are small (256B), good for L1 locality
- The kernel code already uses `params->K0` dynamically

**TODO: evaluate K0=128.** See follow-up task below for what this requires.

### Upstream RVV i8 patch

Updated clean patch at `patches/upstream/riscv64-mmt4d-i8-rvv-only.patch`.
Contains only standard RVV code (no xsmtvdot, no xopu, no fp8). 6 file diffs:

1. New `mmt4d_riscv_64_v_i8.c` (201 lines, pure RVV intrinsics)
2. `mmt4d_riscv_64_tiles.inl` (5 `_v` s8s8s32 entries)
3. `query_tile_sizes_riscv_64_entry_point.c` (i8i8i32 query with `_v` branch)
4. `CPUEncodingExternalModels.cpp` (standard RVV i8 tile enumeration)
5. `CMakeLists.txt` (add source to bitcode library)
6. `BUILD.bazel` (add source to srcs)

## Follow-Up Tasks

- **Saturn FPGA/simulator test**: verify OPU int8 and fp8 correctness
- **Upstream PR**: submit the RVV i8 patch to IREE upstream
- **FP8 OPU on-board**: test `OPFMACC` assembly on Saturn hardware
- **E5M2 altfmt**: add `VSETVLI_ALTFMT` support (vtypei bit 8 = 1)

### TODO: Evaluate K0=128 for OPU

Increasing K0 from 16 to 128 eliminates ~5% outer-loop overhead per tile.
The kernel code already uses `params->K0` dynamically, so no kernel changes
are needed. The changes required are compiler-side only:

**Files to modify (3 files, ~6 lines total):**

1. `CPUEncodingExternalModels.cpp` — change xopu tile from `{16, 16, 16}`
   to `{16, 16, 128}` in `enumerateMatmulTileRiscv64()` for both i8 and fp8
2. `query_tile_sizes_riscv_64_entry_point.c` — change xopu return from
   `{.M=16, .K=16, .N=16}` to `{.M=16, .K=128, .N=16}`
3. `mmt4d_riscv_64_tiles.inl` — change K0 from 16 to 128 in all xopu entries
   (10 lines: 5 for s8s8s32, 5 for f8e4m3f8e4m3f32)

**Implications:**

| Aspect | K0=16 | K0=128 |
|--------|------:|-------:|
| Pack tile size (LHS) | 256B | 2KB |
| Pack tile size (RHS) | 256B | 2KB |
| Outer loop overhead | ~5% | ~0.7% |
| K divisibility | K%16==0 | K%128==0 |
| L1 cache pressure | minimal | still fits (32KB L1) |

**Risks:**
- K must be divisible by 128 (or IREE pads, adding waste). Common NN
  dimensions (128, 256, 512, 1024) divide cleanly. Odd dimensions
  (e.g. K=192) would need 128+64 split with K0=128 and padding/fallback
  for the remainder.
- Larger pack tiles may delay pipeline start: the first mmt4d invocation
  can't begin until a full 2KB tile is packed, vs 256B with K0=16.
- No impact on correctness — the kernel uses `params->K0` dynamically.

**Validation plan:**
1. Change the 3 files above
2. Rebuild host (`--config release --with-plugin`)
3. Compile `matmul_i8.mlir` with `+xopu` and verify `.s` shows same
   `vle8.v` + VOPACC hot loop (just more inner iterations)
4. Benchmark K0=16 vs K0=128 on Saturn simulator for 1024x1024 matmul
5. If K0=128 shows measurable improvement, adopt it as default

---

## OPU Encoding Resolver Results (2026-04-02)

### Background

The standard IREE CPU encoding path for matmul produces 3 dispatches:

1. **Pack LHS**: `[M, K]` -> `[M/M0, K, M0, K0]` (set_encoding -> linalg.pack)
2. **Pack RHS**: `[N, K]` -> `[N/N0, K, N0, K0]` (set_encoding -> linalg.pack)
3. **Fill + mmt4d + Unpack**: compute on packed 4D tensors, then copy 4D -> 2D

The unpack dispatch copies from packed 4D `[M/M0, N/N0, M0, N0]` to 2D `[M, N]`.
This is pure overhead: the OPU ukernel can write directly to 2D output using
strided `vse32.v` stores. For 1024x1024, the unpack was ~20% of total cycles.

The OPU encoding resolver eliminates the unpack by using **identity encoding**
for the matmul result. It follows IREE's GPU data-tiling pattern:

- `OPUEncodingResolverAttr` returns empty `MaterializeEncodingInfo` for the
  result operand, so `unset_encoding` folds to a no-op
- LHS/RHS keep packed encoding with 64x64x1 tiles for contiguous VOPACC loads
- The matmul lowers directly to `iree_codegen.ukernel.generic "iree_uk_opu_matmul"`
  instead of `linalg.mmt4d`, since mmt4d requires a 4D result shape

After this change, the dispatch flow is:

1. **Pack LHS** (unchanged, hoistable for constant weights)
2. **Pack RHS** (unchanged, hoistable for constant weights)
3. **Fill(2D) + iree_uk_opu_matmul(packed LHS, packed RHS -> 2D output)**

Key files:

- `compiler/.../CPU/IR/IREECPUAttrs.td` — `OPUEncodingResolverAttr`
- `compiler/.../ExternalInterfaces/CPUEncodingExternalModels.cpp` — 5 external models
- `compiler/plugins/target/LLVMCPU/LLVMCPUTarget.cpp` — selects OPU resolver for `+xopu`
- `runtime/.../arch/riscv_64/opu_matmul_riscv_64.c` — OPU matmul ukernel with 2D output

### Hardware Configuration

- **Config**: Saturn OPU V128-D64 Shuttle (`alveo_u250_firesim-opu-v128-d64-shuttle`)
- **VLEN**: 128 bits -> 16 i8 elements per vector register
- **DLEN**: 64 bits -> (DLEN/8)^2 = 64 MACs/cycle in the 8x8 MACC array
- **OPU peak**: 2 x (DLEN/8)^2 = 128 FLOPs/cycle (counting both mul and add per MAC)
- **Matrix registers**: 4 x 16x16 (m0-m3), VOPACC takes 4 cycles per instruction
  (processes 8x8 sub-tile per cycle, 16x16 = 4 sub-tiles)
- **FPGA**: Alveo U250 via FireSim

### How Performance is Measured

Benchmark harness: `samples/SaturnOPU/simple_embedding_ukernel/simple_embedding.c`

```c
// rdcycle hardware counter (RISC-V CSR)
uint64_t start = read_cycles();
for (int i = 0; i < 10; ++i)
    iree_vm_invoke(context, main_function, ...);
uint64_t end = read_cycles();
uint64_t avg_cycles = (end - start) / 10;

// Total ops = 2 * M * N * K (counts both mul and add per MAC)
uint64_t total_ops = 2 * M * N * K;
// ops_per_cycle = total_ops / avg_cycles
```

**What the cycle count includes**: IREE VM dispatch, HAL command buffer,
pack LHS dispatch, pack RHS dispatch, fill + matmul dispatch, memory allocation.
This is end-to-end latency, not just ukernel compute time.

**Utilization** = ops_per_cycle / peak = ops_per_cycle / 128

**Inputs**: A[i][k] = 2, B[k][j] = 4 (constant i8 values).
Verification: C[i][j] = 8 * K (checked every 101st element).

### Results

| Size | OPU Cycles | OPU Ops/Cycle | RVV Cycles | RVV Ops/Cycle | Speedup | Utilization |
|------|-----------|--------------|-----------|--------------|---------|-------------|
| 64x64x64 | 132,712 | 3.95 | 318,555 | 1.64 | 2.4x | 3.1% |
| 128x128x128 | 299,041 | 14.02 | 1,800,432 | 2.32 | 6.0x | 11.0% |
| 256x256x256 | 1,137,634 | 29.49 | 13,846,932 | 2.42 | 12.2x | 23.0% |
| 512x512x512 | 6,321,465 | 42.46 | 138,156,393 | 1.94 | 21.9x | 33.2% |
| 1024x1024x1024 | 37,057,772 | 57.94 | 1,580,608,373 | 1.35 | 42.9x | 45.3% |
| 2048x2048x2048 | 261,803,423 | 65.62 | 13,726,856,451 | 1.25 | 52.5x | 51.3% |

### Optimization Journey (1024x1024x1024)

| Step | Ops/Cycle | Date | What Changed |
|------|-----------|------|-------------|
| RVV Baseline | 1.35 | 03-22 | Standard RVV vectorized matmul, no ukernels, no OPU |
| Initial mmt4d+OPU | 0.84 | 03-21 | First mmt4d OPU ukernel; tile query returned 1x1x1 (wrong) |
| Fixed Tile Query | 14.61 | 03-22 | Correct 16x16x1 tile selection in `query_tile_sizes` |
| K-loop Unroll x4 | 34.82 | 03-22 | mmt4d K-loop unrolled by 4 with register pair rotation |
| Const Weights | 38.98 | 03-23 | Compile-time packed RHS via `--iree-global-opt-data-tiling` |
| Encoding Resolver | 57.94 | 04-02 | Identity result encoding eliminates unpack dispatch, 64x64 tiles |

**Key observations**:

1. The encoding resolver gave **1.67x improvement** over the best mmt4d result
   (57.94 vs 34.82 ops/cycle) by eliminating the unpack copy.

2. At 2048x2048, the OPU achieves **51.3% utilization**. The remaining ~49% is
   pack dispatch overhead, IREE runtime dispatch, fill, and L1 cache misses from
   2D strided stores (8KB between rows for N=2048).

3. The RVV baseline degrades at large sizes (2.42 -> 1.25 ops/cycle from
   256 to 2048) due to cache pressure. The OPU benefits from packed data layouts
   that improve cache behavior.

4. Assembly analysis of the ukernel K-loop confirmed it runs at hardware
   saturation: the VOPACC pipeline is the bottleneck, not scalar instruction
   overhead. Branch elimination and K-loop unrolling produced no measurable
   improvement.

5. Speedup grows with matrix size because the RVV baseline is memory-bound
   (no packed layout) while the OPU benefits from contiguous packed data access.

### Plots

Generated by `benchmarks/SaturnOPU/plot_firesim_results.py`:

- `benchmarks/SaturnOPU/performance_scaling.png` — Ops/Cycle vs Matrix Size
- `benchmarks/SaturnOPU/speedup_vs_rvv.png` — Speedup over RVV
- `benchmarks/SaturnOPU/utilization.png` — Hardware utilization
- `benchmarks/SaturnOPU/optimization_journey.png` — 1024x1024 progression

Raw data: `benchmarks/SaturnOPU/firesim_v128d64_results.csv`

---

*Dev-blog written by:* Agustin Coppari Hollmann
