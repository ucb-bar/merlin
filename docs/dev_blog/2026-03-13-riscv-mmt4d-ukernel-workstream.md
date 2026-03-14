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

## Follow-Up Tasks

- add true FP8 hardware-accelerated ukernels for `+xsmtvdot`
- add true FP8 hardware-accelerated ukernels for `+xopu`
- revisit OPU FP8 `K=128` after a native lowering exists
- add explicit hot-loop assembly checks for FP8 once there is a real target
  instruction sequence to validate
- decide whether the final target-specific FP8 support should stay in-tree in
  `third_party/iree_bar` or move to a cleaner out-of-tree extension layer
