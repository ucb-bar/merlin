# Upstream IREE Patch: RISC-V 64 Int8 MMT4D Ukernel

## Summary

Adds int8 x int8 -> int32 mmt4d microkernel support for RISC-V 64 with
the V extension (RVV). Uses standard RVV widening multiply-accumulate
instructions (`vwmul.vx` + `vwadd.wv`), no vendor extensions.

Tile: `MxNx1` where M={1,2,4,7,8} and N=VLEN/32*2 (runtime-determined).
For VLEN=256: `8x16x1`.

## Patch

`riscv64-mmt4d-i8-full.patch` contains ALL our mmt4d changes including
vendor-specific extensions (xsmtvdot, xopu). For the upstream PR, only
the following content should be included:

### New file: `mmt4d_riscv_64_v_i8.c` (lines 1-201 only)

The generic RVV int8 kernel function:
- `iree_uk_mmt4d_tile_s8s8s32_1xXXx1_to_8xXXx1_riscv_64_v()`
- M0 specializations for M=1,2,4,7,8
- Uses only `<riscv_vector.h>` intrinsics (no inline assembly)

**Exclude:** Everything after line 201 (xsmtvdot, xopu kernels)

### Modified: `mmt4d_riscv_64_tiles.inl`

Add after the f32 tiles:
```c
IREE_UK_MMT4D_TILE(riscv_64, s8, s8, s32, 1, 1, _v)
IREE_UK_MMT4D_TILE(riscv_64, s8, s8, s32, 2, 1, _v)
IREE_UK_MMT4D_TILE(riscv_64, s8, s8, s32, 4, 1, _v)
IREE_UK_MMT4D_TILE(riscv_64, s8, s8, s32, 7, 1, _v)
IREE_UK_MMT4D_TILE(riscv_64, s8, s8, s32, 8, 1, _v)
```

**Exclude:** xsmtvdot and xopu tile entries

### Modified: `query_tile_sizes_riscv_64_entry_point.c`

Add i8xi8->i32 query function:
```c
static iree_uk_matmul_tile_sizes_t
iree_uk_query_matmul_tile_sizes_riscv_64_s8s8s32(
    const iree_uk_query_tile_sizes_2d_params_t* params) {
#if defined(IREE_UK_BUILD_RISCV_64_V)
  if (iree_uk_cpu_riscv_64_v(params->cpu_data)) {
    return (iree_uk_matmul_tile_sizes_t){.M = 8, .K = 1, .N = 16};
  }
#endif
  return (iree_uk_matmul_tile_sizes_t){.M = 8, .K = 4, .N = 8};
}
```

Add case in `iree_uk_query_matmul_tile_sizes_arch()`:
```c
if (op == IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_S8S8S32) {
  *out_matmul_tile_sizes =
      iree_uk_query_matmul_tile_sizes_riscv_64_s8s8s32(params);
  return true;
}
```

**Exclude:** xsmtvdot and xopu branches

### Modified: `CPUEncodingExternalModels.cpp`

In `enumerateMatmulTileRiscv64()`, add for i8xi8->i32 with standard RVV:
```cpp
// Standard RVV i8 widening path.
int N0 = vlen / 32 * 2;  // targets LMUL=2 for i32 accumulators
return {
    TileMxNxK{8, N0, 1},
    TileMxNxK{7, N0, 1},
    TileMxNxK{4, N0, 1},
    TileMxNxK{2, N0, 1},
    TileMxNxK{1, N0, 1},
};
```

**Exclude:** xsmtvdot and xopu tile blocks

### Modified: `CMakeLists.txt` and `BUILD.bazel`

Add `mmt4d_riscv_64_v_i8.c` to the `ukernel_bitcode_arch_riscv_64_v`
library source list.

## Verification

Compile a quantized matmul with ukernels=all and verify the assembly
contains `vwmul.vx` + `vwadd.wv` in the hot loop:
```bash
iree-compile matmul_q_i8.mlir \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-triple=riscv64-unknown-linux-gnu \
  --iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c,+v,+zvl256b \
  --iree-llvmcpu-enable-ukernels=all \
  --iree-llvmcpu-link-ukernel-bitcode=true \
  --iree-opt-data-tiling=true \
  --iree-hal-dump-executable-files-to=./files/ \
  -o matmul_q_i8.vmfb
```

## Assembly Reference

Expected hot loop (tile 8x16x1):
```asm
.LBB_inner:
    vsetvli  zero, zero, e8, m1, ta, ma
    vle8.v   v10, (rhs_ptr)           # Load N=16 i8 RHS elements
    lbu      a4-t5, (lhs_ptr)         # Load M=8 scalar i8 LHS elements
    vwmul.vx v8, v10, a4              # Widening multiply: i8*i8 -> i16
    vsetvli  zero, zero, e16, m2, ta, ma
    vwadd.wv v0, v0, v8               # Widening add: i16+i32 -> i32
    # ... repeated 8 times for M=8
    bnez     loop_counter, .LBB_inner
```
