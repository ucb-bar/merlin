# Gemmini multi-N-tile MVIN-D OOB: the bug that hid behind the FC-only test

**Date:** 2026-05-22
**Author:** dev-blog
**Status:** Fixed (Spike-verified all 11 dronet shapes; FireSim verification pending)

## Tl;dr

After the [2026-05-21 walkback-offset fix](2026-05-21-gemmini-indirect-binding-offset-fix.md) landed, `dronet_with_intermediate × Gemmini × FireSim` bit-perfectly matched the scalar baseline. But `dronet (non-intermediate) × Gemmini × FireSim` still diverged. The two variants share the same compiler and the same lowering pipeline — why does one work and the other not?

The answer: `with_intermediate` happens to only exercise the FC matmul (`1x1x2048`, a single 16x16 tile after padding). The conv-stack matmuls, exclusive to non-intermediate dronet, all use multi-N-tile shapes. A long-standing latent bug in the D (bias) buffer allocation hid behind the FC-only test coverage.

## The bug

In `compiler/src/merlin/Dialect/Gemmini/Transforms/LowerTileToISA.cpp`, both lowering patterns (`LowerMatmulTileToISA` for the tensor tier and `LowerBufferizedLinalgMatmulToTileMatmul` for the memref tier) allocated D as a hardcoded `16x16` i32 alloca + zero-init, with `repeatingBias=false`.

In `spTiledMatmulOs`, MVIN-D iterates per j-tile:

```cpp
for (size_t i0 = 0; i0 < i; i0++) {
    for (size_t j0 = 0; j0 < j; j0 += dBlocks) {
        const size_t biasRow = repeatingBias ? 0 : i0;
        const size_t offset = (biasRow * strideD + j0) * dim * sizeOfAccT;
        ...
    }
}
```

For the 16x16 alloca, `strideD = 16` (= D's column count) and `sizeOfAccT = 4` (i32). MVIN-D's `config_ld` stride is therefore `16 * 4 = 64` bytes per row.

For a multi-N-tile shape (e.g. M=16, N=32), the second j-tile MVIN-D fires with `offset = 1 * dim * sizeOfAccT = 64` bytes. The MVIN reads 16 rows × 16 cols of i32 with row stride 64:

- Row 0: bytes [D+64 .. D+127] (row 1 of D)
- Row 1: bytes [D+128 .. D+191] (row 2 of D)
- ...
- Row 14: bytes [D+960 .. D+1023] (row 15 of D — last in-bounds row)
- **Row 15: bytes [D+1024 .. D+1087] — OOB past the 1024-byte alloca**

Whatever is on the stack past D becomes "bias" for accumulator row 15 of tile-j=1. The compute then adds A·B = K on top:

> output[15, col_in_tile_1] = K + (i32 at stack OOB offset for col_in_tile_1)

`with_intermediate` only has a single 16x16 matmul tile, so `J=1` and MVIN-D never iterates → bug invisible. Non-intermediate dronet's conv-stack matmuls all have J≥2 → bug consistently fires at the last M-row.

## The smallest reproducer

Spike + bench:

```bash
./merlin build --profile firesim --cmake-target bench_gemmini_spike_matmul \
    --cmake-arg="-DGEMMINI_SPIKE_MATMUL_SHAPE=16x32x16"
SPIKE_HARTS=1 ./tools/spike-hetero/spike-hetero \
    build/firesim-merlin-release/runtime/plugins/merlin-samples/SaturnOPU/simple_embedding_ukernel/bench_gemmini_spike_matmul
```

Pre-fix output:

```
row    0..14: first=16 min=16 max=16 errs_in_row=0
row   15: errs_in_row=2 first_err_col=16 last_err_col=18 [16]=-938966256 [18]=-2147409298
```

Exactly two wrong cells, both at the start of the second N-tile. Cols 17, 19, 20, …, 31 are correct (= K = 16) — i.e. the OOB read happens to return 0 for those positions but two nonzero i32 values at cols 16 and 18.

## The fix

Allocate D as `1×N` i32 + zero-init, and set `repeatingBias=true` on the `gemmini.tile_matmul`. With `repeatingBias=true`, `spTiledMatmulOs`:

- forces `dStride = 0` (the MVIN replicates the same DRAM row 16 times into the spad)
- forces `biasRow = 0` so per-i-tile, per-j-tile offset stays at `j0 * dim * sizeOfAccT`

For our 16x32 output: D is 1×32 i32 = 128 bytes. MVIN-D[j=1] reads at offset 64 with stride 0 → reads bytes 64–79 of D (cols 16–31 of D's only row, all zeros) and replicates 16 times. All in-bounds. No OOB.

Stack footprint is `N * 4` bytes regardless of M, so this also works for large-M shapes (`3136x32x27` would have been ~400 KB if we'd sized D = M×N; that breaks the 32 KiB `max_stack_allocation_size`).

The two diffs are in `compiler/src/merlin/Dialect/Gemmini/Transforms/LowerTileToISA.cpp` at the two `auto i32MemRef = MemRefType::get({16, 16}, …)` sites and the corresponding `/*repeatingBias=*/false` flips.

## Verification

All 11 dronet matmul shapes PASS on Spike post-fix:

| shape | M-tiles × N-tiles × K-tiles | pre-fix | post-fix |
|---|---|---|---|
| 1x1x2048 | 1 × 1 × 128 | PASS | PASS |
| 196x32x32 | 13 × 2 × 2 | (untested isolated) | PASS |
| 49x64x32 | 4 × 4 × 2 | (untested isolated) | PASS |
| 16x128x64 | 1 × 8 × 4 | FAIL | PASS |
| 196x32x288 | 13 × 2 × 18 | (untested isolated) | PASS |
| 49x64x288 | 4 × 4 × 18 | (untested isolated) | PASS |
| 49x64x576 | 4 × 4 × 36 | (untested isolated) | PASS |
| 16x128x576 | 1 × 8 × 36 | FAIL | PASS |
| 16x128x1152 | 1 × 8 × 72 | FAIL | PASS |
| 3136x32x27 | 196 × 2 × 2 | BUILD-FAIL (alternate D-size attempt overflowed stack) | PASS |
| 16x32x16 | 1 × 2 × 1 | FAIL (smallest reproducer) | PASS |

FireSim verification on `dronet (non-intermediate) × Gemmini` and `yolov8n × Gemmini` is the remaining work.

## Why we didn't catch this earlier

The post-mortem in [2026-05-21-gemmini-indirect-binding-offset-fix.md](2026-05-21-gemmini-indirect-binding-offset-fix.md) already covered most of the meta-process lessons. Two specifically new ones from this bug:

**1. "Same-compiler, same-pipeline → must produce same result" is a wrong invariant for accelerator backends.** Two MLIR modules that hit the same lowering can still diverge at the hardware level if they exercise different tile-loop configurations. We assumed the FC pass-through implied the conv-stack pass-through. The actual independent variable is *which spTiledMatmulOs inner-loop iteration counts each input exercises*.

**2. Test fixture coverage was wrong-dimensional.** The existing Spike fixtures covered the FC shape (`1x1x2048`) and the conv-stack shapes — but they all passed because:
- 1x1x2048 = single tile (J=1, doesn't trigger MVIN-D iteration)
- conv-stack shapes pass the *first-row-of-output* check that the existing bench used

Adding `matmul_16x32x16` (the minimal multi-N-tile shape) and a per-row diagnostic in the bench (showing `first_err_col`, `last_err_col`, and the wrong cells' values) surfaced the bug in seconds. The "passes/fails" binary signal masked it; the *where it fails* signal didn't.

The codified takeaway: when adding fixtures for an accelerator backend, parameterize across `(M_tiles, N_tiles, K_tiles)` × `(corner-aligned, partial-tile)` independent variables — not just shape sizes. And bench diagnostics should report **which cell** is wrong, not just whether any cell is wrong.
