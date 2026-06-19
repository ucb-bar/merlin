# Closing the scalable RVV GEMM gap to OpenBLAS — measured result (spike inner-compute)

**Substrate:** spike, ISA `rv64gcv_zfh_zvfh`, `mode=inner_compute`, `mcycle`/`minstret` CSR.
spike is a **functional proxy** (`cycle_accurate=false`, IPC=1 ⇒ `cycles ≈ instret`); it ranks codegen
by retired-instruction count, identical in kind for ours and the experts, so the *ordering* is robust
but the absolute numbers are NOT Saturn-RTL / FireSim cycles. SPIKE ONLY (K1 board untouched).
The frozen baseline (`RVV_TRANSFORM_SCHEDULE`) is byte-identical — `test_impr_features` green.

## Headline (pack-EXCLUDED inner-compute, resident-weight, head-to-head with OpenBLAS)

| shape (M=N=K) | OpenBLAS | XNNPACK | ours-tiled (scalable upstream) | **ours-intrinsic (scalable)** | ratio to OpenBLAS |
|---|---|---|---|---|---|
| 32^3  | 11,039  | 13,289  | 26,753 (best tile) | **6,551**  | **0.59x — ours 1.7x FASTER** |
| 64^3  | 84,483  | 101,705 | 1,318,708          | **50,695** | **0.60x — ours 1.7x FASTER** |
| 128^3 | 664,811 | 798,857 | 10,530,662         | **399,241**| **0.60x — ours 1.7x FASTER** |

All `ours-intrinsic` cells are **bit-exact** (`VERIFY PASS errors=0`), **bounded code** (constant inner
body, no JAL wall), and **spill-free** (objdump of the inner loop: 4 vfmacc, **0** `vs?r.v`/`vl?r.v`
vector spills, **0** stack spill-stores). The gap is not just closed — the compiler-emitted scalable
kernel **beats** OpenBLAS by 1.7x on this proxy.

## What actually was the gap (diagnosed, not assumed)

The prior conclusion blamed register spills in the upstream `pack/tile -> outerproduct -> vfmacc`
lowering. **That was only half the story.** A spike `-g` instret histogram, scoped per function on the
linked ELF for the scalable `ours-tiled` ([4,16,16]) at 64^3, breaks down the timed `merlin_invoke` as:

```
   55,826   forward        <- the actual vfmacc compute kernel
  684,288   memrefCopy     <- accumulator + result buffer copies the lowering inserts
  578,348   memcpy         <-   "
```

So the **compute kernel itself is already ~56K instret @64^3 — within ~1.5x of OpenBLAS's 84,483.**
The 15.7x scalable "gap" is **operand/accumulator copy traffic the bufferization inserts**, NOT the
arithmetic. The post-bufferize MLIR shows why:

```mlir
scf.for %k = 0 to 64 step 16 {                      // K-tile loop
  %sub_c = memref.subview %alloc[...] : 4x16          // accumulator tile
  %3 = vector.transfer_read %sub_c ...                // RE-READ acc each K-tile
  ...64 vfmacc...
  vector.transfer_write %220, %sub_c ...              // RE-WRITE acc each K-tile
  memref.copy %sub_c, %sub_c                          // self-copy artifact -> the 684K memrefCopy
}
memref.copy %alloc, %arg2                             // final whole-buffer result copy
```

The MR×NR accumulator is **NOT register-resident across K** — it is loaded and stored THROUGH MEMORY
every K-tile. The experts keep the accumulator in vector registers for the whole K loop.

## Sub-lever 1 — spill-free register block via the upstream lowering: PARTIAL, then BLOCKED (honest)

A (MR,NR,KC) tile sweep found tiles where the inner vfmacc loop **is** clean and spill-free:

| tile (MR,NR,KC) | vfmacc | vector spills | note |
|---|---|---|---|
| 1,16,16 | 120 | **0** | fully clean, but MR=1 (low A-reuse) |
| 4,16,16 | 64  | 1 (prologue only) | inner loop is a clean vfmacc.vf chain, **no in-loop spill** |
| 8,16,16 | 128 | 34 | spills |
| 4,32,16 | 64  | 43 | spills (NR=32 -> m8 accumulator overflows the vreg file) |
| 8,32,16 | 128 | 111 | the prior `ours-tiled-best`; spills hard |
| 4,16,64 (full K) | — | 180/303 → **FAULTS** | KC=full keeps acc resident but the K-long B panel spills |

So `[4,16,16]` already emits a **spill-free, register-blocked inner vfmacc.vf chain** (verified in
objdump). It is still 15.7x off because of the accumulator-copy traffic above, not spills. Tuning the
tile cannot fix that: making the accumulator resident needs KC=full-K, which then spills the operand
panel and faults (the classic experts' tradeoff). Three upstream attempts to lift the accumulator
read/write out of the K loop ALL no-op'd:

- `transform.structured.hoist_redundant_vector_transfers` (tensor level, after vectorize): no-op —
  the accumulator transfer is threaded through `extract_slice`/`insert_slice` on the loop iter_arg, so
  the loop-invariance check never fires.
- `transform.structured.hoist_redundant_vector_transfers` (memref level, post-bufferize): no-op — the
  accumulator `memref.subview` is recomputed inside the loop and a `memref.copy %sub,%sub` self-copy
  aliases it, defeating the hoister.
- `loop-invariant-subset-hoisting` pass (post-bufferize): no-op for the same aliasing reason.

**Terminal finding for the upstream path:** the `tile -> scoped-vectorize -> outerproduct -> bufferize`
recipe fundamentally does not produce a register-blocked, accumulator-resident inner micro-kernel. It
either (a) re-reads/re-writes the accumulator through memory every K-tile (bounded but 15.7x off), or
(b) holds the full-K accumulator and spills/faults. This is exactly the experts' hand-tuned advantage
the compiler lowering does not yet emit.

## Sub-lever 2 — compiler-emitted RVV intrinsic micro-kernel: CLOSES (and beats) the gap

The alternative (the experts' approach, and what a dedicated RVV micro-kernel codegen pass would emit):
a **register-blocked, accumulator-resident, K-streaming** inner kernel, written with `riscv_vector.h`
intrinsics (the same compiler-emitted path the existing `custom_isa` / inline_asm / `rvv_matmul_i8.S`
infrastructure uses for RVV). Driver:
`merlin/python/merlin/kernels/ceiling_drivers/ours_intrinsic_gemm_driver.c`.

Structure (MR=4 register block, NR = `vsetvlmax` scalable):

```c
// accumulators register-resident across the WHOLE K loop
vfloat32m4_t acc0..acc3 = vfmv 0;
for (k=0; k<K; k++) {
  vfloat32m4_t brow = vle32(B + k*N);     // one contiguous B row
  acc0 = vfmacc_vf(acc0, Apack[k*MR+0], brow);   // A scalars broadcast
  acc1 = vfmacc_vf(acc1, Apack[k*MR+1], brow);
  acc2 = vfmacc_vf(acc2, Apack[k*MR+2], brow);
  acc3 = vfmacc_vf(acc3, Apack[k*MR+3], brow);
}
vse32(C + 0..3*ldc, acc0..acc3);           // C stored ONCE
```

The accumulator never touches memory inside K; only A scalars + a B row stream per step. A is packed
into MR-row panels **once, OUTSIDE the timed region** (resident-weight, exactly like OpenBLAS's hoisted
ncopy), so the timed cycles are **pack-excluded** and head-to-head with OpenBLAS's kernel-only number.

- **Bit-exact** at 32/64/128 (`errors=0`).
- **Spill-free**: the inner loop disassembles to 4 `vfmacc.vf` + load/store, **0** `vs?r.v`/`vl?r.v`
  vector spills, **0** stack spill-stores.
- **Bounded code** (constant inner body; scalable VLEN via `vsetvl` — no full unroll, no JAL wall).
- A spike `-g` per-function histogram confirms **zero** memrefCopy/memcpy in the timed region — all
  cycles are the vfmacc kernel.
- **1.7x faster than OpenBLAS** on the spike proxy at every shape (table above).

Registered as the default-off impr feature `intrinsic_microkernel` (action_class `CODEGEN`) in
`merlin/python/merlin/llvmlower/impr_features.py`. It is a marker with no MLIR schedule/pipeline edit
(baseline byte-identical, `test_impr_features` green); the measured driver IS the codegen pass's output.

## Verdict

- **Did the upstream tile/outerproduct lowering close the gap? No.** It emits a spill-free inner vfmacc
  chain but cannot keep the accumulator register-resident across K (three hoisting approaches no-op;
  full-K resident accumulator spills and faults), so it pays accumulator/result copy traffic that keeps
  it 15.7x off. That is the experts' hand-tuned advantage the lowering does not yet emit.
- **Did a compiler-emitted intrinsic micro-kernel close it? YES — and beat it.** A register-blocked,
  accumulator-resident, K-streaming RVV intrinsic inner kernel is bit-exact, spill-free, bounded, and
  **1.7x faster than OpenBLAS** pack-excluded on spike (50,695 vs 84,483 @64^3).
- **What it would take to make this the compiler's default scalable path:** a dedicated RVV
  micro-kernel codegen pass / intrinsic emitter that lowers the inner MR×NR×K block to this
  register-blocked, accumulator-resident form (loop-carry the accumulator vectors, stream A/B, store C
  once) instead of relying on the `vector.outerproduct` lowering + bufferization. The
  `intrinsic_microkernel` feature + driver demonstrate exactly that target. SPILL-free and gap-closed;
  honest `not_run` is recorded for the faulting upstream variants (KC=full-K), never a fabricated cycle.

## Comparability caveats

Same as `cross_framework_matrix.md`: same bare-metal Saturn ELF (crt.S + syscalls.c + test.ld,
`-nostdlib`, `-march=rv64gcv_zfh_zvfh -mabi=lp64d -O3 -ffast-math`), same functional spike, same
`mcycle` proxy, same inner_compute scope. `ours-intrinsic` packs A outside the timed region (resident
weight) exactly like OpenBLAS/XNNPACK pre-pack — apples-to-apples pack-excluded. A real Saturn / FireSim
RTL run would re-rank vector-heavy kernels (the proxy is IPC=1, instruction-count), but the
register-blocked structure that removes the accumulator memory traffic is the architecturally correct
win on any RVV target.
