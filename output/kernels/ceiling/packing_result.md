# Operand-PACKING RVV GEMM — measured result (spike inner-compute)

**Feature:** `vfmacc_packed` (default-off impr feature, `merlin/python/merlin/llvmlower/impr_features.py`).
**Substrate:** spike, ISA `rv64gcv_zfh_zvfh`, `mode=inner_compute`, `mcycle` CSR.
spike is a **functional proxy** (`cycle_accurate=false`, IPC=1 ⇒ `cycles ≈ instret`); it ranks codegen
by retired-instruction count, identical in kind for ours and the experts, so the *ordering* is robust
but the absolute numbers are NOT Saturn-RTL / FireSim cycles.

## What the feature does (the mined `packed_rhs_policy`)

`transform.structured.pack %matmul packed_sizes=[MR,NR,KC]` packs A→`<M/MR×K/KC×MR×KC>`,
B→`<K/KC×N/NR×KC×NR>` (then `pack_transpose inner_perm=[1,0]` so the B panel is `[NR,KC]` and the
contraction needs **no runtime `vector.transpose`**), C→`<M/MR×N/NR×MR×NR>` — the [MR,NR,KC] register
tile is the **contiguous inner block** of each packed panel (OpenBLAS ncopy/tcopy, XNNPACK goi-prepack
layout). Then: tile the packed outer dims to 1 → `lower_pack`/`lower_unpack` to copy loops → fold unit
dims to a rank-3 inner op → **stream the KC reduction by KS=4** → scoped-vectorize →
`reduction_to_contract` → `lower_contraction(outerproduct)` → `lower_outerproduct` →
`vector.fma` → `llvm.intr.fmuladd` → **`vfmacc.vf`**. A pipeline edit inserts `eliminate-empty-tensors`
before bufferize (otherwise CSE merges the A-pack and C-pack `tensor.empty` dests onto ONE buffer).

## Confirmed (the packing concept works)

- **Lowers and forms the intended datapath.** At 32^3 the compiled `model.o` inner loop is
  **16 `vfmacc.vf` + 8 `vle32.v` (unit-stride contiguous) + 0 `vlse` (zero strided loads)**. The packed
  panels are read with contiguous `vector<MR×KS>` / `vector<KS×NR>` transfers — the strided
  per-tile `vector.transfer` that the tile-tuning sweep could not remove is gone.
- **Bit-exact at 32^3:** `VERIFY PASS errors=0`, `maxabs_err = 0` vs the scalar reference.
- **Bounded inner body** (constant MR·KS fma regardless of shape).

## Measured cycles (32^3, the shape that runs)

| column | 32^3 cycles (spike, inner-compute) | note |
|---|---|---|
| OpenBLAS `sgemm_kernel_8x8` | 11,039 | pack hoisted out (resident) |
| XNNPACK `1x4v__rvv` | 13,289 | weights pre-packed |
| ours-tiled-best `[8,32,16]` (no pack) | 26,753 | the prior best ours |
| **ours-packed (PACK-INCLUDED)** | **68,563** | pack of A/B/C inside the timed `forward` |

**Pack-INCLUDED, 32^3:** packing is **2.6× SLOWER** than the non-packed tuned tile (68,563 vs 26,753)
and **6.2× slower** than OpenBLAS. The pack/unpack copy traffic dominates: of the timed `forward`,
the `vfmacc` matmul loop is ~18.7 K committed insns and the pack+unpack+fill copy loops add the rest
(plus the `linalg.copy`/`memcpy` pack panels). **Pack-EXCLUDED** (compute-loop only, the apples-to-apples
vs the experts' hoisted pack) the matmul region is roughly the 18.7 K-insn loop — still **larger than
ours-tiled-best (26,753 includes its own loop overhead) is not beaten**, and nowhere near OpenBLAS's
11,039. So even hoisting the pack out, this MLIR-emitted packed micro-kernel does NOT close the gap at
32^3: the win the experts get from packing is hand-tuned panel-resident inner kernels (register-blocked,
software-pipelined), not the layout transform alone.

## Honest BLOCKER: faults at M ≥ 48 (64^3, 128^3 = `not_run`)

The packed feature **faults (`tohost = 1337`, `trap_store_access_fault`) at 48^3 and above.** Root cause,
traced on the spike commit log: the pack+fold+vectorize lowering produces **8–32 vector-register spills**
(`vs4r.v`/`vl4r.v`) in the inner kernel — vs **2 spills** for the proven non-packed tiled recipe — and at
M ≥ 48 a **mis-computed memref base pointer** (a buffer address ≈ `arena_base + ~16 MB` while only ~48 KB
was allocated) drives a vector store PAST its buffer, corrupting the bare-metal bump allocator's
`merlin_bm_off` and the TLS `buflen` (which then makes the first `printf` fault in `vprintfmt`). This is a
real **MLIR pack/fold/vectorize lowering bug** (wild pointer + spill-scratch overrun), **distinct from the
packing concept** (which is bit-exact at 32^3). It could not be removed from inside the transform schedule
or the default-off feature seam (tried: B pre-transpose to kill the runtime transpose — dropped 7.6M→3.1M
insns but still faults; KS-streaming KC=16→4 — halved 32^3 work and dropped spills 32→8 but still faults;
NR=8/m2 tiles; `eliminate-empty-tensors`; dropping `hoist-static-allocs`). None fixed the M≥48 wild store.

## Verdict

- **Does packing close the gap (inner-compute)? No — not as an in-microbench transform.** At the one shape
  that runs (32^3), packing is *slower* pack-included, and pack-excluded the MLIR-emitted packed loop still
  does not beat the tuned non-packed tile, let alone OpenBLAS.
- **The lever is real but needs more than a layout transform.** Packing *does* deliver the contiguous
  unit-stride inner transfers (0 `vlse`, confirmed). What it does NOT deliver, as emitted by the upstream
  pack→outerproduct→vfmacc path, is the experts' register-blocked, software-pipelined, spill-free inner
  micro-kernel — which is where their cycles come from. The MLIR outerproduct lowering spills heavily
  (8–32 vs 2), and at M≥48 that spill scratch + a base-pointer miscompute overruns.
- **Next step (to make packing actually free + correct), a runtime change, NOT just a hoist:** (1) move
  the A/B/C pack to a *resident-weight prepack* outside the timed kernel (the apples-to-apples
  resident-weight scenario) so the pack copy cost is amortized across M-tiles / reused weights; and
  (2) fix the inner micro-kernel's register allocation (bound the live panel / pipeline the K loop) so it
  is spill-free and the base-pointer miscompute that faults at M≥48 is gone. Until (2) lands, the packed
  kernel is correct only at 32^3 and is recorded `not_run` (with the precise blocker) at 64^3/128^3 — no
  cycle number is fabricated for the faulting shapes.

(Matrix: `output/kernels/ceiling/cross_framework_matrix.md`, column **ours-packed (pack-incl)**;
blockers persisted to `output/kernels/ceiling/cross_framework_notrun.jsonl`.)
