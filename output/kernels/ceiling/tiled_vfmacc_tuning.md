# Tiled-vfmacc register-tile tuning sweep (spike inner-compute, 64^3)

Goal: tune the SCALABLE tiled-vfmacc schedule's register-tile **(MR, NR, KC)** to close the gap to
the expert OpenBLAS/XNNPACK GEMM, on the SAME spike inner-compute harness
(`merlin.kernels.ceiling_drivers.multishape_compare.measure_ours`, bare-metal Saturn ELF, `mcycle`,
fill-subtracted, bit-exact-verified). The frozen baseline (`RVV_TRANSFORM_SCHEDULE`) is untouched;
every tuning point is a default-off `vfmacc_t_<MR>_<NR>_<KC>` impr feature emitting the proven
bounded-code recipe (`merlin/python/merlin/llvmlower/impr_features.py::vfmacc_tiled_schedule`).

**Honesty:** the spike number is an IPC=1 retired-instruction proxy (`cycle_accurate=false`); it
ranks codegen by instruction count, identical-in-kind for ours and the experts, so the
cross-framework ORDERING is robust but a real Saturn/FireSim would re-rank. Reference points (same
harness): OpenBLAS @64^3 = **84,483**, XNNPACK = 101,705, ours-full-unroll = 123,094, the current
tiled `[4,16,16]` = 1,328,219.

## Grid @ M=N=K=64 (MR∈{4,8}, NR∈{16,32}, KC∈{16,32,64})

Each tile divides 64 cleanly (no tail). Cycles are matmul-only (fill subtracted), bit-exact-verified.

| rank | feature | MR | NR | KC | cycles | status | OpenBLAS ratio | inner-body fma (MR·KC) |
|---|---|---|---|---|---|---|---|---|
| 1 | vfmacc_t_8_32_16 | 8 | 32 | 16 | **1,318,708** | pass | 15.61x | 128 |
| 2 | vfmacc_t_4_32_16 | 4 | 32 | 16 | 1,318,833 | pass | 15.61x | 64 |
| 3 | vfmacc_t_4_16_16 | 4 | 16 | 16 | 1,328,219 | pass | 15.73x | 64 |
| 4 | vfmacc_t_8_16_16 | 8 | 16 | 16 | 1,329,287 | pass | 15.74x | 128 |
| — | vfmacc_t_4_16_32 | 4 | 16 | 32 | — | not_run | — | spill fault (tohost=1337) |
| — | vfmacc_t_4_16_64 | 4 | 16 | 64 | — | not_run | — | spill fault (tohost=1337) |
| — | vfmacc_t_4_32_32 | 4 | 32 | 32 | — | not_run | — | spill fault (tohost=1337) |
| — | vfmacc_t_4_32_64 | 4 | 32 | 64 | — | not_run | — | spill fault (tohost=1337) |
| — | vfmacc_t_8_16_32 | 8 | 16 | 32 | — | not_run | — | spill fault (tohost=1337) |
| — | vfmacc_t_8_16_64 | 8 | 16 | 64 | — | not_run | — | spill fault (tohost=1337) |
| — | vfmacc_t_8_32_32 | 8 | 32 | 32 | — | not_run | — | spill fault (tohost=1337) |
| — | vfmacc_t_8_32_64 | 8 | 32 | 64 | — | not_run | — | spill fault (tohost=1337) |

`not_run` cause (recorded, never fabricated): every **KC≥32** config faults on spike with
`*** FAILED *** (tohost = 1337)` — the exact oversized-vector register-allocator spill that overruns
the bare-metal stack into BSS, documented as the root cause of the original K-untiled recipe's fault.
KC=16 keeps the tile vectors small enough to stay in the vector register file; KC=32/64 do not,
independent of MR/NR. So **KC=16 is a hard register-pressure ceiling** on this lowering/harness.

## Winner

**`vfmacc_t_8_32_16`** (MR=8, NR=32, KC=16): **1,318,708 cycles**, **15.61x OpenBLAS** (84,483).

- vs current `[4,16,16]` (1,328,219): **1.007x faster** — a marginal 9,511-cycle (0.7%) win.
- vs full-unroll (123,094): still **10.7x slower** — does NOT approach the full-unroll.
- Bounded + correct: inner-body vfmacc count is **constant across shapes** (128 fma = MR·KC at
  32/64/128; objdump-confirmed). Builds, runs and verifies bit-exact at **32^3 (26,753)**,
  **64^3 (1,318,708)** and **128^3 (10,530,662)** — no `R_RISCV_JAL` truncation, no spill fault.

The #1–#4 passing configs are within **0.8%** of each other; #2 `vfmacc_t_4_32_16` is a 125-cycle
statistical tie with the winner. The register tile barely moves the needle.

## Why the tile barely matters (and where the real gap is)

- **Larger MR amortizes A-reload, larger NR cuts N-loop trips — but they net-cancel.** Going
  MR 4→8 doubles the inner body (64→128 fma) yet halves the M-loop trips; widening NR 16→32 cuts
  N-loop trips but the per-tile work grows. The retired-instruction proxy sees these trade off
  almost exactly, hence the ~1.318–1.329M plateau across all passing tiles.
- **KC is the only lever that would cut the K-loop trip count (64/KC), and it is capped at 16 by
  register pressure.** With KC=16 the K-loop still runs 4 trips, each paying transfer/loop overhead.
  KC=32/64 would halve/quarter those trips but spill and fault — so the one knob that could close the
  gap is unavailable on this lowering.
- **The dominant cost is loop overhead + per-tile vector-transfer traffic, not the FMA tile.** The
  experts win by **B-packing / packed-resident weights** (OpenBLAS A-ncopy/B-tcopy; XNNPACK
  goi-pre-packed) so their microkernel streams contiguous packed panels with negligible
  address/loop overhead. Our recipe re-loads strided tiles every trip.

## Honest remaining gap

Register-tile tuning closes essentially **none** of the 15.7x gap (best 15.61x vs 15.73x — a 0.7%
improvement). The gap is **inherent to the un-packed tiled lowering**: it cannot be closed by tile
choice because (a) the only trip-cutting lever (KC) is pinned at 16 by the spill/register-pressure
ceiling, and (b) the remaining cost is strided per-tile transfer + loop overhead that the experts
eliminate with a **pack pass** (pre-packed, contiguous, resident weight panels). Closing the gap
therefore requires a SEPARATE compiler feature — a packing / packed-resident-weight pass
(`linalg`-level B-pack + panel-resident accumulation) — not a tiling-parameter change. That is out
of scope for this register-tile sweep and is the recommended next mining target.

## Harness / reproduction

- Sweep: `measure_ours('tune_'+nm, [nm], M=64,N=64,K=64)` over the 12 `vfmacc_t_*` grid features.
- Inner-body bound check: objdump `vfmacc` count of `model.o` at 32/64/128 (constant ⇒ bounded).
- spike: `/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike --isa=rv64gcv_zfh_zvfh`.
