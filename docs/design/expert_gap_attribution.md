---
title: "Design note: attributing the expert-kernel gap (instructions vs stalls)"
kind: design
status: current
owner: core
last_verified: 2026-07-18
related: [beam_search]
code_refs: [merlin/python/merlin/rvvgen/pmu.py, merlin/python/merlin/kernels/microkernel.py, merlin/python/merlin/kernels/ceiling_drivers/k1_harness/util.h]
---

# Attributing the expert-kernel gap

## The problem with ranking on wall time

A kernel can be slow two ways, and they call for opposite fixes:

1. it **executes too many instructions** — a codegen-quantity problem a schedule *can* fix;
2. it **stalls on each instruction** — a memory/dependency problem a schedule *cannot* fix.

Wall time sums the two. A beam ranked on wall time alone therefore cannot see which one a fork
improved, and — worse — cannot tell either from a lever that did nothing at all.

## The instrument

The K1 exposes a PMU but ships no `perf(1)` and no native compiler, so counters are cross-built:
`merlin.rvvgen.pmu` for whole-ELF counts, and `k1_harness/util.h` making `read_csr(minstret)` real
so the ceiling drivers — compiled unchanged — report retired instructions **on the same bracket as
the rdtime timing**.

Kernel-region, not process-wide, is the one that matters for kernel comparisons: these drivers spend
most of their process in a scalar verification reference whose cost differs per driver, so
cross-driver process totals do not cancel. Measuring process-wide first produced a contradiction
(a cycle delta ~10x larger than the timed region could explain) that only resolved once the counter
moved inside the bracket.

## What it says

f32 GEMM 128³, K1 real silicon, kernel region:

| kernel | ticks | instructions | ins/tick |
|---|---|---|---|
| XNNPACK 7x4v | 11,547 | 250,361 | 21.7 |
| ours, best time (v3 MR=4, NR=16) | 41,195 | 1,710,650 | 41.5 |
| ours, fewest instructions (NR=32 + `unroll_m`) | 42,706 | 480,729 | 11.2 |

Against our best-*time* configuration the 3.57x gap is **~3.5x instruction count, ~1x IPC**. The gap
is overwhelmingly codegen quantity, which is the fixable kind.

This **corrects** an earlier conclusion, reached by hand-counting the inner loop in an objdump, that
the two kernels execute comparable instruction counts. They do not, and the method was the error:
counting the inner loop misses per-tile prologue/epilogue, which is where the extra work lives.
Hand-derived instruction counts should not be trusted when a counter is available.

## The two frontiers

No single configuration is best on both axes:

- `NR=32` + `unroll_m` reaches **1.91x** XNNPACK's instruction count — near parity — but stalls
  (11.2 ins/tick). The initial reading was register spilling, since low instruction count with low
  IPC at lmul4 fits that signature (an lmul4 spill moves 128 B in a *single* instruction: cheap to
  count, expensive to execute). The disassembly **refutes** it — the 8 spills are all in the
  prologue, outside the loops. The real cause is loss of B reuse; see below.
- `NR=16` plain v3 has 6.8x the instruction count but ~2x XNNPACK's IPC, and wins on time.

Combining the first's instruction count with the second's IPC is the open opportunity, and is worth
more than any remaining point in the shape space — see below.

## The shape space is swept out

Replicating the expert's *shape* does not replicate its performance. Measured on the board:
`MR` ∈ {1..8} × `NR` ∈ {16,32}, with and without `unroll_m`, plus real K-blocking. XNNPACK's **own**
`1x4v` shape (`MR=1, NR=32`) lands at 5.81x — worse than our own best `MR=4, NR=16` at 5.0x. Cache
blocking is flat (KC 16/32/64 → 5.0x/4.9x/4.9x), as expected once you notice 128³ f32 is ~192 KB and
already resident.

## Where the instructions actually go

Disassembling the two frontier configurations settles it. Our best-time inner loop is **already
better shaped than the expert's**: one B load amortized across four accumulators,

    vle32.v v24,(a0)          <- one B load
    vfmacc.vf v8,fa5,v24      <- four FMAs against it
    vfmacc.vf v20,fa4,v24
    vfmacc.vf v16,fa3,v24
    vfmacc.vf v12,fa5,v24

which is 4 FMAs per 12 instructions = **3.0 ins/FMA, against XNNPACK's 3.82**.

That loop executes 128 iterations x 256 tiles x 12 instructions = **393,216** instructions. The
counter measured **1,710,650**. So the hot loop is only ~23% of retired instructions and **~77% is
per-tile prologue/epilogue** -- 256 tiles of accumulator setup, writeback and address computation
around 1,536 instructions of useful work each.

This is the actionable target, and it is neither the inner loop nor the memory system: the fix is
fewer, larger tiles (amortizing the per-tile cost) rather than a better micro-kernel body. It also
explains why the whole shape sweep was flat -- every point in it re-tiles the same way, so it moves
the 23%, never the 77%.

It further explains why `unroll_m` is structurally wrong rather than merely inert: it emits **MR
sequential K-loops** (17 backward branches vs 3), each with a *single* `vfmacc` and a B reload every
step (`addi a1,a1,512`, a full row stride), amortizing nothing. Its spills are real but confined to
the prologue, so spilling was not the reason it stalled -- losing B reuse was.

## The actual defect: a `memrefCopy` per tile

Scaling retired instructions across N = 64 / 128 / 192 separates the two terms exactly. Fitting
`instret = a*N^3 + b*N^2` gives **a = 0.197 ins/MAC** and **b = 79 ins per output element**, and
predicts the N=192 point to within 0.4%.

The `N^3` coefficient is the hot loop, and it is essentially optimal: the disassembly predicts 12
instructions per 64 MACs = 0.1875, against 0.197 measured. The `N^2` term is pure overhead, and at
N=128 it is 1.29M of the 1.71M total -- **the entire gap**. XNNPACK's 250,361 is almost all `N^3`
term; it has no meaningful per-element cost.

The mechanism is named in the object file: `model.o` leaves **`memrefCopy` undefined**, and the tile
epilogue calls it once per tile after the accumulators have *already* been written by `vse32.v`:

    vse32.v v8,(a1) ... vse32.v v12,(a4)   <- accumulators stored (the real writeback)
    addi a2,a0,-64 ; mv sp,a2              <- then: dynamic alloca, x4 frames
    sd s2,-64(a0) ; vse64.v v8,(a0)        <- memref descriptor built in memory
    li a0,4 ; jalr ra                      <- generic strided-copy runtime call

`memrefCopy` is MLIR's rank-generic strided copy: it walks elements with dynamic rank/stride
handling. For a 4x16 tile that is ~5,000 instructions to move 64 values -- matching 79/element.
`model.ll` carries 2 such call sites and 4 allocas.

**This is a bufferization/lowering defect, not a scheduling one**, which is precisely why the entire
shape space read flat: no choice of MR/NR/KC/`unroll_m` can remove a runtime call the epilogue emits
regardless. It also corrects an earlier claim that the lowered kernel contained no
`memrefCopy`/`memcpy`/`malloc` and that the runtime was therefore ruled out -- that check read the
wrong artifact.

Removing the `N^2` term would take N=128 from 1.71M to ~0.42M instructions -- about 1.7x XNNPACK's
count. Since our measured IPC is roughly twice XNNPACK's on this path, that is the first change with
a credible route to parity, and it is worth more than every remaining point in the shape space
combined.

## Levers must be proven by emitted code

Two declared levers were **inert**, and neither was caught by the existing gates:

- `KC` — the v3 schedule contained no K-blocking whatsoever, so a beam-tunable knob was a no-op.
  Now real via `k_block` (`ensure_v3_kblocked_microkernel`).
- `MR` under `unroll_m` — the schedule text *does* differ (`transform.loop.unroll factor = 2` vs `7`)
  yet retired instructions are flat across MR ∈ {2,3,4,7} (0.4% spread). Unrolling the M loop cannot
  reduce per-FMA overhead because after tiling M by 1 the M loop sits **outside** the K reduction;
  the expert keeps MR accumulators live inside a **single** K loop. Still open.

The second is the more dangerous class: every layer looks wired, a schedule diff exists, and
`UnsupportedAxis` correctly does not fire. Only the emitted code disagrees.

**Rule:** a schedule-text diff is not evidence that a lever works. Confirm a lever with a measured
delta in emitted code — retired instructions are the cheap ground truth — and treat flat results
across a wide parameter sweep as an inert-lever suspicion rather than a tuning plateau.
