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
  (11.2 ins/tick). Low instruction count together with low IPC at lmul4 is the signature of
  register spilling: an lmul4 spill moves 128 B in a *single* instruction, so it is cheap to count
  and expensive to execute.
- `NR=16` plain v3 has 6.8x the instruction count but ~2x XNNPACK's IPC, and wins on time.

Combining the first's instruction count with the second's IPC is the open opportunity, and is worth
more than any remaining point in the shape space — see below.

## The shape space is swept out

Replicating the expert's *shape* does not replicate its performance. Measured on the board:
`MR` ∈ {1..8} × `NR` ∈ {16,32}, with and without `unroll_m`, plus real K-blocking. XNNPACK's **own**
`1x4v` shape (`MR=1, NR=32`) lands at 5.81x — worse than our own best `MR=4, NR=16` at 5.0x. Cache
blocking is flat (KC 16/32/64 → 5.0x/4.9x/4.9x), as expected once you notice 128³ f32 is ~192 KB and
already resident.

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
