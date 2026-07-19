---
title: "Design note: attributing the expert-kernel gap (instructions vs stalls)"
kind: design
status: current
owner: core
last_verified: 2026-07-19
related: [beam_search]
code_refs: [merlin/python/merlin/rvvgen/pmu.py, merlin/python/merlin/rvvgen/k1.py, merlin/python/merlin/kernels/microkernel.py, merlin/python/merlin/kernels/ceiling_drivers/k1_harness/util.h, build_tools/scripts/k1_microkernel_ipc_sweep.py]
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

This is the actionable target, and it is neither the inner loop nor the memory system. The next
section identifies what that 77% actually is — a runtime call, not merely bulky setup — so the fix
is to remove it, not to amortize it over fewer, larger tiles.

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

## The vector-width/IPC tradeoff was a mis-specified target, not a tradeoff

Once the per-tile `memrefCopy` was erased, the shape space re-measured into what looked like a hard
frontier: `NR=32` (wide vectors) bought a low instruction count but collapsed IPC, `NR=16` bought IPC
but issued ~2x the instructions, and no point had both.

There was no frontier. We were compiling the model object with `-march=rv64gcv`, which promises only
the RVV **minimum** vector length of 128 bits (`zvl128b`). The K1 X60 has **VLEN=256**. Since our
codegen emits *fixed-width* vectors, the backend had to size each register group for that worst
case, so every vector value got exactly **double the LMUL the board needs** — read straight off the
disassembly, `vector<16xf32>` became `e32,m4` and `vector<32xf32>` became `e32,m8`. Two costs follow:

- **Register pressure doubles.** An `m8` value occupies 8 of the 32 architectural vector registers,
  so an `MR=4, NR=32` accumulator block wants all 32 and the allocator spills *inside* the K loop.
- **Half the datapath idles.** `vsetivli zero, 16, e32, m4` on a VLEN=256 core sets `vl=16` against a
  `VLMAX` of 32: the instruction reserves a four-register group and uses half of it.

Pinning the real VLEN (`merlin.rvvgen.k1.codegen_march` → `..._zvl256b`) is a one-flag change, and it
is the whole tradeoff. Inner-loop instruction counts, from the objdump, before → after:

| config | inner-loop insns | in-loop spill insns |
|---|---|---|
| `MR=4, NR=16` | 12 → 12 | 0 → 0 |
| `MR=2, NR=32` | 8 → 8 | 0 → 0 |
| `MR=4, NR=32` | 25 → **12** | 4 → **0** |
| `MR=8, NR=32` | 92 → **46** | 14 → **4** |

and on the board (128³, kernel region, min of 3, correctness-gated):

| config | ticks before | ticks after | instret before | instret after |
|---|---|---|---|---|
| `MR=4, NR=16` | 21,869 | 16,592 | 475,898 | 475,887 |
| `MR=2, NR=32` | 26,674 | 19,896 | 341,217 | 341,241 |
| `MR=4, NR=32` | 54,963 | **13,932** | 617,227 | **275,055** |
| `MR=8, NR=32` | 75,531 | 29,789 | 841,739 | 463,075 |

`MR=4, NR=16` is the control that proves the mechanism: its instruction count is **unchanged to 11
instructions** while time drops 1.32x. That config never spilled, so nothing about its *quantity* of
work could change; all it gained was a full-width `vl`. The configs that were spilling gained on both
axes at once.

`MR=4, NR=32` — previously the worst of the four — becomes the best point in the space, and its inner
loop is now the textbook one at full vector width, with no spill and no `vmv` shuffle:

    vle32.v v8,(s1)           <- one 32-lane B load  (e32,m4 == 256 bits)
    flw fa5,-0x400(a1) ...    <- four scalar A loads
    vfmacc.vf v12,fa5,v8      <- four FMAs against that one B load
    ...                       (12 instructions, 128 lane-FMAs)

The expert never hit this because it does not emit fixed-width vectors at all: XNNPACK sizes to the
vector length it *queries at run time* (`__riscv_vsetvl_e32m4`). That is why its advantage read as
"lanes per issue" — it was, and the lanes were being thrown away by the `march` string, not by any
property of wide vectors.

### What this leaves

At 256³ our retired-instruction count (1,916,395) is now **below XNNPACK's** (1,921,611), and the
whole residual 1.36x is IPC — 19.0 vs 25.9 ins/tick. The leading suspect is B-operand reuse: our
`MR=4` amortizes one B load over 4 FMAs, the expert's `MR=7` over 7. Reaching `MR=6..7` needs an
M-tail story, because in the 2-D `vector<MRxNR>` formulation `MR` must divide `M` — `MR=6` at M=128
falls off a cliff to 2.24M ticks (masked transfers → scalar), which is the same
small-M/masked-transfer wall documented elsewhere, not a register-pressure effect.

**Rule (companion to the one above):** an instruction-count/IPC "tradeoff" that no configuration
escapes is a reason to check what the *backend was told about the target*, not only what the schedule
asked for. Retired instructions and disassembly LMUL together localize it in one measurement;
neither alone does.

### `unroll_m` is wrong, not inert — and the VLEN fix does not rescue it

Re-checked against the emitted code (no board needed — lowering and disassembly answer it), at the
same `NR=32` where the 2-D formulation now emits its clean 12-instruction loop:

| recipe | back-edges | inner-loop insns | `vfmacc` | `vle32` | ins / lane-FMA |
|---|---|---|---|---|---|
| 2-D `vector<4x32>` | 4 | 12 | 4 | 1 | 0.094 |
| `unroll_m`, MR=2 | 8 | 6 | 1 | 1 | 0.188 |
| `unroll_m`, MR=4 | 16 | 6 | 1 | 1 | 0.188 |
| `unroll_m`, MR=7 | 36 | 6 | 1 | 1 | 0.188 |

The back-edge count scales with `MR` while the loop body stays at **one** `vfmacc` against a freshly
reloaded B: unrolling the M loop replicates the whole N+K nest `MR` times into `MR` *sequential* K
loops, so B reuse is 1 no matter what `MR` says. Per lane-FMA it is exactly 2x the 2-D form, at every
`MR`.

This **refines** the earlier "inert lever" reading. The emitted digests *do* differ per `MR`
(`7833b2e3…`, `a06a1564…`, `46e48729…`), so the lever is not a no-op — it changes the code without
changing the economics, which the flat retired-instruction count across `MR` had made look like
inertness. The two failure modes need different fixes, and only the disassembly separates them:

- **inert** — emitted code identical; the knob never reached the backend (`KC` before `k_block`);
- **structurally wrong** — emitted code changes with the knob, per-unit cost does not.

`unroll_m` is therefore not the route to the expert's `MR=7`; a correct recipe has to keep the `MR`
accumulators live inside a **single** K loop, which is what the 2-D form already does — its only
limitation is that `MR` must divide `M`, so the real missing capability is M-tail handling
(pad/peel), not M-unrolling.

`k_block` re-checked as genuinely live, and correctly so: `KC` ∈ {32, 64} each produce a distinct
digest, while `KC=128` at `K=128` reproduces the un-blocked digest **byte for byte** — blocking the
reduction by its own full extent is a no-op, and the digest says so. The inner loop is untouched in
every case (12 insns, 4 `vfmacc`, 1 `vle32`, 0 spills), which is the point: K-blocking can only move
cache behavior, never the inner-loop instruction mix.

Both verdicts then confirmed on silicon (128³, `NR=32`, one locked board session):

| point | ticks | instret | loop insns | digest |
|---|---|---|---|---|
| 2-D, un-blocked | 13,842 | 275,066 | 12 | `7bde3077…` |
| `k_block` KC=32 | 13,677 | 278,310 | 12 | `74b4b287…` |
| `k_block` KC=64 | 13,639 | 276,992 | 12 | `c6e411f7…` |
| **`k_block` KC=128** | **13,579** | **275,055** | 12 | **`7bde3077…`** |
| `unroll_m` MR=2 | 33,388 | 473,127 | 6 | `7833b2e3…` |
| `unroll_m` MR=4 | 33,794 | 478,242 | 6 | `a06a1564…` |
| `unroll_m` MR=7 | 33,507 | 478,571 | 6 | `46e48729…` |

### A byte-identical digest is a free noise control

The `KC=128` row emits the **same code as the un-blocked row** — same digest, same instret. So the
two rows are the same binary measured twice, and their 13,842 vs 13,579 spread is **pure measurement
noise: 1.9%**, established without a single extra run.

That retires the `k_block` result. Both blocked points (13,677, 13,639) sit *inside* that band, and
are in fact *slower* than the identical-code control — so the 1.2-1.5% that looked like a consistent
cache-blocking win is noise, exactly as the untouched inner loop predicted. Read in isolation, three
descending numbers (13,842 → 13,677 → 13,639) are a tempting trend line.

**Rule:** when a sweep contains a point whose emitted code is byte-identical to another, measure it
anyway — it costs one run and calibrates the noise floor that every other delta in the sweep must
clear. Where the parameterization does not hand you one for free (`KC=K` did), construct one.

`unroll_m` on silicon closes its own case: ~33.4-33.8K ticks against the 2-D form's 13,842, i.e.
**2.4x slower**, and flat across `MR` ∈ {2, 4, 7} to 1.2% in time and 1.2% in instret while the
digests all differ. Changed code, unchanged economics.
