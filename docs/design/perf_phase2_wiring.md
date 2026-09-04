---
title: "Design: wiring phase 2 — what the performance search can measure, ask, and refuse"
kind: design
status: current
last_verified: 2026-09-04
owner: gemmini-perf-bench
related: [compiler_plane, expert_gap_attribution, command_stream_reorder_emitter]
code_refs:
  - merlin/experiments/gemmini_perf_bench/scripts/perf_agent_stage.py
  - merlin/experiments/gemmini_perf_bench/scripts/perf_model.py
  - merlin/experiments/gemmini_perf_bench/scripts/perf_holdout_corpus.py
  - merlin/experiments/gemmini_perf_bench/scripts/run_perf_bench.py
  - merlin/python/merlin/perf/handshake.py
  - merlin/python/merlin/perf/roofline.py
  - merlin/python/merlin/targetgen/coverage_report.py
  - merlin/python/merlin/targetgen/cert_cost.py
---

# Wiring phase 2

Phase 1 generates a functional compiler and is frozen. Phase 2 takes that frozen submission and
optimises it for speed, and this note records what phase 2 can now measure, what it can ask, what it
refuses, and — the part worth reading — the several places where a check quietly reported success
because it could not run.

Every number below is measured, from campaign `20260903T222654Z` (three trials) and the frozen
phase-1 run `merlincirct_arm4_func_20260902_codex5_evidence_gsim` unless stated otherwise.

## The recurring defect: a check that could not run reported success

Four analyses refused on a conformant target, and in each case the target's own facts held the
answer while the consumer was looking somewhere else — usually at *another target's spelling*.

| analysis | presented as | actually was |
|---|---|---|
| ISA / `machine_facts` | "this target ships no ISA definition" | the decode table was in the RTL facts; nothing consulted it |
| `fill_drain_depth` | "the circuit could not be read" | the wrong target's module names were passed in |
| `vector_term` | "no 256-bit VPU data row in manifest" | the unit name was never derived from the target's own units |
| accumulate datapath | manifest declared operand format only | the RTL carried operand **and** accumulator with evidence strings |

The fix in each case is the same shape: derive from `facts`, so the answer changes when the RTL
changes. Two of these deserve their own account.

### Fill/drain depth: a delay line the emitter did not name

The upstream pass measures the array's pipeline depth as the length of a register chain whose **own
name** contains `valid`. That is a naming convention, not a structural fact. On one design firtool
named the chain `%r_256_0 … %r_1115_0` and put the word `valid` only on the *signals each stage
samples* — so the pass reported "no output-valid delay-line found in @Mesh" for a circuit holding
257 registers of exactly that delay line.

`merlin/python/merlin/perf/handshake.py` now walks the path instead of matching a name: a stage is a
register, and the valid signal crosses a submodule through the ports whose names carry the
handshake's own `valid` (port names are the *design's* vocabulary; register names are the emitter's
invention). Measured:

```
gemmini   dim=16  depth=17   law systolic_2d predicts 30  -> REFUTED for this design
atlas     dim=32  depth=62   law systolic_2d predicts 62  -> agrees
```

The law that holds for one array is wrong by 76% on the other. A model that swept with it everywhere
would carry that error into every small-tile estimate, because fill/drain is an *intercept* — paid
once per weight reload, dominant exactly where tiles are small.

### `must_accelerate` cannot fire where fallback actually happens

`coverage_report.py` graded the offload demand as:

```python
violated = must and eligible and not accelerated
```

The `eligible` conjunct disarms the check precisely where it is needed. A region is ineligible
*because* the hardware cannot run it — and that is the same region that will quietly drop to the
host. Measured on an int8-only array: all twelve bf16 capsules declare `must_accelerate: true` and
are ineligible on dtype, so `violated` is `False` for every one of them, forever. **Zero of the
twelve are graded at all** — the frozen phase-1 run has 213 results over 44 distinct capsules, every
one int8.

Meanwhile the compiler is equally quiet: `--convert-iface-to-gemmini` on a bf16 matmul exits **0**
and emits the `linalg.matmul` unchanged. No gemmini op, no diagnostic. It then lowers to a scalar
bf16 loop on the CPU.

The fallback itself is *correct* — the RTL is `input i8` / `accumulator i32`, with `@PE` ports
`i8 -> i20`, so there is nowhere for a bf16 operand to live. What was missing is that it was
unobservable. A `declined_offload` state now records it: reported, never scored, because failing it
would punish a conformant submission.

## Where the time actually goes

Across three trials, 4,287 s of wall time:

| | seconds | share |
|---|---|---|
| agent reasoning | 2,262 | 53% |
| GSIM measurement | 1,984 | 46% |
| compilation (all four actions) | 43 | 1% |

18 measurements, mean 110 s. Compilation is free; measurement and thinking split the run.

## Why the loop cannot move to the cheap tier

The obvious saving is to judge candidates at L2 (spike) instead of L3 (GSIM). The data forbids it.
Over 247 points carrying both tiers:

```
L3/L2 ratio            min 3.11   median 5.69   max 14.67
WITHIN-capsule order agreement, L2 vs L3:   483/926 = 52.2%
```

**52% is a coin flip** for the exact comparison the search makes — same capsule, two schedules.
Spike counts retired instructions and cannot see mesh occupancy, DMA overlap or scratchpad
pressure, which is what a schedule change moves. L2 is a correctness tier, not a timing tier.

So the saving is elsewhere: stop spending a 110 s measurement to discover that a candidate is worse.

## Letting the search ask a question

Of the fifteen broker actions the agent was given, **none could ask anything**: four compile the
candidate, ten probe the environment, one is the measurement. `merlin.perf.differential` was named
four times in the prompt as a GO requirement with no way to invoke it. Every derived number reached
the agent only as a field the host had already written.

That made the measurement the sole judge, so a losing candidate cost exactly as much as a winning
one. Measured: excursions of **+5.9%** and **+11.1%** burned ~220 s of oracle time.

`analyze-command-buffers` prices the candidate's **own** emitted artifacts — work volume, both
ceilings, and a differential verdict. It reads no oracle, no golden and no holdout, so it costs
nothing and can leak nothing. In exchange it is ordering-only: it reports the differential basis
(`EXACT`, `ORDERING_ONLY`, `REFUSED`) and never an absolute cycle count, which is the licence a
corpus-calibrated model actually has. A candidate whose *work volume* differs is called out
separately, because a cycle delta there is not a schedule comparison.

## Two ceilings, and why the achievable one is the target

- **Structural**: 256 MAC/cycle, from `facts.arrays` (16x16 mesh times the MAC idiom). Unreachable
  by construction; kept as context.
- **Achievable**: 80.01 MAC/cycle = 31.3% of structural, via `envelope.Peak.observed_ceiling`, which
  re-falsifies against every sample. `perf/roofline.py` admits only this kind (`n_samples >= 4`,
  `is_ceiling`), so a nameplate peak is structurally excluded.

Attainment is judged against the achievable bound. Judging against the structural one would never
fire; judging against a nameplate would stop the search early for a reason about arithmetic rather
than about the machine.

## Rationing the expensive tier

The cycle-accurate tier used to be selected by a string a generator wrote down —
`"L2+L3" if macs <= 2_000_000 else "L2_only"` — and, worse, the paired bench ran
`setdefault("sim_hint", "L2+L3")`, so a kernel nobody had labelled took the **most expensive path by
default**. That is the one direction that cannot be recovered from: a wrongly-cheap plan
under-certifies and says so; a wrongly-expensive one silently spends the budget the rest of the
corpus needed.

`plan_cert_tier` derives the decision from `cert_cost`, whose fit is measured on the target's own
certified runs and which refuses rather than guessing. Cheapest-first within a declared budget, with
a recorded reason for every kernel held back. Measured on the 31-member corpus: an unbounded budget
certifies all 31; a 600 s budget certifies 20 and names the 11 it dropped.

The fit itself is a finding: `69.2 s + 3.61 ms/cycle, R^2 = 0.016` over 55 samples. Cycles explain
about 2% of certification cost — it is almost entirely fixed overhead. **The number of certified
members dominates, not their size**, which cuts against sizing by MACs at the root.

## What the corpus does and does not represent

All 31 performance capsules are matmul-family, rank-2:

```
ops:    matmul 15,  resident_reuse 12,  fused_matmul_bias 2,  bias_add 2
shapes: distinct M (= N): {16}          <- ONE value
        distinct K: 16, 32, 64, 128, 2048, 4096, 4112, 6144, 8192, 8208, 12288, 16384
```

Zero conv2d, zero attention, zero pooling, zero normalization — all of which phase 1 grades. Set
against phase 1's measured headroom, the corpus is inverted:

| workload | mac/cycle | % of achievable | headroom | in the perf corpus |
|---|---|---|---|---|
| conv2d | 2.67 | 3.3% | **29.9x** | no |
| conv + maxpool | 5.30 | 6.6% | 15.1x | no |
| small matmul | 13.52 | 16.9% | 5.9x | yes |
| mlp | 58.94 | 73.7% | 1.4x | no |
| deep-K matmul | 80.01 | **100%** | 1.0x | yes |

The corpus optimises the operation already sitting at the achievable ceiling and does not measure
the one with thirty times the headroom. That, and not agent skill, is the likeliest explanation for
three independent trials converging to 43.9–44.3% of attainable and stalling.

Generalisation follows the same shape: K spans three orders of magnitude, M and N are a single
point. Nothing in the corpus can say whether a schedule that wins at 16x16 still wins at 256x256.
Inter-layer scheduling is untested (no capsule spans two different ops), and 27 of 31 capsules claim
`DIFFERENTIAL` while the differential analyzer is not wired into the measurement path.

## Reproducibility lessons

- **Runs execute from an immutable snapshot.** Editing a treatment source mid-run killed two
  campaigns with `NO-GO: pinned telemetry implementation changed`. Each campaign now copies the
  harness and records its digest.
- **Pin only committed bytes.** A GSIM certificate was sealed against a `cxxwrap.sh` that existed in
  **no commit** — someone's working-tree version. A later checkout replaced it and every launch
  refused. The pinned bytes are unrecoverable.
- **A cycle-accurate simulator's output is a property of the RTL, not of the host compiler.**
  Rebuilding the emulator yields a binary differing in 23,661 bytes of relocation layout, and the
  same ELF through both reports `cycles=208128` on each — 52 of 53 output lines byte-identical, the
  only difference being wall-clock seconds. So re-certifying costs no fidelity.
- **A generator and its schema drift silently.** The holdout generator stamped revealed capsules with
  a `source_role` the capsule schema does not define, so the run died at the reveal step *after* all
  three candidates had sealed and all three functional regrades had passed. The two are now compared
  directly in a test rather than trusted to stay in step.
