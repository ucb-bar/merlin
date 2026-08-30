---
title: "Design note: what a performance query actually costs, and what is scarce"
kind: design
status: current
owner: core
last_verified: 2026-08-29
related: [expert_gap_attribution]
code_refs: [merlin/python/merlin/targetgen/tier_policy.py, merlin/python/merlin/targetgen/program_oracle.py, merlin/python/merlin/targetgen/capsule_runner.py, merlin/python/merlin/kernels/measurement.py, merlin/contract/hardware_pins.yaml]
---

# What a performance query costs, and what is actually scarce

A performance layer that rations expensive evaluations has to know which evaluation is expensive.
The obvious assumption — that cycle-accurate simulation is the scarce resource and everything else
is free — is **false on Atlas**, by two to three orders of magnitude. This note records the
measurement, the decision it forced, and two corrections it produced.

## The measurement

42 samples per tier (14 capsules x 3 repetitions), replayed through
`capsule_runner.oracle_adapters("atlas")` with the real command buffer and `kernel.S` of each
capsule. Halt cycle counts reproduce the historic run exactly, so these are faithful
re-executions rather than stubs.

| tier | oracle | median | min | max | cost per halt cycle |
|---|---|---:|---:|---:|---:|
| L3 | mlc arc cosim | 3.68 s | 0.44 s | 9.43 s | 3.63 ms |
| L4 | Verilator `VAtlasCore` | 0.276 s | 0.044 s | 0.673 s | 0.255 ms |

There is **no build step**. Assembling the agent's `kernel.S` with stock `llvm-mc` +
`llvm-objcopy` takes a median of 5.7 ms — 0.15% of an L3 query. The arc shared object and
`VAtlasCore` are prebuilt; nothing compiles per query. Cost is linear in halt cycles.

Producing the `kernel.S` that those queries evaluate costs, per run,
900-31,061 s of wall clock (`cost_time_toolcalls.yaml`; the best Atlas run to date,
`merlincirct_atlassg1`, took 31,061 s / 222 M tokens / $147 notional). A complete 14-capsule
L3+L4 sweep is ~68 s of serial wall.

> **At capsule scale, the oracle is 0.2-0.4% of the cost of producing one Atlas datapoint.**

## The regime this measures, and the regime that matters

That ratio is real but it is **scoped to the corpus it was measured on, and that corpus is
toy-scale.** The largest tensor anywhere in the Atlas capsule corpus is 5,400 elements; nearly
every capsule is a 32x32 tile halting in 178-2,614 cycles. Wall time was small because the work
was trivial, not because the simulators are fast.

The rates say so plainly. **0.255 ms/cycle is ~4,000 cycles/s**, which is *slow* for Verilator;
arc's 3.63 ms/cycle is ~275 cycles/s. Neither is a fast simulator. On a 32x32 tile that does not
matter. On a model layer it decides everything: the same linear law that gives a 0.28 s query at
1,090 cycles gives hours at layer scale.

**Pricing a scaling behaviour from a single regime is the error this repo has already recorded
once** — a tiled matrix unit cannot be priced by one `macs_per_cycle`, because cost is a rate term
plus a fixed overhead and one point cannot separate them. Taking a toy-scale ratio as the budget
verdict repeats it at the level of the experiment design.

## The decision

**Which unit is scarce is regime-dependent, and the honest answer needs both regimes measured.**
At capsule scale the synthesis call dominates by two to three orders of magnitude and simulation
is free. At layer scale the same law projects hours per arc query, and the comparison may invert.
The verdict is therefore recorded as: *scarce unit at capsule scale = synthesis call; scarce unit
at layer scale = pending measurement*, and the search machinery must read the unit from
measurement rather than bake one in.

What generalizes is the rule, not the number: **measure the tiers in the regime you will actually
work in, then ration whichever is scarce there** — never assume, and never extrapolate a cost
ratio across two decades of problem size without saying so.

The same machinery on Radiance faces GSIM at ~115 s and Verilator at ~45 min per kernel, where
simulation is scarce even at capsule scale.

## The corpus cannot answer the question it is being asked

The deeper finding behind the regime problem: **there is no layer-scale Atlas workload anywhere.**
The 21 perf-eligible capsules are 32x32 tiles; the 25 shipped npu_model Programs bind a static
`.S` at class-definition time with hardcoded immediates, so a new shape is a hand-written
assembly file; and the "full model shapes" the program docstrings cite ([241,960], 1024x3072,
(50,720)) have no program, no golden and no run. `ParameterizedMatmul32x32x32Program` and its
siblings are referenced only by merlin's orphaned `dse/calibrate_npu.py` and **do not exist** in
npu_model at all, which is independent confirmation that that file is dead.

So layer-scale performance work has a prerequisite: the layer-scale workloads must be generated,
and the only thing on hand that can generate them is merlin's own emitter, which produces
`kernel.S` word streams for an arbitrary command buffer. That makes "predict cycles at a shape
merlin emitted" the primary experiment and "recover a fraction of a shipped reference" the
secondary one, because off the 21-capsule corpus **no reference implementation exists to recover**.

## Two numbers that were wrong, and why

**The ~136 s figure is a different target.** `merlin/experiments/capsule_bench/harness/.oracle_timing.json`
records `{"verilator_per_capsule_s": 131.3, "config": "GemminiRocketConfig"}` — the gemmini
chipyard-Verilator tier. No Atlas arc query came within 14x of it.

**The 24.5 s "arc cosim" figure in `tier_policy.py`'s docstring is host contention, not oracle
work.** The historic grade ran 16 workers (`capsule_grade.default_grade_workers` ->
`min(nproc-2, 16)`). On identical capsules and identical cycle counts, the contended median is
23.4 s against a serial median of 3.7 s — **6.3x inflation**. Pricing machinery off that number
prices it off the scheduler.

The general lesson: a per-query cost measured inside a parallel grade is a throughput figure
wearing a latency figure's clothes. Record the concurrency alongside the number, or the number
does not mean anything.

## Correction: the cheaper tier is the better one

L4 Verilator is **13x faster than L3 arc** and is the higher-fidelity substrate:
`derived_from_rtl: true`, `fidelity: elaborated_rtl`, against arc's `derived_from_rtl: false`,
`fidelity: rtl_derived_model`. Across all 21 perf-eligible capsules the two report **identical
cycle counts**, capsule for capsule.

Atlas's declared authority (`cycles_from: arc_program`, `cycles_tier: cycle_model`) therefore
understates what is available: the same numbers are obtainable at measurement tier `rtl`, at a
thirteenth of the cost. The authority should be re-declared against L4, which upgrades every
Atlas performance claim from `cycle_model` to `rtl` and makes it cheaper to produce.

## What L2 is not

`program_functional_adapter` runs (median 1.35 s) but is **not** a usable cheap screen: its cycle
counts disagree with both RTL tiers on the same program (AT2: 3081 against 1090), matching the
documented functional-core decode ambiguity. Recorded as `usable_as_screen: false`. A tier that
returns a number is not thereby an authority for it.

## Persistence gap

`tier_policy.record_cost` is process-local: `_COST` is a module dict behind a module lock, and
the module performs no file I/O. Every grader process starts uncalibrated and re-pays
`tier_order`'s unmeasured-tiers-first probe. On Atlas that is one ~3.7 s query per process and
does not matter; on a target whose expensive tier is the 131 s gemmini Verilator it is 131 s per
process. Persisting per-`(target, tier)` sample lists under
`artifacts_dir()/capsule-bench/<target>/tier_policy/` would close it without changing any
signature. `.oracle_timing.json` is not the place — it is written by `readiness_check` for a
different consumer (driver timeouts) and holds one tier for one config.
