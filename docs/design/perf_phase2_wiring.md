---
title: "Design: wiring phase 2 — what the performance search can measure, ask, and refuse"
kind: design
status: current
last_verified: 2026-09-08
owner: gemmini-perf-bench
related: [compiler_plane, expert_gap_attribution, command_stream_reorder_emitter]
code_refs:
  - merlin/experiments/gemmini_perf_bench/scripts/perf_agent_stage.py
  - merlin/experiments/gemmini_perf_bench/scripts/functional_coverage.py
  - merlin/experiments/gemmini_perf_bench/scripts/perf_model.py
  - merlin/experiments/gemmini_perf_bench/scripts/perf_holdout_corpus.py
  - merlin/experiments/gemmini_perf_bench/scripts/perf_snapshot.py
  - merlin/experiments/gemmini_perf_bench/scripts/perf_suite.py
  - merlin/experiments/gemmini_perf_bench/scripts/run_agentic_perf_experiment.py
  - merlin/experiments/gemmini_perf_bench/scripts/run_global_perf_experiment.py
  - merlin/experiments/gemmini_perf_bench/scripts/run_paired_perf_bench.py
  - merlin/python/merlin/perf/handshake.py
  - merlin/python/merlin/perf/global_planner.py
  - merlin/python/merlin/perf/agent_guidance.py
  - merlin/python/merlin/perf/artifact_activity.py
  - merlin/python/merlin/perf/command_buffer_diagnostics.py
  - merlin/python/merlin/perf/execution_policy.py
  - merlin/python/merlin/perf/model_placement.py
  - merlin/python/merlin/perf/movement_balance.py
  - merlin/python/merlin/perf/phase2_portfolio.py
  - merlin/python/merlin/perf/whole_model_report.py
  - merlin/python/merlin/xdsl_dialects/lowering/global_plan.py
  - merlin/python/merlin/xdsl_dialects/lowering/global_plan_emission.py
  - merlin/python/merlin/runtime/program.py
  - merlin/python/merlin/perf/roofline.py
  - merlin/python/merlin/targetgen/coverage_report.py
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

## Functional certification is explicit sampling, not a size heuristic

The former `sim_hint` / `plan_cert_tier` path is not the campaign policy. It was removed after its
cost fit could not price the active corpus and therefore withheld every cycle-accurate cell. The
paired campaign instead requires an exact tuning certificate for every selected member; a missing
or unpriced member is refused before authoring rather than silently promoted or dropped.

The separately expensive public+hidden **functional** equivalence certificate is exact by default.
An operator may explicitly request `functional_coverage.py`'s stratified policy. It groups the
frozen cohort by operation plus semantic attributes, selects one member per stratum, records every
selected and unsampled identity, and states that the result proves output agreement only for those
selected workloads. A supplied cost model is accepted only when it names the same reference engine
and exact engine pins; if a stratum is only partly priced, the whole stratum falls back to an
input-element proxy. Producer and consumer independently re-derive the strata and selection, and
both reject missing strata, changed selection, and certificate extras.

This sampling policy does not widen timing authority. The tuning and held-out performance paths
still execute through the certificate-bound GSIM engine on the exact campaign cohort.

## One claim per sealed campaign

`perf_suite.py` is the repository-owned launcher. It partitions the explicit member list by analyzer
claim before authoring, seals one private source snapshot for the suite, and gives every claim its own
campaign, three predeclared trials, cohort, replicate schedule, held-out reveal, and result seal.
Static preflight and a real baseline execution preflight both finish before paid authoring begins.
`run_agentic_perf_experiment.py` coordinates each claim and invokes `run_paired_perf_bench.py` for
the actual baseline/candidate cells. New candidate records name that paired runner as their consumer;
schema-v3 records already sealed with the former `run_perf_bench.py` name remain readable.

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

## Macro/global loop now exposed automatically

The Phase-2 authoring loop now treats the complete model as the objective and a reduced execution as
one calibration instrument. The shared machinery is target-neutral: it contains no accelerator name,
opcode spelling, or inferred resource role.

The compiler/runtime side has four explicit layers:

1. `GlobalPlan` represents selected multi-operation regions and the representation transitions between
   them. The planner selects an exact, non-overlapping cover of the model rather than independently
   accepting locally attractive regions.
2. Target-neutral command-buffer analysis derives work, declared boundary movement, representation
   directives, synchronization, residency/fusion findings, and repeated-model projections. Occupancy,
   physical traffic, and executed conversions remain `UNKNOWN` unless an event adapter or counter
   receipt establishes them.
3. `GlobalPlanEmission` is a checked target-adapter seam. Its receipt must account for every logical
   dispatch exactly once, every selected region and materializing transition, and bijective model
   input/output mappings with unchanged external shape and dtype. `build_program` uses the emitted
   program only after that verification; without an emitter the plan remains explicit shadow analysis.
4. `whole_model_report` refuses promotion when compute, movement, occupancy, overlap, or encoding
   evidence is incomplete. It ranks those gaps as optimization opportunities instead of converting
   absence into zero.

Every agent round now receives these host-generated fields in `STAGE_CONTEXT.json`:

- `initial_whole_model_analysis`: frozen-baseline versus live-candidate command buffers, structural
  findings, and a ranked `optimization_brief`;
- `automatic_optimization_inventory`: real Python AST symbols and manifest command consumers, plus
  structurally verified, author-declared `optimization_surfaces` mapping a semantic lever to an exact
  file and symbol; the subsequent emission comparison tests whether the declared effect occurred;
- `reduced_global_profile`: the frozen reduced witness, why it was selected, and the
  `profile-reduced-global-witness` broker action;
- `iteration_measurement_contract`: at most 600 simulator seconds, one unmeasured warm invocation,
  exactly one measured invocation, total compute cycles as the primary metric, and only the movement,
  occupancy, overlap, and encoding counters needed to explain it.

The objective is declarative rather than inferred from whichever model is smallest. A public model
capsule may set `performance.global_objective: true`; exactly one such model wins selection and more
than one is a refusal. For an older immutable Phase-1 snapshot without that metadata, the target
experiment must name `performance.global_objective_capsule`; absence, disagreement, or a name outside
the sealed snapshot is a refusal. Gemmini names `M2_microvit_gemmini`, the complete reduced model that
the existing campaign priced at roughly four L3 seconds. It can no longer silently substitute the
smaller `M3_host_island_seam_gemmini` seam harness. The full-size ResNet-50 capsule added after that
92/96 snapshot is marked for a future snapshot and resource-excluded from mandatory L3, but is not
retroactively treated as Phase-1-qualified evidence. Thus an actual frozen model graph drives the
current loop without rewriting Phase 1 or placing a multi-hour simulation in the search loop.

The agent can refresh the source map with `inspect-optimization-surfaces`, rerun complete-model
emission with `analyze-whole-model`, and request the warm reduced profile only when occupancy or
overlap is the deciding unknown. The reduced result is labeled as calibration and cannot be promoted
as an end-to-end result. In-round complete-model analysis is optional screening: after a clean Codex
exit, the host snapshots the exact submitted candidate and performs the mandatory final static
portfolio analysis under its independent full-graph deadline. This keeps each authoring round and its
broker at or below 600 seconds even when multi-model compilation needs longer; the post-authoring path
uses the same answer-masked compiler worker, memory admission, and no-full-model-simulation policy.
The declared whole-model objective, every evaluation/mixed-lane harness, and the witness selection
cannot change after candidate evidence is observed; formal promotion still evaluates the complete
sealed cohort.

L3 is deliberately sparse. Per round, the broker permits one optional occupancy profile and two
tuning GSIM calls: at most one exploratory promotion check and one call reserved for the exact bytes
being sealed. All other iterations use the no-simulator whole-model analysis, command-buffer analysis,
and source-surface inventory. This is not because the cheaper signals are treated as timing truth:
the analyzer records that neither command-buffer cost nor the instruction-count tier reliably orders
same-workload schedules. Cheap evidence may reject added work, movement, barriers, conversions, or an
inert edit; only the sparse timing gate may promote a remaining candidate.

Full-size execution is outside the search loop and is not a Phase-2 prerequisite. It is optional
post-freeze validation for environments that have FireSim. If FireSim is explicitly used, the queue
owns exactly this lifecycle for each execution:

```text
firesim kill
firesim infrasetup
firesim runworkload
firesim kill
```

### Fast four-model evaluator and accuracy gate

The full-model authoring path has an optional host-owned evaluator that runs serially over exactly
four content-addressed portfolio members. Each member has a 60-second-or-smaller wall budget. The
binding explicitly says `host_analytical_only`, `serialized_one_model_at_a_time`, and
`full_model_simulation_allowed: false`; it also pins the evaluator implementation and calibration
receipts. This path cannot invoke L3, FireSim, or a complete-model simulator.

The standard held-out quality schema gives the classification member at most 0.7 percentage-point
top-1 degradation. Each other member requires cosine similarity at least 0.99 and normalized RMSE
at most 0.02. All four corpus members need exact content identities. If even one corpus is absent,
the report switches to `exact_only_fallback`: approximation is disabled, while exact compiler
transformations and static analysis can continue.

For both baseline and candidate the provider reports a conservative cycle interval, physical bytes
moved, an explicit occupancy/overlap timeline, encoding-conversion count/bytes/cycles, a composed
physical roofline with declared resource floors, and calibration risk. Coverage is a first-class
topological record: supported source work placed, the exact connected-region partition and largest
region, typed host islands, and boundary crossings/bytes. `UNKNOWN` never becomes zero.

The shared gate checks every known objective for per-model Pareto regression. A loss in any member
rejects the candidate rather than being averaged away. Increased accelerator coverage alone is not
a win: at least one model must robustly improve cycles, movement, utilization, overlap, conversion,
or boundary cost without regressions elsewhere. Unlike-model cycle counts are never summed; the
only summary is the dimensionless geomean of conservative per-model speedups.

The same report is sealed into each iteration, surfaced through `portfolio_action_digest`, and fed
back to the authoring agent. Recommended levers are intersected with the host-frozen edit contract,
so a candidate cannot grant itself new compiler files or symbols. The movement term may be calibrated
with `movement_balance`, whose controlled multi-size fit carries its one-sided roofline licence; a
missing or invalid fit stays missing rather than manufacturing a bandwidth.

Two limitations are deliberate and visible. First, the frozen Phase-1 package inspected on 2026-09-06
contains 343 indexed AST symbols but no `optimization_surfaces`; changing that sealed manifest would
invalidate its digest. The current loop therefore exposes its full source index and tells the agent to
add structurally validated semantic mappings, while newly generated packages are prompted and
schema-enabled to declare them initially. Second, the shared emitter protocol does not invent target instructions. Each
target adapter must implement legal plan emission before a global plan can replace executable
dispatches. Until that happens, the analysis is useful but no compiler speedup is claimed.

## Live full-model audit: the first blocker is composition, not a tile

The joined analyzer was run on 2026-09-06 against the immutable 92/96 Phase-1 compiler and the
declared `M2_microvit_gemmini` objective. It completed in about 2.5 seconds without L3 or FireSim. The
result is a refusal, not a cycle estimate:

```text
CPU-lane program for @forward needs about 666948 straight-line element evaluations,
past backend 400000 budget; emitted kernel is single-block so no loop to roll them into.
```

The command buffer contains no executable commands, and the 114-byte lowered artifact is an empty
target function. Both are now reported as `declined`/`UNKNOWN`; neither is allowed to masquerade as
zero arithmetic, zero movement, complete encoding coverage, or successful residency. The same cheap
scan found this full-model emission failure for the frozen M0, M1, M2, and M3 interfaces. Phase 1 was
not rerun and the frozen package was not edited.

The refusal still carries useful compiler evidence. M2 has 132 placed regions: 12 in one declared
lane, 120 in the other, with 24 adjacent lane transitions. Thirteen captured contractions account for
124,928 structural MACs. The compiler places 118,016 MACs (94.4672%) on its `on_mesh` lane and 6,912
(5.5328%) on its `scalar_rvv_lane`; all thirteen contractions fall in the target-derived
`fits_double` capacity regime. This does **not** mean the model is 94.5% accelerated: non-contraction
work, boundary movement, and the declined host program remain unpriced. It means the next compiler
change should address rolled host lowering and/or a larger legal fused/offloaded cover before another
micro-schedule search.

The no-simulator loop now joins four evidence levels on every re-emission:

1. captured graph work and compiler-declared lane placement, weighted by exact contraction MACs;
2. command-buffer movement, materialization, synchronization, representation and residency shape;
3. target-artifact instruction roles obtained from the selected target's own facts, plus program CCA;
4. decoded trace conformance and the existing exact redundant-residency-reload detector.

`gap_coverage` tells the agent whether each mechanism has both evidence and a verified editable AST
surface. Its rows cover whole-model placement, arithmetic lowering, encoding/layout, movement,
cross-operation residency, fusion/host boundaries, loop offload, latency hiding/double buffering,
synchronization, and capacity/contention. A row is `ready` only when both sides exist; missing evidence
or a missing edit surface remains explicit. The Phase-1 package currently declares no semantic
surfaces, so the agent must add candidate-owned mappings that pass AST ownership and editable-CCA-axis
validation.

Three gaps remain before this is an end-to-end optimizer rather than a complete diagnostic loop:

- the candidate compiler needs a `TargetPlanningAdapter` and legal `GlobalPlanEmitter` that turn the
  shared exact-cover plan and representation transitions into executable dispatch;
- host-lane codegen needs rolled loops (or equivalent structured lowering) so a real mixed model does
  not expand past the straight-line budget;
- dynamic contention and realized latency hiding still require a target resource-event adapter and,
  only when they decide between candidates, the bounded warm reduced profile.

These are protocols, not Gemmini assumptions. Lane names, resource roles, instruction meanings,
capacity, and legal encodings come from the selected compiler/target adapters. A new accelerator can
reuse the planner, activity timeline, CCA ownership, emission accounting, report, and experiment
policy and supply only its facts and legal emitter.
