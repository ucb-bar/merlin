# Tuning after the F2 runs

The proposed additional decisions and their actual implementation status are
in [OPTIMIZATION_KNOBS.md](OPTIMIZATION_KNOBS.md). ResNet selects the initial
profile-driven experiments; every optimization must apply by IR semantics and
capabilities. New full-shape source regions/fixtures are now prepared in
`out/artifacts/handoff/gemmini-firesim-dse-round2/v1/latest/`; see
[FIRESIM_ROUND2_HANDOFF.md](FIRESIM_ROUND2_HANDOFF.md) and that product's
readiness record before execution. It contains 18 full-shape ResNet regions,
21 structurally distinct contraction workloads with 10 tuning owners, and an
explicit TinyLlama coverage/missing-input roster. No new FireSim result is implied.
The published branch carries a complete restorable copy; use the fresh-machine
entry point in [FIRESIM_ROUND2_HANDOFF.md](FIRESIM_ROUND2_HANDOFF.md).

New runs follow [CURRENT_RUN_POLICY.md](../../numerics/CURRENT_RUN_POLICY.md).
The supplied bitstream sets the expected DDR3 memory model. Before more paid
DSE, qualify [ADAPTER_CONTRACT.md](ADAPTER_CONTRACT.md): the October 26 F2
adapter completed slow candidates and filtered them afterward; its total-cycle
cap did not implement the user's live active-cycle cutoff. No TVM optimizer
monkeypatch or hardware-memory reconfiguration is part of this audit.

## Next ResNet DSE from the raw F2 profile

The supplied `tvm-chunk-raw-20261026.tar.zst` has SHA256
`b4922d8504b3764409bc6168670cbfb18c463306ffe28539dae6b2949ae687e3`.
The retained audit is `out/artifacts/audits/gemmini-f2-tvm-followup/v1/latest/`.
Its `analyze.py`, `audit-summary.json`, `resnet-operators.csv`,
`resnet-regions.json` and `winning-knobs.json` bind the raw receipts/traces to
the actual prepared graph. See its `README.md` for the complete findings.

The default/tuned full-model pair repeated exactly: 3,028,578,593 versus
3,027,080,096 active cycles, a **0.0495% reduction**. The library still tunes
only the classifier: **1/68 extracted tasks, 1/67 eligible, 1/21 contractions**.
The earlier 24/672 figure describes that local classifier's declared compact
space before pruning, not ResNet-wide coverage or FireSim search coverage.

The instrumented model has 3,031,383,095 active cycles. Its 128 operator
intervals sum to 3,031,358,387, leaving 24,708 cycles outside those intervals:

| Placement | Operator cycles | Share of operator sum |
| --- | ---: | ---: |
| CPU functions | 2,141,496,755 | 70.64% |
| Gemmini functions, including their host wrappers/epilogues | 889,861,632 | 29.36% |

This is function placement attribution, not a hardware-stall breakdown.
Host-only quantization/layout/im2col and residual/epilogue functions account
for most elapsed cycles, including any cache/DRAM, instruction-fetch and
dependency stalls. It does not prove CPU compute saturation. The exact ELF
audit in `out/artifacts/audits/gemmini-resnet-elf-attribution/v1/latest/` verifies
the 128 timed calls, host-only call graphs and Gemmini completion-fence calls.
Serialized/idle-device phases are an inference under the platform's fence
contract, not measured array utilization. Profile available CPU retirement/
memory-hierarchy and Gemmini activity/wait counters before prioritizing new
host scheduling work. The classifier
itself is only 712,174 cycles in the profiled tuned graph (0.0235%). Improving
only device functions cannot address the 70.64% host portion; with host work
fixed, even eliminating all their costs saves at most 29.36% of this profile.

Use **real connected regions with full shapes**, selected by this profile:

| Region | Package calls, half-open | Measured cycles | Share |
| --- | --- | ---: | ---: |
| Stem, including quantization/input preparation/pooling | [0,4) | 482,857,415 | 15.93% |
| First bottleneck, including projection shortcut and residual merge | [4,13) | 403,898,232 | 13.32% |
| Larger early prefix, including three bottlenecks and fused boundary preparation | [0,27) | 1,564,004,578 | 51.59% |

Package-call numbers are **not semantic Relax binding indices**. The region
manifest identifies actual symbols/live-ins/live-outs; map them through the
semantic inventory before slicing. Produce independent boundary inputs and all
live-out references from the same frozen numerical release. Do not shorten K,
crop the image or remove residual/layout work to make the proxy faster. The
earlier preparation had actual source-region exports and independent
INT64-contraction fixtures for stem, first bottleneck and an early prefix,
checked through host LLVM. The early source prefix stops before the next stage's
quantization; its boundary differs from the fused historical package window
above. Current preparation expands to 18 regions with zero-error host checks
and a complete new-source baseline ELF that passed Spike controls. New FireSim
costs, deployment and model quality still need measurement.

1. Reconcile the existing ResNet capture with the current numerical policy.
   All 54 stored epilogue scale arrays repeat within groups of 16, and static
   activation scales are present. Actual source inspection also shows FP32
   bias added after INT32-to-FP32 conversion. Round-two preparation now supplies
   an actual INT32 path and all 54 bound proofs; compiled/quality gates remain
   separate. Preserve the old source/receipts.
   Publish the fixed-set FP32/top-1 comparison separately; a v00 logits check
   is not the new accuracy gate.
2. Start with the stem and first-bottleneck regions. Prioritize the existing
   full-shape contractions M/N/K = **12544/64/160** (stem, logical K=147 with
   packing padding), **3136/256/64** (expansion/shortcut) and **3136/64/576**
   (3×3). Default reuse groups are mostly one tile. Search M/N/K reuse,
   traversal, OS/WS, cache placement/hoisting and legal pipelining through
   stock MetaSchedule traces; do not prescribe a winning parameter tuple.
3. Keep calibration/rounding fixed. Hold invariant graph preparation fixed
   during the first operator study. Remove no-op decisions by structural/code
   equivalence; retain alternatives only when they change generated work.
   Only expose bias/readout fusion when numerical and hardware proofs hold.
   Never spend a trial toggling an option that the recognizer cannot apply.
4. Begin with **48 proposals globally across the five initial owner chunks**,
   batches of four: stem eight, first stage-one block sixteen, and the first
   block of each remaining stage eight. Inspect completed work, admitted labels
   and marginal gains before the optional 20-proposal ResNet continuation.
   Reserve 24 proposals for TinyLlama once its actual inputs are prepared; keep
   all campaigns within 96 proposals and the same shared four-hour deadline.
   This is a proposal budget, not a runtime prediction. Reuse all actual
   independent slots with shared physical locks and persistent managers.
5. Include a diagnostic region run with mcycle/minstret and verified available
   memory/activity counters, as described by the ELF audit's
   `next-profile-plan.json`. Keep its instrumentation overhead separate from
   unprofiled performance comparisons. Do not infer CPU compute bottlenecks
   or array utilization from function latency alone.
6. Validate winners on the **whole connected regions**, then the larger early
   prefix or complete model. Use a separate incumbent for each measured scope.
   Finish with unprofiled full ResNet baseline/tuned pairs and all-logit checks;
   do not publish sums of operator minima as an E2E gain. Full runs are feasible
   candidates here: the four archived unprofiled runs took 157.49–159.49 seconds
   each on that particular image/installation. Those observations are not a
   runtime guarantee for another manager or newly prepared release.

The current FireSim driver selects **contractions only** and fixes the graph.
It does not tune the dominant CPU-only quantization/im2col functions. Improving
that coverage requires exposing their loops to supported TVM CPU schedule rules
or adding proven graph/lowering alternatives, then measuring connected regions.
That is additional compiler work, not a knob already searched in this run.
Keep it separate from claiming that a larger Gemmini tile search solves host
cost. Use stock scheduling/search APIs and general IR-derived legality.

## TinyLlama follow-up and search evidence

The two rounds made **804 proposals plus nine default/seed measurements**.
There are 264/267/258 distinct proposal traces for decode K/V, prefill K/V and
down-panel across both rounds; 813 ELF hashes are distinct, which does not prove
813 distinct schedules. Reconstructing the compact decisions from the source
and every trace gives:

| Chunk | Distinct decision tuples | Declared compact combinations | Coverage before legality/code equivalence |
| --- | ---: | ---: | ---: |
| Decode K/V | 264 | 560 | 47.14% |
| Prefill K/V | 267 | 2,240 | 11.92% |
| Down-panel | 258 | 576 | 44.79% |

This denominator includes the 8/16/8 root-epilogue Gemmini sketch families,
divisor-based M/N/K reuse groups up to 64 tiles, and unroll 0/16. All observed
decision tuples fall within it. It precedes memory legality and code-equivalence
pruning and excludes other graph/placement/epilogue spaces. It is not coverage
of all legal code-distinct TVM schedules, full TinyLlama or model layers. The
reproducible calculation and source hashes are in `decision-coverage.json`.

Whole rebuilt chunks improve 804,251→449,424 (decode K/V, **1.79×**),
7,475,291→6,685,020 (prefill K/V, **1.12×**), and 893,044→635,920 (down-panel,
**1.40×**). The down-panel task improved again in the larger round:
173,881→168,979 cycles (**2.82%**). It retains K=5632 but only N=128 of the
parent N=2048 output channels; its exact record cannot match the parent shape.

These are historical dynamic-row/per-channel-weight probes. Prepare meaningful
chunks from the new frozen-scale release before further qualification. Keep
full-K projections and surrounding conversion/epilogue work; use full-output
down projection confirmation before reuse, then the actual 32+32 E2E session.
Full TinyLlama/PPL qualification remains NOT_RUN in this archive.

The search used stock EvolutionarySearch/XGB and a supported custom feature
adapter that applies stock PerStoreFeature before tensorization. Only **23/804
proposals** were admitted as successful runner labels. The other **781** completed
but were reported as dominated errors and stored with the stock 1e10 failure
cost. Thus `training_samples=256` does not mean 256 valid cycle labels. A trained
booster and late improvements establish that learned search ran, but do not
establish convergence or superiority to an equal-budget random search. Do not
change TVM's algorithms to manufacture such a claim. One API-conformance debt
remains in the legacy backend: it puts raw cycles in `RunnerResult.run_secs`.
The new FireSim driver wraps that runner through the supported `PyRunner` API
and converts using a required, documented target cycle-counter frequency.
Raw-cycle evidence remains separate; new databases use seconds. The FPGA host
frequency and emulation throughput are not substitutes for that frequency.
Old databases remain unchanged. No TVM source or search algorithm is patched.

## Historical local campaign

The four-hour local campaign is complete. The next work is the **remote-only**
FireSim handoff in [FIRESIM_CHUNK_TUNING.md](FIRESIM_CHUNK_TUNING.md): ready
full ResNet default/tuned/profiling guests, three short trained-weight TinyLlama
probes, then exact-export/full-shape checks and complete 32+32 E2E. The bounded
FireSim runner keeps stock TVM search and parallelizes across assigned slots.
The historical policy and results below remain the evidence for local tuning.

This plan follows the user's 2026-10-24 direction: use gSIM if available, otherwise
continue with Verilator, and use the 30-cycle memory model for now. The existing
verified Verilator variant supplies read/write latency 30/30 and outstanding
limits 16/16. Keep this an explicit provisional timing target; it is not a
calibration of the F2 image.

The audit pilots are closed. The user's subsequent direction authorizes deeper
parallel Verilator tuning and requires TVM's own DSE to choose schedules. The
completed follow-up used stock MetaSchedule, described below. Use
[REMOTE_TUNING_README.md](REMOTE_TUNING_README.md) to move sources, inputs and
evidence to a larger machine.

## Current policy: four-hour local DSE

The user superseded the multi-day budgets on 2026-10-25. The previous supervisors
were stopped at **4:50 PM PDT**. The immutable local deadline is **8:50 PM PDT
on October 25**, including setup and validation. Unfinished coverage moves to the
FireSim handoff; do not extend or reset this product's deadline.

The current product is `perf-studies/gemmini-four-hour-dse`; TVM source is the
isolated `tvm-dse-budgeted` worktree, based on the same corrected `c50c4e2`
native library. Its frozen `source-pins.json` and complete patches bind the run.

- Three selected tasks: actual ResNet classifier (M1/K2048/N1008), synthetic
  full-dimension TinyLlama decode K/V (M1/K2048/N256), and synthetic SmolVLA action
  output (M50/K720/N32). The latter two remain operator studies, not EL4 exports.
- Each has 24 stock MetaSchedule proposals in batches of six. TVM retains task
  extraction, EvolutionarySearch, XGB, JSONDatabase and exact-IR trace application.
  The compact profile uses a population of 32, four initial unmeasured samples
  and two genetic iterations. It fixes root epilogue placement, restricts unroll
  steps to 0/16 and excludes CPU placement. Sampled M/N/K groups, applicable
  traversal, OS/WS, hoisting, pipeline, staging and tails remain searchable.
- A matching old trace is remeasured first; otherwise measure the default.
  Historical costs are not imported. CPU and extra baseline comparisons are
  deferred. If bootstrap cannot establish a validated incumbent, defer the
  task instead of launching an unbounded search without a useful cutoff.
- Every new fastest validated result atomically lowers that exact task's cycle
  limit. The host simulator rereads it during execution, including for in-flight
  candidates. CSR markers bracket only the measured second inference; startup,
  first warmup and validation remain outside the moving limit.
- The observer reads architectural cycle state without changing RTL or adding
  target interrupts. It checks every 64 simulator evaluations and refreshes the
  cutoff every 256 evaluations. Small polling overshoot is possible; a completed
  result above the latest incumbent is rejected. Pruned results supply no valid
  latency label. Stock database penalty values are not measured cycles.
- A source-bound Verilator relink keeps the 30/30-cycle, 16/16-outstanding memory
  model. Positive/wrong-oracle controls pass, an unpruned same-ELF run matches the
  parent simulator's cycles, and lowering a cutoff during measured inference
  demonstrably aborts that inference. `controls-v3/summary.json` retains proof.
- The shared host pool still admits 18 external jobs and two builders above a
  4 GiB memory floor. The supervisor stops all its owned work at the deadline.
  No automatic follow-on campaign is authorized after that stop.

The first replacement launch failed before measurement because the baseline
database directory had not been created. It is preserved under `attempts/attempt-1`;
the startup fix has a real JSONDatabase creation/commit/reopen test. The restart
does not extend the deadline or import performance labels from that attempt.

Read the new product's `status.json`, `policy.json`, `incumbents.json` inside each
campaign, per-candidate measurements and stock databases for progress. Early
termination or budget completion does not establish full model coverage.

## Previous stock TVM follow-up — stopped

The new isolated TVM worktree is `tvm-dse-throughput`, based on `c50c4e2` plus
the retained audit patch. Its Python-only execution changes reuse the matching
`tvm-f2-followup/compiler` native library. Frozen patches and source hashes are
inside the new `perf-studies/gemmini-stock-tvm-dse` product.

- TVM extracts tasks and call-count weights, then its gradient scheduler
  allocates trials. EvolutionarySearch uses stock mutation and XGB ranking;
  JSONDatabase stores traces for exact-IR replay by MetaScheduleApplyDatabase.
- The target supplies legal scheduling sketches and measurements. Search retains
  sampled M/N/K group sizes, traversal, OS/WS, hoisting, pipelining, staging,
  applicable epilogue placement and the CPU placement alternative. Degenerate
  axes are pruned by the existing structural rules. Resource and correctness
  checks remain mandatory. No new heuristic selects winning schedules.
- Each campaign has a global budget of 32 times its eligible task count,
  allocated by TVM, with batches of eight. This is a budget, not 32 guaranteed
  successful measurements per task. The prepared graph stays fixed in this
  follow-up; graph-partition DSE and excluded CPU-only tasks remain limitations.
- Baselines run after search so hundreds of slow baseline simulations cannot
  prevent the first learned search batches. Every candidate still needs Spike
  positive/wrong-oracle controls, RTL validation and confirmed memory settings.
  Comparison and deployment claims wait for matching passing baselines.
- Three campaigns cover the fresh actual full ResNet's eligible tasks,
  synthetic full-dimension TinyLlama decode operators and synthetic full-dimension
  SmolVLA expert operators. The latter two do not cover the missing EL4 exports,
  prefill/prefix/session state or complete-model numerical qualification.
- A host supervisor shares 18 of 20 CPUs across builds and measurements, allows
  two compiler workers, and admits new work only above 4 GiB available RAM.
  Requested runner concurrency is eight per campaign, with the shared CPU pool
  enforcing the aggregate limit. It stops its owned campaigns below 1 GiB.
  Host limits control execution only; they never rank or choose candidates.
- Verilator uses the pinned 30/30-cycle, 16/16-outstanding memory model, one warm
  invocation followed by one measured invocation, a 40M total-cycle ceiling and
  a 7,200-second RTL wall ceiling. CPU placements have a 2M total-cycle ceiling.
  Capped or failed measurements supply no valid cost label.

Read the product's `status.json`, `host-history.jsonl`, campaign logs,
`trials.jsonl` and `database/` for progress. Campaigns running or budgets
increasing do not establish convergence, successful coverage or model speedup.

The all-task ResNet run registers 35 workloads, then spends an extended interval
generating the space for a large fused quantization/im2col task. It has not
produced a timing. A separate campaign therefore uses **all 21 actual contraction
tasks**, classified from their mathematical IR, with the same prepared graph,
inputs, stock task scheduler and a 672-proposal global budget. It joins the same
18-CPU/two-builder pool; it does not increase the shared resource ceiling.
The original broad campaign and its evidence remain intact. The other 46
eligible preprocessing/pointwise tasks remain unresolved, alongside the one
excluded CPU-only task. This is an explicit scope boundary, not full ResNet
tuning coverage. No schedule or tile values are selected by the task filter.

The focused product is under `perf-studies/gemmini-resnet-contraction-dse`.
A bounded standalone probe of the large preprocessing space exited with a
segmentation fault while diagnostic traceback collection was enabled; its log
is retained in the broad product. This diagnostic failure does not establish
the cause of the live campaign's slow initialization. Keep further compiler
repairs separate from the active schedule searches.

## What the results establish

The supplied F2 report at TVM `c50c4e2` reports 3,245,476,965 inference-window
cycles for exact ResNet50 and 56,534,457,360 for TinyLlama 32+32. It reports
strict on-device checks for all 99 TinyLlama outputs; seven UART-streamed
outputs were also compared with Spike before readback was stopped. These are
reported results: their raw F2 receipts and complete exported programs are not
present in this workspace. They are not cross-device comparisons.

Startup is outside the session's measured inference interval. The window
includes the complete session, routing and completion; reset, oracle checking
and UART output are outside it. Startup therefore does not explain these
inference numbers. Loader/readback speed remains a separate throughput issue.

The same report says `fased-v1` and `rtl-v1` matched no real ResNet task and no
TinyLlama stage task (0/174 for prefill). These runs used defaults. The existing
16-record FASED library is a bounded synthetic-operator study, not a tuned
complete-model baseline. Its strongest connected result, optional CSE on the
two-projection K/V fixture, is a region result only. Four earlier proposals per
task do not establish convergence or trained-model ranking.

## Audit before allocating more simulation

An exposed flag is not necessarily an effective degree of freedom. The new
`audit_graph_plans.py --program ... --output-dir NEW` enumerates the options on
the exact semantic input and groups prepared modules using structural equality,
including constants. Hash collisions are checked with equality. It reports
which one-option changes actually affect preparation. It does not rank distinct
programs or infer that they have the same compiled performance.

The local audit of all 32 configurations found:

| Input | Distinct prepared graphs | Inactive choices observed |
| --- | ---: | --- |
| Independent two-matmul test | 4 | Bias preload and CSE; fusion 1/2/3 agree |
| TinyLlama compact K/V pair | 8 | Bias preload; fusion 1/2/3 agree |
| ResNet compact convolution pair | 6 | CSE; some bias alternatives; fusion 1/2/3 agree |
| SmolVLA compact action transition | 4 | Bias preload and CSE; fusion 1/2/3 agree |

These are the retained synthetic/region programs from `FASED_SWEEPS.md`, not the
F2 exports. Repeat this audit on each actual stage. `tune_chunk.py` now measures
one representative per equal prepared graph, retaining aliases as explicitly
unmeasured results rather than duplicate observations.

Use this distinction when reducing the search surface:

| Mechanism | Treatment |
| --- | --- |
| Capacity, alignment, dtype, overflow, rounding and synchronization proofs | Mandatory compiler behavior; never search by disabling them |
| Config shadowing, provably redundant loads/zero fills, legal DMA coalescing | Deterministic lowering; keep profitable proven simplifications enabled |
| Constant weight packing and bias canonicalization | Default legal optimizations; retain ablations and search only distinct implementations with a plausible memory/layout tradeoff |
| CSE | Default candidate for pure repeated work; retain alternatives when increased liveness or materialization can matter |
| Fusion levels | Collapse identical partitions; move toward legal per-edge producer/consumer partitions rather than treating four global integers as four useful schedules |
| M/N/K reuse and macro tiling, traversal, OS/WS, hoisting, staging, pipeline overlap, producer/epilogue placement | Actual schedule DSE; compare generated commands, traffic and resource allocations |
| Intrinsic rectangles/tails | Search distinct legal implementations; avoid duplicate full-tile variants and account for padding/staging |
| CPU/device placement, CPU fusion and loop scheduling | Meaningful for small work and host bottlenecks; CPU-only task tuning is still missing |
| Cross-function SRAM residency | Not implemented; needs representation, capacity, alias/lifetime and completion support before becoming a knob |

Do not proclaim every apparently obvious optimization universally best. Keep a
choice in the measured search when it changes code and has a defensible resource
tradeoff. Keep diagnostic off switches outside the normal DSE budget.

## Execution order

1. **Profile actual stages.** Compile diagnostic packages with
   `compile_package.py --profile-operators`; link with
   `build_package_session.py --profile-stages`. This adds buffered per-`call_tir`
   counters as well as per-session-call counters. Warm and measured values are
   retained separately for every invocation and printed after validation as
   `TVM_SESSION_OPERATOR_CYCLES`. A one-call ResNet session now has useful
   internal attribution. Join symbols to `placement.json` for CPU/device
   placement. Wrapper checks, final copies, routing and instrumentation are
   not assigned to operators by subtracting unrelated minima. Keep an
   uninstrumented final A/B because counters and snapshots add overhead.
2. **Bind the real intake.** Recover the F2 `.pt2` stages or prepare a fresh
   independently identified release from the pinned checkpoint, calibration
   and visible inputs. A fresh release is not the historical run's identity.
   Use the numerically corrected TVM source/library pair from the F2 delivery.
   A local `c50c4e2` native library has now been rebuilt and checked, including
   four FP-contract tests. This workspace's older `compiler-v5` remains the
   attribution for the synthetic pilot; keep its costs separate.
3. **Extract and audit every stage before search.** Use the same importer,
   graph choices, target and arithmetic policy as compilation. Record all
   tasks, invocation weights, CPU exclusions and actual database hits. Tune
   prepared functions with their exact epilogues: clipping to `[-127,127]`
   does not cover a ReLU function clipping to `[0,127]`. Report coverage by
   observed runtime as well as task count once profiling exists.
4. **Run useful inner searches.** Start with the highest-cost uncovered task
   families. Remeasure defaults and applicable previous traces under the same
   inputs, engine, memory and fixed-warm policy. Use several measurement batches
   after XGB warmup, preserve failures/caps and compare a second seed or
   ReplayTrace under an equal budget when assessing search quality. Count
   distinct successful generated implementations rather than cached repeats.
   Do not turn a simulator timeout or an unvalidated timing line into a cost.
5. **Improve the space where profiling demands it.** Add single-hart scheduling
   for currently excluded CPU functions. Replace the arbitrary 64-group
   sampling guard with justified resource-derived bounds after checking
   allocator/pipeline limits. Add finer fusion/row-block alternatives only
   with legality and tail cases. Avoid expanding ineffective global toggles.
6. **Measure connected work, then the full session.** Recompile using the new
   exact-matching library and measure each selected graph alternative with its
   CPU glue and intermediates. Test the original strict numerical contract and
   carried state. Chunk warm-cache winners need not win under a full model's
   intervening weights. Final uninstrumented default/winner comparisons belong
   on the same F2 image, exports and numerical release, with identical warm/reset
   policy. Do not sum operator minima into end-to-end claims.

## Local follow-up and missing inputs

The completed local schedule pilot targets the full M50/K720/N32 action-output
fixture. All eight accelerator proposals passed. The selected proposal measures
59,505 warm cycles versus a fresh 64,600 for the previous winner (7.89% fewer).
Its first call also falls from 82,210 to 74,571. These are single measurements on
the same frozen inputs under 30/30/16/16 fixed-warm timing. This is a synthetic
operator result, not new F2 coverage or full-model performance.

Fresh ResNet preparation checked the official weights, every source in the
260-item calibration roster and the committed visible `v00` crop. Its full
export has 68 unique tasks, with 67 containing an eligible region and one
CPU-only mean. A complete diagnostic package checks all 1,000 logits exactly
on Spike, rejects the wrong oracle and records 128 operator calls per phase.
Spike counters count instructions, not target timing. The exact real classifier
task was also searched: its default and four proposals reached their 600-second
RTL wall limits. No classifier timing or winner was admitted. Its CPU control
reached its separate bound. Keep those failures for the remote campaign.

A local gSIM compiler and native model were built with the same neutral latency
pipe. Small positive/wrong-oracle controls passed, and positive warm cycles
matched Verilator. Both larger projection controls exhausted 120-second wall
limits. No gSIM tuning cost or speedup is admitted; further setup is deferred
under the user's audit scope. Continue with the verified Verilator runner.

The 80-image legacy comparison recipe, reduced diagnostics, fresh local ResNet
and unavailable original F2 captures remain separate numerical identities.
