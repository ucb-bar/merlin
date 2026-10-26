# Tuning after the F2 runs

The current remote order is TinyLlama baseline plus ResNet baseline/current
tuned ResNet, then TinyLlama chunk ML DSE, then tuned TinyLlama E2E. The supplied
bitstream already sets the expected DDR3 memory model. Abort candidates at the
current best active-cycle count, excluding startup/loading/warmup. No guessed
F2 specifications or per-job time/cycle caps apply.

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
