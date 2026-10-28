# Simplified TVM path: implementation and measurement review

[ARCHITECTURE.md](ARCHITECTURE.md) is the current top-down design and paper-claim
review. It also discloses the remaining custom placement, memory lowering and
CPU producer-inlining step; removing graph preparation passes did not remove
all custom integration. Operation boundaries are in [TENSORIR_OP_AUDIT.md](TENSORIR_OP_AUDIT.md).

**Historical ELF preparation release:** [source, checks and restore instructions](../../../../out/artifacts/handoff/gemmini-simplified-tvm/v1/handoff_gemmini-simplified-tvm_v1_20261028T021958Z_dd079d3/README.md).
Release TVM pin: `619c7c202ff1fe133eceb8fddf3cc9081a6c7e8a`. All 71 unchanged operator
fixtures pass diagnostic Spike with wrong-oracle rejection; fresh extraction
retains 47 contractions/26 owners over 36 chunks. This supersedes local-only
status in historical checkpoints below. Remote/full-model qualification and
spec-pinned oracle provenance remain outstanding.


The user requested removal of unnecessary custom optimization passes and a
smaller, direct TensorIR/TensorIntrin mapping. This supersedes the earlier
instruction to begin with automatic bias preload and constant prepacking.
The isolated implementation is `/scratch/grammy/tvm-gemmini-simple`, branch
`gemmini/direct-intrinsics`, based on TVM `42b3c1ef2` plus the published dispatcher
fix. Its source and local checks are packaged above; it is not hardware-qualified.

## Campaign re-audit: 2026-10-27

Current TVM source: `64bed0cb9c876b89707e69584909abe5e13b0384`.
[ReplayTrace/default verification](../../../../out/artifacts/verification/gemmini-replay-defaults/v1/verification_gemmini-replay-defaults_v1_20261028T040938Z_9794880/README.md)
contains portable source patches, checksums and compressed logs: 53 schedule/graph
tests plus 21 subtests, 36 wrapper/publication tests, 3 inventory/replay tests plus
51 operator subtests, and 57 FireSim control tests pass. Nine inherited module-size
checks still fail in unrelated files; these are not execution failures.

Use **stock MetaSchedule ReplayTrace, up to 25 candidate trials total per
kernel case or ResNet chunk**, shared across its internal tasks and graph options.
There is no ML warmup, guided phase, XGBoost import/training or learned ranking.
The 25 are sampled schedule proposals, not 25 different mathematical fixtures.
Repeated timings and the three baseline watchdog calibrations are separate and
consume the same occupied-slot allowance. Do not multiply the trial budget by
head count or internal task count. Duplicate/failed proposals count against the
cap; report actual unique successful candidates. Stop at the cap, wall budget,
or an explicit search stop; stochastic failure is not proof of space exhaustion.

Start from a deterministic shape-derived baseline: largest exact legal tile
(up to 16), bounded M/N operand-reuse groups and K batching. Prefer less logical
operand traffic, then larger legal groups; validate the actual aligned physical
allocation. Try at most eight ranked choices and a one-group fallback. Preserve
explicit user schedule settings. This is a heuristic, not a cycle predictor or a
claimed optimum; pipeline/hoisting/dataflow alternatives remain search choices.
Keep the baseline in the database and choose only correct measured winners.

ResNet retains **4h total DSE and 30min per E2E run**; the time cap wins even
if some chunks have fewer than 25 trials. For the 16-case **TVM** kernel sweep,
use a conservative **24 occupied-slot-hour aggregate ceiling**, excluding CPU
preparation. This is an authorized planning estimate, not measured runtime:
400 proposals at an illustrative 3 minutes occupy 20h, plus 48 baseline jobs
at 3 minutes occupy 2.4h, leaving 1.6h for confirmations. Actual calibration and
profile counts may differ; the same cap still wins. This is not 24h per case,
not 24h for every compiler, and concurrency must not multiply the allowance.
Close admissions at budget; finish active jobs under their frozen median
watchdogs and report overrun. Preserve actual cycle labels and failures.

### Stack findings and limits

- Ordinary mathematical TensorIR and registered TensorIntrins remain the device
  interface. Automatic bias folding/prepacking/scale specialization, readout
  normalization and DMA-merging peepholes remain removed. Required SRAM allocation,
  capacity, host-DMA hazards and completion remain. No new graph-rewrite pass.
- ReplayTrace resamples decisions from existing legal traced schedule spaces.
  The shared tuner uses round-robin task allocation and no feature extraction or
  cost-model update callback in replay mode. TVM's API receives an unused stock
  RandomModel placeholder; ReplayTrace does not consult it. The evolutionary/XGB
  route remains an explicit legacy option outside this FireSim campaign.
- Raw completed target cycles remain evidence; the runner converts to seconds
  using the documented counter frequency for ordinary JSONDatabase records.
  Wall duration controls watchdogs/occupancy only. Best selection uses successful
  actual measurements; complete slow results remain valid and timeouts are errors.
- Prior 71-case Spike evidence does not qualify changed default schedules or the
  new 16 kernel inputs. Exact per-program software-spec and selected-hardware
  admission remain separate from matcher recognition. Capability defaults bind
  an older source profile; do not infer deployment equivalence.
- Current driver measures extracted contractions, not grouped complete calls.
  Kernel T01/T02 require all 32 heads and T04/T06 all outputs in one timed call.
  New complete-case fixtures/builders and remote correctness/cancellation
  qualification remain outstanding; no FireSim cycle result is claimed.
- The local env.sh points at the older tvm-device-op-audit Python tree. Explicitly
  select the current TVM Python path and native-library identity alongside app
  scripts before remote building. --tvm-source alone does not select Python imports.

### Paper methods text

"TVM used stock MetaSchedule ReplayTrace to sample up to 25 candidate schedules
per kernel case or ResNet chunk, shared across internal tasks and graph settings.
We used a deterministic shape/capacity-based baseline favoring large legal tiles
and operand reuse, and selected the fastest numerically correct measured result.
No learned cost model, ML warmup or guided phase was used. We recorded unique
successful trials, duplicates, failures, actual slot occupancy and stop reasons.
The optimization label was completed target cycles, converted to seconds at the
TVM Runner boundary. Baseline watchdog calibration and timing repetitions were
separate from candidate count. Full TinyLlama/SmolVLA inference was backlogged;
its isolated kernels do not represent full-model results."

Fill actual counts, hardware/cache scope and stop reasons before publication.
The earlier XGB experiment and tests remain immutable evidence at
[the prior audit revision](https://github.com/ucb-bar/merlin/blob/f0b2d0681/examples/gemmini/comparisons/tvm/OPTIMIZATION_KNOBS.md);
they no longer describe the active tuning method.

## Current implementation work

- Automatic bias-to-accumulator rewriting has been removed from the active
  preparation pipeline. Explicit `bias_preload=True` plans refuse rather than
  silently changing meaning. A bias addition remains in the mathematical IR.
- Explicit accumulator initialization already present in TensorIR retains its
  ordinary initialization TensorIntrin and DMA implementation.
- Command-level zero-init elimination and DMA-merging peepholes have also been
  removed; explicit tensorized commands and required hazards remain.
- Frozen-scale specialization, constant prepacking and custom readout normalization
  are also removed. Exact compatible readout uses traced scheduling and intrinsic
  proof at tensorization; source scale arrays and weight layouts remain intact.
- The user approved retaining only spec-declared, proved composites. Full-range
  INT32-add intrinsics have now been removed; their original fixtures use host
  arithmetic. Affine ranks are capped at five and scaled per-operand sums at two.
  Unbounded standalone i32 readout no longer masquerades as a byte-affine map.
  Bounded affine metadata reports accumulator and PE partial-sum bounds.
  Remaining matchers still require their selected proof/emission evidence.
- Keep mathematical legalization, standard graph passes, schedule transforms,
  registered intrinsics, necessary SRAM allocation/hazard handling and stock
  MetaSchedule. Removing a pass does not authorize changing quantization or tolerances.
- Fresh extraction preserves 47 contraction workloads across 26 owners over
  all 36 ready chunks, but none of their keys matches the old catalog. Rebuild
  packages and remeasure; do not reuse old costs or assume matching traces.
  Frozen input/reference payloads and receipts remain unchanged.

## Historical scheduling and cost-model audit (superseded by ReplayTrace above)

Audited the active TVM `gemmini_search.py`, `gemmini_meta_schedule.py`,
`gemmini_sketch.py`, `gemmini_tensorir.py`, `apps/gemmini/tune_models.py`, and
Merlin's `tune_firesim_chunks.py`, `firesim_runner_seconds.py`,
`firesim_job_pool.py` and `run_firesim_study.py`. No new cost model is implemented.

```text
source mathematics -> stock graph passes -> extracted TensorIR tasks
  -> Gemmini schedule rules + stock mutations -> TensorIntrin + SRAM lowering
  -> ELF builder -> FireSim correctness + measured cycles
  -> stock XGB update + JSONDatabase -> stock trace replay -> connected/E2E check
```

| Layer | Current implementation | Simplification disposition |
| --- | --- | --- |
| Frontend/graph | Fork semantic fixes; stock legalization/fusion/CSE; four custom preparation transforms now removed in the isolated branch | Preserve numerical correctness fixes. Keep stock graph knobs; do not expand supported math by rewriting it to fit a device pattern. |
| Extraction/admission | Stock `extract_tasks`, call-count weights; `_supported` recognizes contraction, reduction, pool, elementwise. Unsupported CPU-only tasks are recorded but excluded from search. | One family dispatcher for baseline/search/replay. Recognition is not proof of strict software-spec admission; the audit inventory is not a production gate. |
| Schedule space | Custom rule builds ordinary traced schedules: rectangles/tails, M/N/K reuse, OS/WS, traversal, hoisting, pipeline, staging, epilogue placement, unroll | Keep target legality and ordinary schedule APIs. Prune duplicate/inactive choices before adding knobs. Do not remove necessary movement scheduling to reduce line count. |
| Mutation/search | Stock EvolutionarySearch and mutators. Compact profile: population 32, 2 genetic iterations, 4 initial unmeasured; full: 64, 3, 16. Mutation probability .85; current FireSim epsilon-greedy=0 (historically .1); tile/location/unroll mutation weights .9/.05/.05 | These are custom settings, not a custom search algorithm. Put the selected settings in one explicit campaign profile; avoid unreviewed tuning of the tuner. |
| Task allocation | Stock gradient scheduler inside each invocation; wrapper starts separate owner campaigns and shares slots/deadline externally | This is not one global model-wide task scheduler. Record dedup/shared measurements separately from shared cost-model training. Consolidation into one invocation is a possible later change, not yet selected. |
| Features | `MathematicalFeatures` replays each trace with postprocessing removed, then uses stock `PerStoreFeature` | Retain this small adapter: late tensorization hides loops in extern calls. It is not a Gemmini cycle simulator and has no explicit model of DMA latency, bank conflicts or command setup. Eager tensorization, notably pool, can still hide work. |
| Cost model | Stock `XGBModel`; per-workload normalized score based on best observed cost / candidate cost, median of repeated `run_secs`; one thread; FireSim warmup 50 instead of upstream 100; adaptive training disabled | Fifty total feature-bearing samples enable learned prediction, not demonstrated ranking quality. Warmup is per model object, not per workload, and error samples can count. The current profile explicitly selects 50; do not claim convergence from a saved booster. |
| Build/run | Target adapters create RV64 guests and collect actual measurements. FireSim wraps cycle-valued runner results into seconds using the documented target counter frequency | Keep raw cycles as evidence; use standard seconds in fresh TVM records. Older local databases storing cycles are a separate convention and must not be mixed. |
| Feedback/errors | Stock callback forwards runner errors from successfully built candidates; XGB maps absent timings to 1e10, then a poor normalized score. The median-watchdog pool retains completed results even when slower than the incumbent | This is not censoring-aware training. Keep complete slow timings under the approved wall-watchdog policy. Track timeout, compile failure, wrong output, infrastructure error and success separately; no timeout threshold becomes a measured cycle count. |
| Database/replay | Stock JSONDatabase and MetaScheduleApplyDatabase; custom registration before opening workload-dependent intrinsic traces; measured default fallback retained | Keep exact structural matching and source/hardware/timing bindings. Database reuse is not proof of learned-model transfer or connected speedup. Regenerate changed graph plans. |
| Physical memory/runtime | Required SRAM lifetime/capacity/address lowering, host-DMA hazards and completion; C command ABI, config shadow, legal WS tile retention | Keep correctness lowering. Removed command coalescing/zero-init peepholes. Runtime reuse remains a separate audited optimization, not a claimed XGB knob. |
| Export/session | Custom static graph wrapper, constants/workspace, bare-metal link and session ABI | Required integration with the current guest; not an optimizer. Replacing it with upstream AOT is separate scope. |
| Legacy | `gemmini_tuning.py` bounded search, `gemmini_schedule.py`, legacy graph path remain referenced by old verifiers/tests | Outside current MetaSchedule path. Isolate from current entrypoints/docs first; delete only with consumer migration, preserving historical receipts. |

The FireSim driver now omits sequential Spike admission and incumbent-result
rejection. Its candidate adapter retains all completed correct timings, calibrates
three real baseline jobs, and skips cached results when updating timeout history.
Orchestration defaults disable removed graph preparation and reject old bias/
prepacking requests before launch. Remote wall-cancellation qualification and new
model/task artifacts remain required.

### Evidence and limits

The approved median-watchdog implementation passes **51 focused tests** covering
three real baseline jobs, nine-sample history, floor/ceiling, workload/source
isolation, cached-result exclusion, completed slow timings, timeout failure
labels, physical-slot serialization/quarantine, finish-active behavior and cycle
units. These are local adapter/control tests, not remote FireSim qualification.
Log: `out/build/tvm-simplification/median-bound-tests.log`.
The required structure check passes every category except the same nine inherited
module-size failures; none is a new runner failure. Log:
`out/build/tvm-simplification/structure-median.log`.

The simplified path passes **66 focused tests plus 18 subtests**: explicit
accumulator initialization, unchanged bias/scale numerics, source-layout
convolutions, exact readout, DMA hazards, graph trace and database replay.
These use LLVM plus the command emulator, not Spike/FireSim. Log:
`out/build/tvm-simplification/direct-final-tests.log`.

A bounded feature diagnostic checked five existing fixtures with CPU alternatives
excluded. Subtract, INT64 byte DMA and max-pool each yielded one legal structural
candidate; window average and standalone readout each yielded two. All arrays
were finite (164 feature columns); the two-candidate pairs differed. This tests
feature plumbing, not predictive accuracy or complete mutated design-space size.
Pool's nonempty features do not prove the hidden native window is represented.
Log: `out/build/tvm-simplification/feature-audit.log`.

Proposed acceptance evidence for the cost model: counts of unique successful
measurements, errors by reason, feature-bearing samples per workload, batches
proposed after training, and ranking quality on later measured candidates versus
ReplayTrace under a matched budget. A family with one legal schedule and no
remaining sampled decisions needs measurement, not an XGB search campaign.
Do not generalize the five diagnostic fixtures into that claim for all shapes.

### Follow-up functional checkpoint

The new six-case RV64 guest passes diagnostic Spike; the wrong-oracle guest is
correctly rejected with exit 22. Cases cover small/full/tail contraction, vector
and matrix bias epilogues, and explicit INT8 DMA copy. All six retain device work;
both bias cases report no automatic preload. Both ELFs pass the no-loop-FSM audit.
This is the locally rebuilt RNE diagnostic, not the unavailable pinned oracle.

The [verification checkpoint](../../../../out/artifacts/verification/gemmini-tvm-simplified/v1/verification_gemmini-tvm-simplified_v1_20261028T014231Z_16b2a77/README.md)
retains source patches, ELF hashes, original failures and corrected results.
It is local/unpublished and is not a remote delivery location. A first build
exposed a missing explicit copy-fixture opt-in; corrected source passes.
Package/provenance tests passed 32 cases; five setup errors used the removed
prepacking setting. After correcting that fixture, the database/fallback suite
passes 22 tests plus 10 subtests. The study-admission suite passes 11 tests.
Do not add overlapping suite counts into a claimed unique-test total.

Current TVM README/OWNERSHIP/TUNING/FIRESIM guides now describe the simplified path;
the predecessor's long inventories are linked by immutable source revision.
No complete-model rerun, FireSim measurement or publication is claimed by this
checkpoint. The later simplified task index regenerates the roster separately;
see FIRESIM_HANDOFF.md for exact counts and identity changes.

## Research context (not the current search policy)

[Ansor, OSDI 2020, sections 7.1–7.3](https://www.usenix.org/system/files/osdi20-zheng.pdf)
uses up to 1,000 measured candidate programs per operator/subgraph test and
1,000 times the number of subgraphs as the network measurement budget. Its
cost model selects batches for actual hardware measurement and retrains from
those results. These are CPU/GPU experiments, not FireSim provisioning guidance.

[MetaSchedule, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/e894eafae43e68b4c8dfdacf742bcbf3-Paper-Conference.pdf)
separates composable schedule-space construction from the learning-driven search.
Its hardware-specific tensor-core rule uses ordinary scheduling/tensorization.
The [supplement, A.5](https://proceedings.neurips.cc/paper_files/paper/2022/file/e894eafae43e68b4c8dfdacf742bcbf3-Supplemental-Conference.pdf)
reports 220.66 minutes for its ResNet-50 tuning experiment. This is an experiment
result, not a universal required duration or a prediction for our hardware.
Neither cited description establishes a requirement to kill a running candidate
at the incumbent's measured cycle count.

The pinned TVM v0.19 `LocalRunner` has an explicit wall timeout (default 30 seconds),
while `EvaluatorConfig` separates repetitions from search trials. Those host/RPC
defaults must not be copied blindly into FireSim. A search trial is a different
candidate schedule; repeated execution of one schedule checks timing stability.
XGB's upstream default starts learned predictions after 100 samples; the existing
Gemmini helper overrides that to 4. Four samples do not establish a reliable
model or convergence; any choice must be reported explicitly.

## Approved FireSim measurement policy — remote values/qualification pending

- Keep the selected warm measured-cycle window, exact output checks and target
  identity. Store raw cycles and convert by the bound target counter frequency
  for TVM `run_secs`; host wall duration is not the optimization label.
- The user approved median-based wall timeouts replacing incumbent cancellation.
  Collect three correct baseline wall durations using an explicit bootstrap
  watchdog. Let `m0` be their median and `m` the median of the latest nine correct,
  uncached candidate durations (use `m0` until nine exist). At slot admission set
  `timeout = min(hard_ceiling, max(3*m0, 3*m))` and freeze it for that job.
  Refuse a ceiling below `3*m0`. Source-bound histories persist on disk.
- Duration covers slot occupancy from launch through collection/cleanup, including
  startup and warmup; it excludes queueing and CPU compilation. Partition history
  by workload, compiler/graph, hardware and timing setup. Reused cached results
  are not new duration samples. Keep the baseline floor to prevent progressively
  shorter successful candidates from ratcheting the timeout down.
- A median is not an upper bound. This still censors unusually slow candidates;
  report all timeouts and their threshold, never use the threshold as measured
  cycles. A high timeout fraction means the policy needs review, not convergence.
  Keep every completed correct latency, including candidates slower than best.
- Run a small correctness/timing pilot before committing a candidate count. Report
  startup, execution, result collection and occupied-slot duration separately.
  Measure a few repeated baselines to establish whether cycle measurements are
  deterministic; keep repetitions distinct from unique candidate trials.
- Estimate feasible trials from the remaining occupied-slot budget and observed
  per-trial occupancy, reserving connected-confirmation time. Median occupancy
  is only a planning estimate; the actual occupancy ledger enforces the budget.
  Search batches and per-task allocation remain explicit proposals for review.
- The old roster's 47 tasks would require 47,000 measurements at 1,000 each.
  Three baseline jobs per task alone would require 141 jobs, charged to tuning.
  At 60 seconds of slot occupancy per trial, 4 hours permits at most 240 trials
  and 1 hour at most 60, before confirmation overhead. This is an illustration,
  not a measurement of the remote runner or a chosen trial count.
- Keep the 4h ResNet / 1h TinyLlama / 1h SmolVLA aggregate tuning budgets and
  finish-active policy. Final E2E alone has 30min ResNet / 7h TinyLlama / 7h SmolVLA
  wall timeouts. Candidate timeout values are still unset; no paid run is launched.
- Stock TVM records missing timings with failure sentinels, not measured durations.
  Keep timeout/error counts and raw receipts; never describe those placeholders
  as completed cycles. Completed slow candidates are retained; timeout/failure
  observations never update the successful-duration median.

## Approved composite scope: declared and proved only

A TensorIntrin pairs a mathematical tile with an implementation; TVM does not
require one machine instruction per intrinsic. The simplification decision is
whether to retain small compositions of declared Gemmini primitives, not whether
to replace normal tensorization with a graph rewrite.

| Family | What retaining it requires | Boundary |
| --- | --- | --- |
| Bounded affine `a*x+b*y+c` | Existing semantic matcher and tile desc/impl; byte coefficient loads, primitive WS compute for non-unit terms, accumulator seed/add and exact output. Keep every prefix-bound proof and expose scratch/command work. | At most two same-shape indexed i8 sources, signed-byte coefficients, bounded i32 offset/partials. No arbitrary f32/wrapping-i32 algebra. Bind `elementwise_map_standalone` obligations. |
| Byte residual sums | Widen/accumulate plus readout; scaled-input variant additionally performs separate input rounding. Retain exact declaration-specific scale/rank proofs and exhaustive pair checks. | Single final rounding and per-operand rounding are different contracts. Unit-coefficient sums need no synthetic mesh multiply. |
| Sum via ones | Existing reduction matcher, synthetic ones tile, loads, primitive contraction, full-width result and optional exact output conversion/readout; count all setup/traffic. | This is not a native sum opcode. Bind contraction ranks/PE and accumulator bounds plus the selected reduction declaration; generic f32/i64 sums are not admitted. |
| Window mean | Above sum plus the reciprocal/rounding already expressed by the source program; explicit gather/layout work. | No rewritten float average or new quantization recipe. |
| Max window readout | Keep native pooled-readout intrinsic, complete-window proof and explicit gather. | `pooling_on_readout` is not blanket standalone maxpool support; producer/composition admission remains required. |

The user approved retaining declared, proved compositions as a separately reported
mapping class, with custom graph rewrites removed. This approval is not proof that
every existing matcher discharges the corresponding software-spec obligations. Stock schedules
choose tiles, placement and buffering around each intrinsic. The command sequence
inside an intrinsic is fixed, not a separately searchable MetaSchedule subgraph.
If that sequence itself must be searched, expose its stages as ordinary TIR
blocks/schedules; that is additional implementation scope, not part of this cleanup.

The earlier optimization inventory and measurements remain in
[Git history](https://github.com/ucb-bar/merlin/blob/82b321a89a8a646877484f4bd09637040ce0fa26/examples/gemmini/comparisons/tvm/OPTIMIZATION_KNOBS.md).
