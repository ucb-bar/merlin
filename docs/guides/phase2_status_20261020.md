# Gemmini Phase 2: implementation, evidence and performance gap

Snapshot: **20 October 2026, 21:30 UTC**. New optimization work is paused for this review; an already-launched diagnostic queue may finish. **No Phase 2 full-model speedup has been established.**

## 1. What Phase 2 changes

Phase 1 supplies a working generated compiler and its numerical contract. Phase 2 copies that compiler, then edits compiler policy and implementations: tile sizes, loop order, reuse, packing, pass selection/order and new transformations. The original package stays immutable. Hardware, numerical tolerances, references and the timing evaluator remain fixed during a compiler comparison.

Merlin owns generic planning and checks; `gemmini-mlir` owns Gemmini instruction emission and target adapters. A shared pass becomes useful when its result changes the **actual emitted program**. Passing a library unit test does not establish Gemmini execution or latency.

The selected V26 baseline has 16 files, package SHA256 `29213759089b71de26f8a5ab26ca6ef1b8071dc0bb690041eb7de5d0efb53f24`. We are not substituting the handwritten backend. FSM/hardware-loop instructions remain forbidden.

## 2. Passes: their inputs and current state

**RTL facts describe resources; the software specification describes meaning.** Facts constrain dimensions, widths, banks, capacities and interfaces. The software/ISA contracts describe signed arithmetic, overflow, layouts, configuration persistence, address effects and completion. Neither document alone proves an executable adapter correct.

| Pass/mechanism | How it uses software and RTL information | Current evidence / limitation |
|---|---|---|
| Tiling and operand reuse: `choose_dense_block`, `schedule_contraction` | Shapes/layouts define legal subproblems; array dimensions and scratchpad/accumulator capacities bound blocks. Reconstructs operand ownership. | Shared implementation exists. Small RHS-residency candidate has exact native outputs and an estimated improvement; broad target routing is incomplete. |
| **Bounded weight panels + multiple live output tiles** | Chooses an M-group and N stripe that fit operand and accumulator storage. Preserves increasing K and final-only readout; emits explicit raw PRELOAD/COMPUTE commands. | New separate V26 candidate: **25 static tests pass**, including emitted-operand decoding, independent complete-product checks, tails and negative controls. No native ELF/timing result yet. |
| Temporary allocation and copy elimination | Uses aliasing, liveness, ownership and layout compatibility; allocates within declared storage windows. | Checked shared mechanisms exist; general loop-carried/cross-operation device ownership is incomplete. |
| Fusion/readout composition | Uses exact arithmetic, scale/rounding, activation and layout contracts; selects only available target operations. | Shared integer/neutral-model paths exist. Broad generated-Gemmini epilogue and attention integration remains incomplete. FP32 islands cannot simply be removed while preserving the selected contract. |
| Packing/layout hoisting | Proves a layout conversion depends only on an unchanged operand; moves it outside repeated tile loops, with bounded private workspace. | New transpose-pack helper is an **untested draft**. V26 currently repacks transposed RHS tiles inside M/N/K loops. |
| Larger transfers | Uses load stride, block stride, element width and local-row mapping. Selected hardware permits up to 64 i8 columns per load. | Existing independent arithmetic checker accepts only 16 columns. Wider-load emission and an explicit checked mapping are still needed; hardware support is not the same as compiler/checker support. |
| Persistent-configuration and fence cleanup | Tracks exact configuration-state effects. Removes identical setters or barriers only where no outstanding work/unknown effect invalidates the proof. | Raw-i32 store-configuration hoist has static tests. General emitted-program cleanup is an **untested draft**. Payload/completion fences are retained without a stronger dependency proof. |
| Device spills (`allocate_with_spills`), complete-row reductions, final program checks (`finalize_program`) | Respectively require exact save/reload semantics; full reduction domains and numerical rules; decoded instruction/address/branch limits. | Shared mechanisms exist. General device spills, device attention/KV composition, asynchronous overlap and double buffering are not broadly enabled. Final checks enforce limits; they do not optimize latency. |

Important correction: V26 **already keeps one output tile in the accumulator through full K**. The new scheduling work extends this to several output tiles so weights can be reused across them.

The handwritten branch provides a concrete mechanism reference: stationary B across M, full-K operand caching, bounded accumulator stripes, grouped 64-column loads and configuration outside tile loops. Its inspected GEMM path uses ordinary software loops. We have not copied its kernels or tuned model tables. See [shared-pass interfaces](../reference/shared_compiler_passes.md) and the [integration guide](shared_pass_phase2.md).

## 3. Which hardware, specification and simulators?

All paths below use `T=/work/merlin/tasks/phase2-el4-explore-20261018` and `P1=/work/merlin/tasks/phase1-harness-v2`.

| Selected input | Path and SHA256 |
|---|---|
| FireSimGemminiRocketConfig, 30 MHz facts | `P1/merlin/out/artifacts/probes/gemmini/fire-aligned-30mhz/input/facts-v6.json` — `b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af` |
| Elaborated FIRRTL backing those facts | Same probe's `firesim-target/elaboration/firechip.chip.FireSim.FireSimGemminiRocketConfig.fir` — `b7d9826b1c9f16fd00bc757b4334a7f330a842c6a494e0c28608dc351b305bf0` |
| Reviewed software/numerical specification | `T/gemmini-mlir/docs/v26/inputs/software-spec.yaml` — `3c1b8aa1c77775e1c0e590af871f05e038743d7000a99860fb12843f7053372f` |
| Execution/toolchain contract | `T/gemmini-mlir/docs/v26/inputs/target_contract.yaml` — `bfbfac9e98ff028aa184b5b13ea44e7421afeac2c35e1d3bb572fd0cbad5aa55` |
| Archived gSIM label producer | `P1/.tmp/gsim-fire-aligned-provisional-v3/native/emulator` — `76ddbfdc5b3beadef6b6b37768a08522122327057b92d71d3539a9529e4ea651` |
| Functional Spike | `P1/out/fire-aligned-spike-30mhz/runtime/bin/spike` — `5df9c82ca3e921823da8606284a258b491541defbafee4586fb938639e03c6b9` |

The resource view is a **16×16 array**, **256 KiB scratchpad** (four banks), **64 KiB accumulator**, i8 operands and i32 accumulator storage. Numerical semantics also specify internal MAC overflow and readout rounding; i32 storage does not grant arbitrary reassociation. ISA fields come from the selected `GemminiISA.scala` recorded in the facts; instruction effects are checked against selected controller/receiver sources. The 16-byte client interface is a peak transfer granularity/rate assumption, not measured sustained DRAM bandwidth.

## 4. How Spike was calibrated, and its actual error

Functional Spike executes instructions correctly; it is **not cycle-accurate Rocket timing**, even though Rocket is in order. Our annotated model adds register dependencies, CPU issue/resources, address-based cache/memory events, decoded accelerator commands, queues, local-storage dependencies and fence/completion effects.

Calibration reuses Phase 1's public gSIM **kernel-call cycle labels** and original executable/input/window identities. We observe the corresponding public programs in annotated Spike, constrain resource structure with RTL/ISA evidence, fit service prices on TRAIN sources, then check frozen held-out sources. Variants/repeats of one source stay together. Private model answers are not training labels. Simulator walltime is never a target-cycle label. Compiler candidates cannot edit the evaluator to improve their score.

**Calibration boundary:** use multiple independent public Phase 1 source groups with compatible hardware, memory system and timing windows. Do not fit to the newest checkpoint alone, Phase 2 candidate results, private validation models, or the quoted hand-tuned full-model timings. Freeze the model before compiler comparisons. Additional timing mechanisms are separate, unqualified source versions until independently validated on the declared public Phase 1 protocol; publishing their source does not install them as the evaluator. No coefficient fitting was performed during the subsequent integration/publication task.

The original training archive is `capsule_timing_training_d4c730b33e12.tar`, SHA256 `f815a0d2f4d80415b1e89e6ccdd62a0eb7af039287f539c91bd235451f8cfd5a`: **483 eligible public records**, not 483 independent capsules; zero eligible hidden records. Producer `training_ready=true`, overall `timing_ready=false`. Some initial observations were missing; later recovery/extension reports remain separately identified.

| Evaluation | Median absolute relative error | p95 error | Ordering / conclusion |
|---|---:|---:|---|
| Selected `dc413…` model, primary public archived domain | **10.38%** | **26.95%** | 100% on eligible pairs; 61 held cases, 41 pairs, three families. Passes that domain's gates. |
| Same model, broader normalized 632-record replay | 11.79% | **30.33%** | 100%; fails the 30% p95 limit. |
| Separate TRAIN-fitted successor `e2a…`, same broader replay | 13.22% | **29.21%** | 100%; 88 held cases/81 pairs. Statistical gates pass, but activity-support gate still refuses: held activated-store count 33 exceeds TRAIN maximum 32. **Not installed as the qualified successor.** |

Relative error is `abs(predicted−measured)/measured`; p95 is its 95th percentile. Ordering measures which variant is faster for measured separations above 5%. The primary families are convolution, matmul and resident chains; these statistics do not validate all attention/capacity cases.

The successor changed modeled DRAM latency 12→9 cycles and PRELOAD latency 12→20. These are fitted prices, not newly measured hardware constants. The 632-record replay and 788-record evidence ledger are different sets; neither count denotes independent capsules.

**Current candidate example:** fixed-model estimate 1,175→988 cycles (−15.9%), with all 544 outputs exact. Loads fall 12→6, requested reads 3,744→1,056 bytes, commands 32→26; fences increase 16→18. This is exploratory: its flush count is outside the original TRAIN range. There is no measured gSIM pair validating these particular estimates. The earlier 1,490→1,419 gSIM result is a separate V23 single-replicate hypothesis.

Accuracy gaps are concrete: approximate Rocket pipeline/bypass/load replay and branch prediction; LRU cache replacement where selected RTL uses random replacement; conservative accelerator overlap/completion; weak coverage of large capacity, attention and KV state. Existing data can test structural improvements. Targeted compatible Phase 1 measurements are needed where labels lack those mechanisms; a full Phase 1 rerun is not automatically necessary.

### Simple queue model for schedule selection

The PI's proposed queue model is useful. On this in-order Rocket configuration, the relevant reorder structure is **Gemmini's reservation station**, with separate load/execute/store queues. We should model that structure rather than invent a CPU out-of-order ROB.

An initial finite-capacity event model already exists: `spike-timing/timing/core/include/merlin_timing/reservations.hpp` supplies reservation windows; the selected native provider's `StagedObserver` connects them to actual commands. Selected reservation capacities are **8 load, 16 execute, 4 store entries**. Controller and DMA windows are modeled separately. These capacities come from selected RTL evidence, not cycle fitting.

For each command, model: **CPU arrival → free reservation slot → dependencies satisfied → service resource available → service → effects visible / slot released**. Admission stalls the CPU when the selected reservation pool is full. A controller queue slot, a DMA slot and a reservation entry have different release events; do not release all of them at issue. Fences drain relevant work while preserving the selected pending-PRELOAD behavior. Fit service prices to public gSIM labels, preserving fixed capacities and dependency/completion semantics.

The missing connection is actionable per-capsule feedback: modeled occupancy/high-water marks, admission-blocked cycles, dependency waits, resource waits, compute-idle periods and fence drains. These waits can overlap; do not sum them into a cycle breakdown without exclusive attribution. Use them to choose an experiment: queue-full with idle compute suggests load/dispatch bottlenecks; empty queues with idle compute suggest insufficient feeding; sustained compute activity suggests less headroom from command cleanup.

This is a deterministic, dependency-aware queue model for bursty command streams. A simple average arrival/service-rate formula alone cannot explain ordering, finite capacity or fences. Validate the resulting estimates against the existing held-out labels before extending their qualified domain. Queue instrumentation and model changes are future work; this documentation update does not alter the running evaluator or claim new accuracy.

## 5. What we test and iterate on

A capsule is a small public accelerator-containing program with inputs, independent complete-output references, a numerical contract and a declared timing boundary. Public PyTorch sources/synthetic inputs are captured into typed compiler input, compiled with the selected package, then executed and checked. Public model configurations inform the shape distribution; complete validation models/answers are not optimization or timing-training objectives.

The intended model checks remain full ResNet50 B1/224×224, TinyLlama B1/all22 layers/32 prefill+32 decode with GQA/KV reuse, and SmolVLA B1/one512×512 camera/chunk50/ten denoising steps with prefix reuse. There is no comparable complete-model Phase 2 timing result for SmolVLA.

The planning inventory has **243 members and 90 coverage requirements**, not a sealed admitted cohort. Original roles are 205 performance declarations/38 guards; a separate proposed reconciliation is 186/57 and is not a silent replacement of the original objective. Refusals stay visible.

| Test layer | What it actually covers/status |
|---|---|
| Current ordinary diagnostic queue | `PG_source_traversal_guard_0000`: M17×K16×N32; `PG_source_capacity_0079` / original190:175: M64×K256×N64. Compares retained compiler revisions, complete outputs, fresh observations and feedback. This is not the whole performance cohort. |
| Retained public model-scale packet | 17 existing sources: hidden width 2048, FFN 5632, prefill/decode; GQA 32 query/4 KV heads, head width 64; convolution stem/spatial/pointwise/tails; joint scratchpad/accumulator fit/spill controls. It is a coverage/provenance packet, not 17 new target admissions. |
| Fresh larger capture | M32×K2048×N256: 16.8M MACs and 8,192 exact reference outputs. Publication eligibility refused **before compilation**; all public-side signature checks passed. The generic refusal does not establish its underlying cause. |
| Remaining large/connected cases | Full 2048×2048 and 2048×5632 weight arrays exceed the current one-million-element native array limit. Need checked streamed/partitioned storage retaining full K and complete timing scope. Attention/KV and reduced-depth complete-graph numerical/device contracts remain incomplete. |

Search edits **compiler rules**, not answers: e.g. choose a reusable panel from shape/layout/capacity, then apply that rule to future compilations. Correctness guards remain separate from performance objectives. Actual queued comparisons must precede promotion.

## 6. Where the performance gap is, and what could close it

Reported Phase 1 FireSim invocation results: ResNet50 **2,179,239,099 cycles**, 1,000 outputs exact; TinyLlama all 22 layers/32 prefill+32 decode **80,999,302,990 cycles**, 24.85M elements exact. They include in-model host work, commands and DMA. No newer complete-model Phase 2 speedup is established.

The supplied hand-tuned figures (~24M/~220M) imply nominal 91×/368× gaps. Matching token counts, numerical contracts, cache state, completion boundaries and FSM policy are not yet verified. As a sanity check, all 64 token-forwards through TinyLlama's dense layers alone require about **242M cycles at 256 MACs/cycle**, under dense-work assumptions, before attention/memory/host cost. Thus the 220M reference needs scope reconciliation before becoming our exact target.

**Observed inefficiencies:** the reported first ResNet convolution has ~2.46M activation-row loads and ~160K fences; all contractions are offloaded, yet residual adds/ReLUs/pools and conversions retain host work. TinyLlama's selected multi-plane attention expands to 3,675 contraction calls per invocation, with additional host scaling/layout work. Current V26 reloads operands per output tile and repeats configurations/packing. We have not measured each source's percentage of total latency.

**Why the passes are promising:** panel reuse removes repeated loads, larger transfers amortize command overhead, multi-tile accumulator lifetimes avoid intermediate movement, packing hoists eliminate repeated host loops, and checked fusion/cleanup reduce transitions and serialization. The handwritten backend demonstrates these mechanisms exist without FSM instructions; our small exact-output experiment demonstrates one real traffic reduction. Neither proves a 90× end-to-end gain.

**Remaining closure:** connect passes to emitted programs and qualify them; admit model-scale/connected capsules without dropping failures; expand timing domains; resolve physical transfers, usable ports, packing/host costs and legal overlap in roofline feedback; then perform paired full-model checks. A compute roofline is `ceil(useful MACs/256)`, not a complete latency prediction. CPU-inclusive Spike estimates must not have CPU/DMA/compute costs added again. Complete attainment and the 0.8 target remain unestablished.

There is also an implementation delay we own: the latest completed four-job batch stopped before compilation because the sealed launcher omitted the Chipyard path. The metadata preflight passes after correction; its successor was already launched when this note was requested. Failed setup attempts are not optimization evidence.

### Evidence entry points

- [Shared mechanisms and their limitations](../reference/shared_compiler_passes.md), [integration and editable policy](shared_pass_phase2.md).
- [Paired kernel outputs, counts and exploratory estimate](/work/merlin/tasks/phase2-target-adapters-20261019/out/artifacts/target-adapters/gemmini/overwrite-exploratory-count-delivery-v1/delivery.json).
- [Spike accuracy audit and fixed-model limits](/work/merlin/tasks/phase2-pass-integration-20261018/out/artifacts/verification/spike-accuracy-existing-evidence-v1/accuracy-action-plan.json).
- [Exact capsule packet, sources, coverage and refusals](/work/merlin/tasks/phase2-dev-capsules-20261019/out/artifacts/verification/development-sources/model-scale-public-packet-v1/delivery.json).
- [Full-model comparison boundaries](/work/merlin/tasks/phase2-target-adapters-20261019/out/artifacts/target-adapters/gemmini/reference-fullmodel-comparability-v1/report.json).
- [Live queue status](/work/merlin/tasks/phase2-el4-explore-20261018/out/artifacts/perf-studies/gemmini/el4-20261018/phase2-loop-20261020T190502Z/LIVE.md).
