---
title: Develop functional and performant compilers together
kind: design
status: draft
owner: merlin-experiments
last_verified: 2026-10-23
related: [joint_compiler_author_requirements, ../guides/shared_pass_compiler_generation, ../guides/shared_pass_phase2, ../guides/phase2_resource_floors, ../guides/spike_timing_boundaries]
code_refs:
  - src/merlin/compile/scheduling/schedule_pipeline.py
  - src/merlin/compile/scheduling/compact_dense.py
  - src/merlin/perf/calibration_plan.py
  - src/merlin/perf/fast_estimate_validation.py
  - src/merlin/perf/phase2_feature_calibration.py
  - src/merlin/perf/staged_roofline.py
  - packages/merlin-experiments/src/merlin_experiments/model_observation.py
  - packages/merlin-experiments/src/merlin_experiments/chain_observation.py
---

# Develop functional and performant compilers together

The joint workflow reuses the native Phase 1 authoring, snapshot and evaluation
infrastructure for correctness and performance development in one session.
This document describes its objectives and acceptance requirements; it does not
claim every mechanism is selected, model timing is qualified or all workloads
have passed. The compiler remains editable while functional coverage, scale and
performance develop together. See [execution design](joint_campaign_execution.md)
and [setup requirements](../guides/artifact_evaluation.md) for the concrete interfaces.

## Objective and evidence

Minimize **estimated whole-SoC execution cycles** over the declared development
workloads. Numerical and functional checks are mandatory acceptance gates;
passing them alone does not meet the compiler-development objective. Include
host arithmetic, packing performed at execution, transfers, dispatch, device
execution, synchronization, and carried-state updates within the selected
execution boundary. Record cold/warm conditions and all excluded setup or output
verification separately.

Instruction counts and requested bytes are explanatory features. Neither is the
performance objective, and neither independently establishes target cycles. An
uncalibrated functional Spike counter must not become a cycle score through a
change of label. Simulator walltime measures experiment cost.

Track utilization and waste beside cycles: useful work, issued/padded work,
resource occupancy when observable, spill/reload traffic, launch/configuration
cost, and host/device stalls. A modeled useful-compute utilization envelope is
`useful_MACs / (structural_peak_MACs_per_cycle * estimated_cycles)` on matching
scope and clock definitions. It is not measured array occupancy. Extra padding
or busywork must never improve the workload objective or raise its fixed floor.

The primary score remains complete-workload cycles. Report every primary
workload and its worst regression; an aggregate must not hide a slow decode
path behind a faster prefill path. Use uncertainty intervals when selecting
close candidates. Final physical acceptance requires compatible gSIM or
FireSim measurements, numerical checks, and full timing/provenance binding.

## Derive the experiment before authoring

Create a new explicit source/cohort epoch with these roles:

| Role | Author access | Purpose |
| --- | --- | --- |
| Functional cases | Public source and checker endpoint | Numerical/state/ABI correctness, tails and refusals |
| Calibration training | Public source; timing evaluator owned by operator | Fit supported timing mechanisms |
| Calibration validation | Labels withheld from fitting | Check cycle error, ranking, and activity support |
| Performance development | Primary models, extracted subgraphs, and large shapes | Early compiler architecture and complete-workload optimization |
| Generalization validation | Separate unseen models/shapes/compositions | Evaluate transfer after selection |

Making a formerly private model a development workload changes its role in the
new epoch. It does not preserve an earlier unseen-validation claim. Update
source-role selection and the held-out-layer guard consistently; do not waive a
historical guard or expose historical hidden answers under a public label.

Extract connected regions from the original framework graph, preserving typed
live-ins/live-outs, constants, layout, quantization, aliasing, mutable state, and
source identities. Cover operators, producer-consumer chains, residual branches,
normalization/quantization boundaries, attention/cache updates, and complete
blocks. Include unfused and legally fused alternatives. Verify each extraction
against its original computation, then verify recomposition and final full-model
execution. Isolated operators cannot establish composition or fusion legality.

Build shape ladders around mechanisms: tails and very small dimensions, reuse,
scratchpad/accumulator capacity, spill transitions, queue occupancy, cache working
sets, and larger command streams. Reduced cases must retain the mechanism being
studied; scaling dimensions down until every buffer fits changes the experiment.

## Construct the roofline at intake

Use selected RTL facts to derive structural compute, storage, issue, and transfer
ceilings as available. Derive useful work and independently justified compulsory
interface crossings from source semantics. Publish the initial partial floor
before compiler authoring; unavailable rates or demands stay explicit.

Keep three views separate:

1. A **workload lower bound** from compulsory useful computation and transfers.
2. **Emitted-schedule demand**, including padding, reloads, spills and redundant
   configuration, which explains removable waste.
3. A **SoC cycle estimate** with service, dependency, queue, cache, and completion
   behavior, calibrated against compatible physical observations.

The existing `staged_roofline` and `capsule_roofline.compose_service_floor`
interfaces support partial structural bounds. Sum demand sharing one resource;
combine independent resource lower bounds with a maximum. Serial sums require
evidence of serial completion. A maximum lower bound does not establish achieved
overlap. Requested client bytes do not establish DRAM traffic or sustained
bandwidth. Report partial-floor ratios as partial; do not manufacture attainment
or complete latency from missing host, traffic, or completion terms.

The roofline directs experiments: identify the active bottleneck, the remaining
gap, the largest unresolved term, and an experiment that can distinguish likely
causes. Recompute emitted-schedule demand for each revision while keeping the
workload floor fixed.

## Timing model and simulator boundary

Use a versioned, operator-owned cycle evaluator. Its SoC contract must identify
the CPU, memory/cache/TLB regimes, command issue/backpressure, DMA, local storage,
accelerator execution, resource sharing, synchronization, and final completion.
Source facts constrain structure; measured calibration supplies prices and
supported interaction behavior. A kernel-call fit alone cannot authorize a
complete-application prediction.

Reuse existing calibration planning, feature calibration, grouped validation,
and domain/refusal interfaces. Keep variants and repeats from one source group
together when splitting TRAIN and validation. Validate absolute cycle error,
ordering for meaningful measured separations, and mechanism/activity coverage.
Pin executable, inputs, engine, hardware, memory regime, counter window, features,
model, and source groups. Freeze an evaluator version during compiler
comparisons; the compiler author cannot edit it to lower its own score.

Choose simulation by evidence coverage, not a universal tensor-size cutoff:

| Situation | Required action |
| --- | --- |
| New numerical operation, state effect, ABI, or emitted instruction behavior | Run the applicable functional and physical correctness checks |
| New timing mechanism, resource interaction, or cache/capacity regime | Measure the smallest gSIM cases that preserve that mechanism; add independent controls |
| Large development case inside the evaluator's validated domain | Use the SoC cycle estimate for search and run full numerical/state checks with Spike |
| Unsupported activity, feature range, memory regime, or overlapping estimate intervals | Report UNKNOWN/ambiguous; acquire targeted calibration or a compatible physical comparison |
| Final selected compiler and reference-comparable workload | Run complete FireSim validation with matched execution boundaries |

Small cases do not license extrapolation across cache, capacity, or queue
transitions. If a proposed estimator factors scalable repeated work into
mechanism units, validate that factorization and its interaction terms on
independent larger checks. A rejected generic feature-range extrapolation is not
fixed by removing the range gate. Preserve failed and timed-out observations.

Enforce simulator access per capsule in the task manifest and broker. Small
functional/calibration cases expose their selected Spike and gSIM tiers; large
development cases expose Spike and the SoC estimator. A large functional Spike
case remains mandatory while its cycle estimate is UNKNOWN. The operator can
select additional mechanism-preserving physical calibration cases explicitly;
the author's large-case endpoint must not silently become unrestricted gSIM.

Additional calibration may occur throughout development. Issue a new model
version, validate it independently, and re-evaluate compared compiler revisions
under the same version. Keep raw observations and estimates separately typed.
Final FireSim validation should remain independent of calibration labels.

## Compiler abstractions and optimization search

The fresh author receives reviewed shared passes, scheduling IR, host codegen,
runtime contracts, target facts/ISA, and explicitly public development sources.
Prior target compilers and tuned reference implementations remain operator
inputs. Target support used for execution must not supply a hidden compiler seed.

Preserve the following compilation boundaries:

`typed computation -> layout/fusion -> device operations -> storage/scheduling -> target emission -> actual-program checks`

Keep bounds, numerical policy, dependencies, ownership, persistent configuration,
and completion explicit. Run passes on the selected IR and lower their actual
results. A pass declaration or report without changed emitted/executed code is
not pass consumption. Preserve compact checked loops for large domains where
supported; CPU loop control does not authorize prohibited hardware-loop commands.

Inventory these mechanism families and expose legal bounded choices:

| Family | Choices and obligations |
| --- | --- |
| Computation/layout | Fusion and epilogues, layout propagation, transpose elimination, constant preparation, reduction ordering and numerical legality |
| Tiling/reuse | Multi-level tiling, traversal/dataflow, operand residency, tail handling, padding avoidance, resource-aware shape selection |
| Storage | Lifetime analysis, buffer reuse, accumulator retention, allocation, spill/reload plans, bank/alignment constraints |
| Scheduling | Double buffering, prefetch, software pipelining, dependency scheduling, overlap, queue-pressure limits and completion |
| State/commands | Configuration liveness and elimination, compatible batching, launch amortization, synchronization minimization |
| Host/device boundary | Shared host fusion/requantization/layout codegen, destination forwarding, transfer ownership, asynchronous lifetimes, cache/state updates |

Applicability follows operation semantics, shape/layout, numerical policy, and
declared resources. Model names, provenance IDs, goldens, and benchmark constants
must not choose production strategies. Numerical reassociation, scale hoisting,
and rounding substitutions require their actual premises. In particular, shared
scale syntax does not prove equal per-head values or a nonzero runtime scale.

The existing shared-pass contract exposes typed computation, scheduling, and
target-IR consumption boundaries. Audit the actual selected library inventory;
do not assume every listed mechanism has a complete usable implementation.
Current documented limitations include finite `Kernel` versus arbitrary counted
loops, declaration checks versus realized pass effects, and missing transport of
the shared-pass requirement through the separate sealed model-worker protocol.
Fix a missing integration explicitly before relying on it in the new runner.

## Joint development and the later refinement stage

Every accepted compiler revision must preserve all required correctness cases,
compile and execute the declared large development ladder, and report complete
cycle/utilization feedback. Keep pending and refused cases visible. Maintain a
Pareto set by per-workload cycles, numerical eligibility, and resource waste.
Prefer experiments addressing the largest measured or modeled cost.

Do not freeze a compiler simply because the small functional roster passes.
Before the later refinement stage, require complete development-workload
coverage, a usable validated timing domain, explicit unsupported cases, realized
shared-pass use, and the predeclared per-workload performance gates. The new
experiment must define those gates before authoring rather than inventing a
successful threshold after results arrive. Later refinement can then focus on
tile/traversal choices, pass order, overlap, and shape heuristics while preserving
the established compiler architecture and accuracy contracts.

The accompanying [author requirements](joint_compiler_author_requirements.md)
describe the author-safe part. Campaign measurements and prior implementations
belong in the operator handoff, outside fresh-author grants.
