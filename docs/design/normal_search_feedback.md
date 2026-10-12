---
title: Normal compiler search and diagnostic feedback
kind: design
status: draft
owner: merlin-experiments
last_verified: 2026-10-11
related: [aws_phase2_corpus_search, phase2_cpu_timing_study, independent_performance_feedback]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase1/feedback/selfcheck.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/ordinary_fast_feedback.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/ordinary_calibration.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/search_coordinator.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/stage_prompt.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/public_boundary_semantics.py
  - src/merlin/perf/host_cfg_activity.py
  - src/merlin/perf/executed_trace.py
  - src/merlin/perf/calibration_plan.py
---

# Normal compiler search and diagnostic feedback

This is the minimum proposed extension of the ordinary experiment route.
It reuses Phase 1's public target, specification, ISA, toolchain, neutral runtime
and observers. It supplies no compiler implementation or preferred schedule.
A successful Phase 1 compiler and a qualified performance screen have not been
demonstrated by the source checks. Phase 2 admission still requires that actual
functional result; descriptive candidate inspection can be developed beforehand.

## Current implementation and its limits

| Item | Available now | Remaining limitation |
| --- | --- | --- |
| Phase 1 selfcheck | Shape coverage, offload census, caller layout, ISA lint and original functional execution | Placement declarations and emission counts do not prove runtime coverage or speed |
| Normal cheap feedback | Complete public compile/Spike roster, emitted action-site total and executable bytes | No dynamic census or complete host-cost model; two scalar features omit important interactions |
| Normal search | Frozen revisions, bounded queue, measured portfolio, repeat ranges, priorities and final reserve | Cheap feedback does not populate the beam; retaining measured revisions is not automatic branching |
| Model producer | Live public calibration and separate public validation | Normal round construction does not install that owner; default screening is UNKNOWN |
| Public graph generation | Complete public programs and connected logical boundary cuts | Cut constructor currently supports only its explicit float32 subset; quantized/cast and broader reduction coverage is incomplete |
| Performance objective | Complete measured development roster | Complete host/model costs outside member timers remain UNKNOWN |

The prompt must describe this actual objective and available tools. A request to
optimize an unmeasured complete model, paired with a prohibition on measuring
any complete public program, cannot establish an end-to-end improvement.

## One candidate observation record

Use the candidate's freshly emitted IR and linked ELF, original public inputs,
selected facts and original tools. No certified comparison compiler is needed
for a descriptive record. After Phase 1, the same producer can observe both
issued revisions on identical public members.

| Question | Observation | Evidence and limit |
| --- | --- | --- |
| What reaches each execution lane? | Source operations, shapes, dtypes, work extents, emitted placement and explicit refusal reason | Join source nodes to actual emitted entries; declared placement alone is insufficient |
| Where does representation change? | Casts, quantization, materialization, layouts, copies and publication boundaries | Source/IR observations; logical bytes are not physical traffic |
| What does the host emit? | Arithmetic/conversion families, loads/stores, loop structure, allocations and code/frame/table footprint | Reuse host CFG and linked observers; unsupported control flow stays UNKNOWN |
| What actually repeats? | Executed instruction and command counts, requested addresses and repeated setup | Optional bounded functional trace; attribute entry/harness/startup/readback separately |
| How much extra work is issued? | Useful work, decoded issued extents, tails and repeated operand requests | Derive extents from the current stream and selected ISA facts; do not infer cycles from counts |
| Which resources might constrain it? | Simultaneously live logical/declared allocations, reservations and resource intervals | Physical residency, bank conflicts and overlap need additional evidence |
| Did the change reach the executable? | Source/IR/object/ELF identities and structural differences | Bind the artifacts actually built, including composed transform selections |
| What is still unknown? | Missing joins, unsupported terms, incomplete windows and unavailable prices | Never remove an unknown cost or turn it into zero |

Every observation carries compiler/member/tool/configuration identities, its
exact boundary, completeness and collection cost. State whether a quantity is
declared, derived, functionally executed, modelled or RTL measured. Report the
candidate's own observations and differences, without prescribing a transform.
CCA may join source, representation, resource and dependency facts to editable
compiler owners; it is not a comparison against an expert implementation.

Offload reporting keeps different denominators separate: operation count,
arithmetic work and movement volume. A timed host/device percentage requires
attributed timing. Full-trace instruction totals do not provide that percentage.

## Evaluation ladder and escalation

| Level | Use | Escalate when |
| --- | --- | --- |
| Compile and structural inspection | All sizes: legality, coverage, interfaces, effects, resource feasibility and emitted differences | A result depends on execution or an unsupported source/IR construct |
| Original functional execution | Required output checks on selected public cases | Numerical checks fail or required RTL tiers remain pending |
| Bounded functional census | Selected cases: actual host work, command multiplicity, replay and requested addresses | Cycle price, cache, physical traffic, contention or overlap decides the choice |
| Derived bounds and priced model | Large public shapes/graphs: feasibility, headroom and conditional ordering | Prices are absent, the regime is new, intervals overlap or a prediction contradicts a bound |
| Selected CPU RTL | Complete decisive public components/programs with the original runtime and timer | Resolve a specific uncertain ranking or extend independent model validation |
| Protected final execution | Frozen compiler, original full output and instruction gates | Final hardware performance comparison after freeze |

Do not emit an unbounded instruction log for every large case. Prefer a bounded
online aggregate over the admitted functional runner when available; otherwise
use a streaming trace with explicit retirement, wall, memory and output limits.
Truncation is an incomplete observation. Trace collection overhead is a tool
cost, not target execution time. Entry attribution and unsupported instructions
must remain visible. No in-runner aggregate is claimed implemented here.

Spike establishes functional execution and instruction activity, not target
cycles, cache hits, physical bytes, array utilization or overlap. Native host
timing can reject pathological construction costs, but does not price another
host ISA. Neither replaces selected RTL timing.

## A model richer than roofline

A roofline bounds arithmetic and movement service. It cannot price a program
that performs substantial preparation, conversion, reduction, replay or dispatch.
Represent those stages and emitted device work as a dependency/resource graph.
Serial stages add; independent stages may overlap only under supported
concurrency assumptions. Keep inclusive timers separate from contained costs.

Derive geometry, capacities, legal extents, ports, queue limits and available
latencies from the selected RTL/ISA facts. The minimal software specification
supplies mathematical semantics, numerical/effect permissions and interfaces.
It need not encode a scheduling recipe. Host dependency/cache costs and external
memory service that cannot be extracted require public controlled measurements.

Start with existing observed feature and cost-composition utilities. Price only
terms backed by current evidence; add a detailed event model only when it can
resolve an observed ambiguity. Do not begin by constructing another full RTL
simulator. Target instruction/resource interpretation belongs in target data or
the independent OOT provider; generic composition and host observers belong in
Merlin.

Calibrate on independently public alternatives, then validate on distinct public
sizes, schedules and crossed resource regimes. Preserve current qualification
gates. Refit or abstain outside the domain. Historical reference-compiler labels,
protected workloads and coefficients fitted to them are excluded. A model
estimate is screening evidence and never an independent timing replicate.

## Search protocol

1. Preserve the successful Phase 1 compiler as the immutable initial baseline.
   Establish original numerical/ISA gates and the actual measured objective on
   admitted public programs. Until then, prepare observers and corpus purposes;
   do not start an optimization run under a fabricated admission.
2. Use the incumbent's measured costs and independently derived headroom to
   choose high-value questions. Prefer potential absolute savings across the
   public objective over the largest isolated utilization ratio. Unknown bounds
   keep measured expensive cases visible.
3. Let up to three authors propose distinct hypotheses from a recorded parent.
   Each states scope, expected emitted difference, conditions, regression risks,
   numerical obligations and an observation that would refute the hypothesis.
   The coordinator integrates coherent edits and freezes each revision. Shared
   toolchains and corpus are readonly; retain source once per content identity.
4. Inspect all contenders cheaply. Reject independently established legality or
   numerical failures. Record unchanged emission; deduplicate equivalent artifact
   requests only under exact input/build identity. A source edit is not evidence
   of a performance change.
5. Run affected public functional/census cases and qualified model screening.
   Keep provisional candidates and unresolved alternatives separate from the
   measured incumbent. A partial result cannot become a winning score.
6. Spend RTL on a small diverse shortlist and the cases that distinguish their
   predicted behavior. Include a complete connected member when an inter-operation
   claim depends on host preparation, movement or consumers. Record collection
   costs and revise the shortlist when screening has poor ranking fidelity.
7. Measure the complete required public roster before promoting a new incumbent.
   Compare against its recorded parent/current incumbent, preserve repeat ranges
   and check independent transfer cases. Only then compose independently useful
   changes and measure their interaction.
8. Stop under the existing budgets/reserve. Count a plateau only from completed
   diverse valid comparisons; failed or unresolved observations are not evidence
   that performance saturated. Stop edits, drain work, select measured bytes and
   run the fresh complete final gates.

Existing normal search stores a measured portfolio. A host-owned parent-selection
action and a provisional shortlist would be needed for automatic branching;
they are proposed extensions, not capabilities of the current beam. Keep any
future action inside original lineage/edit authority and resource limits.

Changed-code scope is a scheduling aid, not a sound regression proof. Re-emission
must reopen actual products and dependencies. Global/shared changes affect all
uncertain members. Simulator result reuse remains disabled under the current
reset/non-file closure; existing verified object reuse does not change that.
Full final evaluation stays fresh.

## Public cases that make feedback useful

Every member declares a performance question and a complete output/cost boundary.
Use independent public semantics and facts to vary tails, strides, reuse counts,
escaped users, simultaneous live values and resource transitions. Preserve the
property under investigation when reducing a witness; smaller extents alone
do not preserve capacity, overlap, reuse or fixed-cost balance.

Pair isolated operation witnesses with connected producer/consumer programs,
host transforms, reductions and repeated/shared-input cases. Keep compile-only
large shapes even when their execution exceeds the budget. Admit bounded public
complete programs for timing; protected validation programs remain inaccessible.
Source membership, interfaces, emitted placement and original full-output gates
must be checked before transferring a result between cases.

## Critical implementation order

1. Expose the candidate observation record through the original Phase 1 broker;
   reuse it in normal Phase 2 retained workers.
2. Align the prompt with actual objective, available actions and original numerical
   contracts. State editable compiler/host-IR surfaces; preserve frozen runtime.
3. Add bounded dynamic census and expand explicit public representation/reduction
   coverage where current constructors refuse it.
4. Install qualified screening in the normal launch lifecycle and add a provisional
   shortlist with decisive member measurements. Keep complete measured promotion.
5. Measure tool costs and ranking on AWS after Phase 1 succeeds. Extend model
   terms or corpus only for concrete blind spots found by those measurements.

These are proposed remaining changes. They do not claim an automatic performance
result, a fast model qualification, whole-model timing or a Phase 1 success.
