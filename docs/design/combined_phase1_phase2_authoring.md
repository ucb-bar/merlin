---
title: Combined functional and performance compiler authoring
kind: design
status: draft
owner: merlin-experiments
last_verified: 2026-10-21
related: [fresh_compiler_origin, compiler_authoring_prompts, shared_phase1_scheduling, independent_performance_feedback, component_phase2_workflow]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase1/component_origin.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/providers/authoring_prompt.py
  - examples/gemmini/phase2/continuing_campaign.py
  - src/merlin/transforms/pipeline_state.py
  - src/merlin/xdsl_dialects/schedule.py
  - src/merlin/perf/capsule_roofline.py
---

# Combined functional and performance compiler authoring

This is a proposed **separate, opt-in experiment**, not a change to archived
Phase 1 or Phase 2 runs. One fresh OOT author develops a functionally complete
compiler and continues optimizing the same compiler before freeze. Passing the
small public correctness set does not end authoring. The author starts from an
inert scaffold and admitted RTL/software facts, never from a previous Gemmini
backend. Previously measured compilers may be operator-owned comparators, but
their implementation and protected validation sources stay outside the author
bundle. The run retains exact prompt, tool, source and compiler identities.

## Workload ladder and simulator choice

Derive a public development ladder from captured typed graphs and independent
source references. Retain each original graph identity and numerical contract.
Select small single-operation witnesses; connected producer-consumer, fan-out,
layout and host/device boundaries; and stateful QK, softmax, PV and carried-KV
sequences. Generate smaller representatives by changing public shape parameters
and inputs while preserving their operation and state contracts. Test both sides
of plausible fusion cuts, then compile the connected slice and compare every
live output and state value. A composition claim needs the connected program's
own compile, dispatch and execution receipt; separate capsule passes cannot
establish it. Protected model captures and answers remain evaluation-only.

All admitted small cases receive functional Spike checks and the required
elaborated-RTL correctness certificate. Only a selected subset receives gSIM
cycle labels for SoC model construction: cases must vary host work, FP32 work,
command arrival, DMA bytes, compute, readout, cache behavior, queue pressure,
capacity, tails and connected reuse. Hold out entire source groups and reserve
mechanism variants as extrapolation checks. Reopen compatible public labels
first. A case crosses to **Spike-only performance ranking by model support**, not
by a fixed tensor size: its executed event ranges must lie inside a validated
domain and held-out prediction/ranking gates must pass. Unsupported large cases
remain numerically testable; request the smallest targeted gSIM bridge probe
that extends coverage before using a cycle estimate to promote them. Run the
larger public GEMM, convolution and complete-attention programs in Spike during
authoring, with complete outputs and whole-invocation timing windows.

## Shared compiler surface

Expose reviewed, source-pinned generic passes and host codegen through a
versioned author-visible interface. It should accept typed IR, explicit
numerical and capability contracts, selected pass names and order, and return
transformed IR plus a decision/analysis receipt. This is an interface proposal,
not an existing extra tool grant. The fresh backend supplies target dialect,
physical layout, commands and instruction emission. It must not duplicate
Merlin's scalar postprocessing, whole-tensor fills/copies, allocation, or
dispatch loops when the shared host route owns them.

The schedule dialect records selected tiling, layout, placement, residency,
packing, accumulator lifetime and dispatch decisions. PassSpec prerequisites
and PipelineSession identity stamps reject stale analyses or illegal pass
orders. Each selection must be checked against the emitted program. The author
may change pass implementations or legal order and add passes; useful changes
become semantic, shape- and capability-driven rules rather than capsule-name
cases. The initial mechanism set covers multi-level tiling, operand and output
stationarity, capacity-aware spilling, packing hoists, producer-consumer fusion,
native readout, configuration cleanup, command batching, DMA prefetch, double
buffering and legal load/compute/store overlap. Preserve exact reductions,
tails, aliases, bank conflicts, completion dependencies and all live results.

## Metrics and automatic run

Construct a roofline record for every admitted case at launch from the selected
RTL, memory configuration and software semantics. Record useful work,
compulsory movement and host work, structural necessary floors, independently
supported service ceilings, legal overlap, missing terms and uncertainty.
Qualifying an *attainable* envelope needs independent measurement and
falsification evidence; a compute-peak floor alone is not full attainable
attainment. Mixed CPU/accelerator attention includes FP32 softmax/PV and carried
state. Report compute-peak utilization separately.

Annotate Spike over the full invocation: Rocket issue/FP/cache/memory, RoCC
arrival, reservation queues, DMA, Gemmini service, synchronization and
completion. Fit only missing prices from source-group-separated public gSIM
labels, validate on held-out groups and report per-family errors, pair ordering,
support and abstentions. Freeze a model during candidate comparison. A model
revision gets a new identity and replays both comparator arms; no candidate
timing is silently relabeled. Instruction/command counts and bytes diagnose a
cycle change but never select a winner by themselves.

Use a durable automatic queue for a 12-hour run. Prioritize missing evidence,
then the capsule with the lowest supported roofline attainment. For each frozen
candidate: compile, check complete outputs and device coverage, measure
whole-call Spike cycles, compare to its unchanged control and current incumbent,
retain or reject, and enqueue the next bottleneck. Correctness and legality are
hard gates. The 30% independently qualified attainable-roofline target is a
time-bounded goal, not permission to extend the clock or call an unknown value
a pass. Freeze compiler edits for the final two hours; repeat matched FireSim
before/after measurements on the large public kernels and selected connected
slices. Full-model execution remains a separate claim. Report all refusals,
model errors, per-case attainment and the matched geometric-mean comparison.
