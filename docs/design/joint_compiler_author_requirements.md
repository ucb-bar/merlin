---
title: Requirements for joint compiler authoring
kind: design
status: draft
owner: targetgen
last_verified: 2026-10-21
related: [joint_compiler_development, ../guides/shared_pass_compiler_generation, ../guides/shared_pass_phase2]
code_refs:
  - src/merlin/targetgen/shared_pass_contract.py
  - src/merlin/transforms/pipeline_state.py
  - src/merlin/sched/ir/kernel.py
  - src/merlin/compile/scheduling/schedule_pipeline.py
---

# Requirements for joint compiler authoring

This author-safe brief is a proposed input to a fresh compiler experiment.
Selection into a task requires explicit source/library grants; this document
does not start a run, expose private sources, or establish qualification.

Build a general target compiler from the supplied computation IR, numerical
contracts, target facts and ISA. Use the reviewed shared compiler library,
scheduling abstractions, host codegen, and runtime interfaces supplied to the
task. The target compiler starts from scratch. Prior target implementations,
tuned reference code, historical solutions and operator-private reports are
outside its inputs.

Optimize **estimated complete-workload SoC cycles**, including required host
work, transfers, device execution and synchronization within the declared
execution boundary. Numerical and functional correctness remain mandatory.
Small functional cases and large performance cases both matter throughout
development. An instruction-count reduction alone is not a performance win.
The operator owns the versioned cycle evaluator and its calibration.

Use the roofline and resource feedback to identify bottlenecks and unused
capacity. Treat partial bounds and unsupported timing domains explicitly.
Distinguish useful work from padded/issued work, actual occupancy from modeled
utilization, and execution cost from compilation or simulator startup. Do not
improve a score with redundant computation, excluded transfers, or unaccounted
host fallback. Report regressions by workload and by important stage.

Preserve typed computation, layouts, bounds, reduction/rounding policy, state,
ownership, dependencies and completion through lowering. Consume the actual IR
produced by shared passes. Select a legal pass order and expose bounded policy
choices that later refinement can tune. Record pass realization and inspect
emitted programs; calling a pass and discarding its result is insufficient.

Consider the applicable general mechanisms: fusion/epilogues; layout propagation
and constant preparation; multi-level tiling and traversal; operand residency;
buffer lifetime/reuse; accumulator retention; spilling; bank/alignment and tail
handling; double buffering; prefetch and software pipelining; resource overlap
and queue limits; configuration elimination; batching and launch amortization;
and efficient host/device transfer and synchronization. Use supported shared
passes first. Add missing general mechanisms at the appropriate shared boundary;
target-specific mapping and instruction emission stay in the target compiler.

Reuse the supplied host code generator for host computation. Derive strategy
choices from semantics, shapes, layouts, numerical policies and capabilities.
Do not select behavior by model name, provenance ID, golden result or benchmark
constant. Reject unsupported transformations or retain a correct legal route.
Keep explicit numerical premises for scale movement, reassociation, fusion and
rounding. Follow the declared instruction policy, including any hardware-loop
restriction; CPU loops are a separate mechanism.

Develop across the complete declared shape ladder and connected subgraphs.
Exercise tails, small dimensions, capacity/spill boundaries, layout transitions,
producer-consumer combinations and carried state. Preserve complete live-outs
and cache updates. Passing isolated operations does not establish a full model.
Run the required numerical checks on large cases, even when physical simulation
is reserved for smaller mechanism-preserving cases.

Use evaluator feedback within its supported domain. Request targeted calibration
when a new mechanism or regime is unsupported, and preserve ambiguous/UNKNOWN
results. Keep working on independent supported cases while the measurement is
pending. Do not edit timing coefficients, remove validation gates, or relabel
functional simulator counters as measured cycles.

The finish condition includes functional/numerical qualification, complete
development-workload coverage, realized shared-pass use, and the experiment's
predeclared cycle/performance gates. A functional-only compiler does not meet
that finish condition. Keep pending checks and remaining limitations explicit.
