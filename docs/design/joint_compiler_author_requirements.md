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
historical solutions and operator-private reports are outside its inputs. The
explicitly granted AutoComp archive and pinned Exo examples are allowed source
references; they do not grant access to prior compiler implementations.

Optimize **estimated complete-workload SoC cycles**, including required host
work, transfers, device execution and synchronization within the declared
execution boundary. Numerical and functional correctness remain mandatory.
Small functional cases and large performance cases both matter throughout
development. An instruction-count reduction alone is not a performance win.
The operator owns the versioned cycle evaluator and its calibration.

Prioritize the selected INT8 deployment contracts: integer contractions use
INT8 operands and the declared INT32 accumulation semantics. Preserve quantization
scales, rounding, saturation, overflow and permitted error bounds. Host operations
that the reviewed recipe leaves unquantized remain part of correctness and whole-SoC
cost. Do not silently replace integer contractions with floating-point host work or
change the numerical recipe to improve performance.

Select a small named check set after each edit: the relevant numerical corner,
the changed scheduling/fusion boundary, and a medium or large performance case.
Do not sweep all small capsules on each iteration. Reuse native exact-identity
results; never infer unaffected correctness from a self-declared family hint.
Broaden checks at candidate milestones and require complete correctness coverage
before final promotion. Every original capsule retains its declared gSIM
correctness obligation (intentional refusals retain their refusal gate).
Extra calibration gSIM needs a specific missing mechanism or timing question
after cheap correctness; calibration permission alone does not request a run.

## Compiler boundary checklist

Before broad testing, run one tiny applicable case through emission, actual target
consumption, linking, Spike and complete numerical comparison. Tool readiness is
not execution success. Use the selected schemas and ABI as the authority:

- Every declared command must exist and accept the input stage assigned to it.
  Downstream emission consumes transformed target IR, including its attributes;
  it must not expect the original interface or discard pass results.
- Emit a raw schema-valid command-buffer document, with its required ABI version.
  Do not wrap it in an extra JSON envelope. Respect the declared output-file and
  stdout channels; write diagnostics to stderr, never into the target artifact.
- Derive entry symbols, argument layout and any completion helper from the
  selected ABI. Resolve every called helper and preserve actual completion before
  output readback. A link or stack-report failure is not a numerical mismatch.
- Bind every named output to the correct pointer, shape, dtype, stride and lifetime.
  Preserve multiple live-outs and state updates; never assume the first pointer
  is the only output. Compare every externally visible output.
- Whole-program L0/L1 non-applicability is valid only when the native harness
  establishes it and Spike checks the whole program against its source reference.
  A skipped tier is never a self-declared pass. Preserve expected negative cases.

Fix shared manifest/entrypoint/ABI failures before repeating affected requests.
Use affected numerical boundary cases after a fix; do not change the public
contracts or bypass a check. In particular, exercise applicable tails, aligned and
unaligned activation/readout, quantization rounding and saturation, convolution
stride/padding, and connected regions with multiple outputs or retained state.

## Performance and generalization

Read compiler_diagnostics.feedback_brief first. It combines scoped command/CPU
counts, requested movement bytes, useful/padded work and available compiler
packing, lifetime and fusion receipts without another target run. Preserve each
section's maturity: planned lifetimes, rewritten LLVM and runtime observations
are different evidence. Missing values are unknown, never zero. These summaries
do not add guest instrumentation or establish saved cycles. Use shared observation
producers when their corresponding passes run; do not fabricate receipt metrics.

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

Treat instruction interleaving and buffer scheduling as first-class search
dimensions. Construct dependency-correct schedules in the shared scheduling
dialect: overlap the next tile's load, the current tile's compute and the previous
tile's store when resources permit. Track buffer ownership through asynchronous
completion; preserve read/write hazards, accumulator lifetimes and configuration
dependencies. Tune traversal, buffer assignment, prefetch distance and outstanding
commands against the selected queue and memory limits. More buffering or fewer
fences is not automatically faster or legal. Compare serialized and pipelined
variants using the same numerical contract and timing evaluator.

The simple reorder-buffer queuing model concerns the accelerator's command
reservation and dependency machinery. Rocket remains a scalar in-order CPU.
Use feedback on admission stalls, ready-but-blocked commands, resource overlap
and completion to improve scheduling; do not assume out-of-order CPU execution.

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
