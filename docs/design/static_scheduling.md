---
title: Portable static scheduling and completion boundaries
kind: design
status: current
owner: compiler
last_verified: 2026-11-16
related: [core_dialects, compiler_plane]
code_refs: [src/merlin/compile/scheduling, src/merlin/xdsl_dialects/schedule.py, src/merlin/llvmlower/floating_contractions.py]
---

# Portable static scheduling and completion boundaries

## Problem and ownership

The existing shared dense planner chooses tiles, traversal and storage. The
`schedule` dialect records timing decisions, and `timed_schedule` checks them.
Neither constructing a dialect operation nor successfully lowering a model
establishes that the emitted instructions implement a legal physical schedule.
The earlier integration delegated that last step to an existing target compiler.
An independent backend needs reusable construction and realization, not just
annotations reconstructed after target scheduling.

Merlin owns deterministic scheduling, dependency/resource checks, completion
boundaries and plan-to-emission correspondence. A target owns instruction
selection, exact operand effects, issue throughput, reservation intervals,
latency, delay/wait encodings, register allocation and binary decoding. Timing
facts must come from the selected hardware/model contract. Candidate-authored
cycle claims are not evidence. Numerical replacement/fusion permission remains
independent of scheduling legality.

## Two completion models, not one guessed clock

Fixed-latency instructions supply existing `TimedEffects`: complete byte access
intervals, resource reservation intervals and a completion age. Read/write
intervals include registers, memory and hidden state; capacities include issue
ports and shared read ports, not just compute engines. Every actual instruction
also supplies its natural issue advance. Unknown effects, capacities or times
refuse. All intervals use one explicitly named clock and half-open endpoints.

Event-driven operations retain existing `Call.produces`/`Call.awaits`. A DMA,
queue submission or external engine with unknown duration must use a target
completion wait. A scheduling decision cannot turn that wait into a fixed delay.
The initial completion pass waits immediately after each asynchronous launch,
inside the original counted loop if applicable. This conservative policy permits
no outstanding asynchronous work at a loop backedge or exit. It preserves
existing unit, staging and configuration annotations, which still require the
existing target-specific checkers/spellers.

These two paths compose at drained boundaries. Timing within a fixed block is
relative to its entry after prior work completes, not a promise about absolute
wall-clock time. A variable-duration wait starts a new local timing scope.

## Dialect decisions

Keep `schedule.time_annotation` for issue/completion cycles and `schedule.overlap`
for realized lifetime overlap. Add only two decisions:

- `schedule.drain`: a cycle and clock at which all fixed-latency work in this
  plan has completed. It must be unique, match the plan clock and be at least
  every projected completion. Materialization must reach it before returning.
- `schedule.await_completion`: a handle bound to an actual asynchronous source
  call. Its consumer waits immediately after that call. The token is read from `Call.produces`, never copied into an unrelated
  side table. The completion-plan consumer inserts a target wait carrying the
  matching `Call.awaits`. Every launch must be covered exactly once.

A fixed-timing plan and an event-completion plan have separate consumers; an
unknown or inappropriate operation refuses instead of disappearing. Plans bind
the current kernel digest. Completion handles use static statement paths through
existing counted loops; they do not clone loop execution into a second IR.

## Initial scheduling algorithm

Preserve source issue order. `serial` drains each instruction before the next;
`overlap` permits concurrent lifetimes only when complete projected effects and
capacities allow it. For each instruction:

1. Respect the previous instruction's natural issue advance.
2. Respect every earlier overlapping RAW, WAR and WAW byte interval.
3. Find the earliest resource-feasible start among reservation boundary changes.
   Do not scan a million individual cycles for a long-latency instruction.
4. Record issue/completion annotations and a final drain, then run the existing
   independent plan verifier on the result.

The first version is a deterministic source-order scheduler, not a global
optimizer. It can overlap independent matrix/vector engines or pipelined units
without reordering operations. Multiple-issue machines may declare zero advance
and express their issue capacity as a resource. Resource occupancy and output
latency remain distinct. Stateful instructions need explicit effects; a missing
projection never means independence.

## Realization and verification

The materializer receives target-selected delay instructions. It preserves every
original call exactly once and in order, fills the planned issue gaps and drains
the tail. Delays must have positive natural advance, no byte effects, and no
resource lifetime beyond their advance. Recompute timing from the emitted calls,
check correspondence and run memory/resource validation on the actual resulting
stream. Reject dropped/duplicated/changed source calls, wrong delays and shortened
tails. The target must still decode the final binary and validate that it denotes
this stream; a Python plan cannot certify unseen machine bytes.

For event waits, the target supplies a concrete wait call with exactly the
required token. The existing C emitter must spell `awaits`; absence of a speller
refuses. Target qualification must prove that wait really observes completion.

## Blocks, loops and architecture applicability

Apply fixed scheduling separately to lowered basic blocks, with entry drained
and explicit tail drain before transfers of control. The OOT owns branch timing,
delay slots, loop-counter instructions and CFG verification. Do not claim global
control-flow safety from a block-local checker. Preserve existing compact dense
loops; do not unroll a full model merely to obtain timing annotations.

This contract covers an explicitly scheduled scalar instruction stream, multiple
fixed-latency engines, and pipelined or multiple-issue devices through differing
provider data. A queue-driven accelerator uses explicit completion tokens and
waits. An interlocked target can realize an appropriate conservative policy but
must explicitly state the issue/retirement assumptions; observed fast runs are
not fixed latency. Multiple unrelated clocks, cross-iteration software pipelining,
nonblocking DMA overlap, instruction reordering and modulo scheduling are later
extensions requiring additional proofs, not initial defaults.

## Authoring inputs and validation

Export the scheduler and completion consumer through the existing closed
compiler-library selection and test them under isolated imports. Split pure
floating contraction outlining/signature/bridge generation from the installed
host toolchain driver so an author can reuse source partitioning without importing
backend discovery, reference models or graders. Host execution setup and reviewed
experiment admission remain separately required; export is not admission.

Tests must include multiple synthetic architectures, RAW/WAR/WAW and capacity
hazards, delayed operand reads, long latency, malformed/unknown effects, explicit
overlap decisions, real delay realization, mutation rejection, completion waits
inside counted loops and mandatory speller behavior. Use an independently
implemented small execution oracle, not only checks that mirror the scheduler.
Exercise the isolated export and an existing target provider separately. No OOT
compiler-generation or optimization-agent loop is started by this work.

## Implemented entry points

`static_schedule.schedule_fixed_block` constructs a plan. Its companion
`materialize_timing_plan` returns `TimedEmission` with actual target calls and
source positions; `verify_timed_emission` checks an independently reconstructed
stream against the same plan. `completion_schedule.make_completion_plan` and
`apply_completion_plan` produce and consume explicit immediate waits. These APIs
are public leaves in the example compiler-library selection.

The initial target integration exports a conservative occupancy bound directly
from its existing timing provider, before scheduling. It uses whole-machine
serialization rather than approximating detailed alternative ports. The shared
pass inserts actual delay instructions; only the target timing verifier and
encoder run afterward. Fixed matmul and vector programs execute in the selected
simulator. The provider remains model-derived diagnostic evidence, not a new
independent RTL qualification. Its fixed-only adapter rejects DMA and CFG;
queued completion is independently exercised by a compiled synthetic target.
Whole-model scheduling still uses the existing target pipeline until event and
CFG boundary integration is separately implemented and tested.

## Validation recorded for the initial implementation

193 focused tests cover scheduling, event completion, dialect verification,
contraction outlining, dense planning and isolated compiler-library use. Small
exhaustive cycle/occupancy oracles check earliest source-order placement. Tests
mutate source identity, issue cycles, padding effects, tail drains and completion
coverage. The selected target integration executes fixed matmul and vector
programs using only the exported library; removing emitted delays fails its
independent timing verifier. No full-model shared-scheduler replacement is claimed.
