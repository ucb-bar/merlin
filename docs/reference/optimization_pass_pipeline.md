---
title: General optimization passes and Phase 2 integration
kind: reference
status: current
owner: compile
last_verified: 2026-10-20
related: [shared_compiler_passes, tiled_convolution, panel_transfers, shared_input_scheduling, copy_pipeline, ../guides/shared_pass_phase2]
code_refs:
  - src/merlin/compile/scheduling/block_schedule.py
  - src/merlin/compile/scheduling/shared_input_groups.py
  - src/merlin/compile/scheduling/shared_input_schedule.py
  - src/merlin/compile/scheduling/shared_input_verification.py
  - src/merlin/compile/scheduling/panel_transfers.py
  - src/merlin/compile/scheduling/copy_pipeline.py
  - src/merlin/compile/scheduling/dense_source.py
  - src/merlin/compile/scheduling/configuration_loops.py
  - src/merlin/compile/scheduling/command_compaction.py
---

# General optimization passes and Phase 2 integration

The shared library already contained compact dense blocking, operand reuse,
convolution gathers, temporary allocation, exact physical elementwise fusion,
configuration elimination and ordered host-loop unrolling. Those mechanisms
were not a complete reference-quality generated accelerator pipeline. The
following additions extend them without introducing target instruction code.
Their availability is distinct from a generated backend consuming them.

| Transformation | Existing foundation | Added mechanism | Present boundary |
| --- | --- | --- | --- |
| Resource-aware tiling | `compact_dense`, `dense_planning`, block loop orders | `schedule_tiled_convolution`: bounded output and K panels, continued accumulators | Finite raw integer convolution core; target bias/readout/completion remains separate |
| Operand lifetimes and load hoisting | First-use loads, staged RHS elimination, storage lifetimes | `schedule_shared_input`: retain full-K LHS across independent sibling output waves | Assigned full-K input stripe must fit; each result keeps its original boundary |
| Layouts and transfer aggregation | Gather runs and `Window2D` rectangle predicate | `panel_transfers`: affine source views, packed panels, wider rectangles and exact zero tails | Consumer must support the actual panel row stride and mapping; synchronous copies only |
| Dependency scheduling | Single-copy prefetch and completion-aware allocation | `copy_pipeline`: concurrent copies and bounded launch hoisting into supplied private slots | Explicit independent concurrency/wait semantics; all waits and launch order remain intact |
| Shared-input contraction grouping | Typed matmul and zero-initializer source checking | `find_shared_input_groups`, independent `verify_shared_input` | Consecutive closed signed i8/i32 sibling source regions; no movement across interleaved consumers |
| Exact epilogue composition | Single unary integer dense readout and physical pointwise fusion | `read_source_dense_readout_chain`, `schedule_source_dense(output_operations=...)` | Same-coordinate unary integer chain must exactly match an already qualified readout recipe |
| Configuration and command generation | Flat setup cleanup, LLVM index bridge, ordered unrolling | `hoist_loop_configuration`, `compact_command_stream` | Qualified invariant pure setup only; compacted loops preserve every dynamic primitive command |

## Inputs and pass composition

Software source supplies immutable value identities, shapes, affine access
maps, exact arithmetic, initialization and observation boundaries. Selected
RTL facts supply structural bounds. Software/ISA effects and independent
execution evidence supply persistence, completion and arithmetic capabilities.
Unknown effects or missing mechanisms retain their source route or refuse.

Reuse existing selected dense/storage contracts to derive usable rows, padded
row widths, primitive dimensions, reservations and alias windows. Source
formats establish element widths; a physical store width does not establish
an internal arithmetic width. Transfer limits are independent of compute tile
width. Copy concurrency requires separate qualification and fact-derived queue
limits; enough storage alone cannot establish overlap.

The intended ordering is source admission/grouping, jointly selected layout and
blocking, checked storage assignment, logical move/compute scheduling, optional
qualified copy pipelining, exact output composition, configuration cleanup and
bounded host-loop emission. Each changed layout requires replanning addresses;
changed allocation requires rechecking reuse and completion. The target encoder
and actual-operand projector must agree with the selected layout and effects.
No fixed universal ordering silently makes independently available capabilities
compatible. In particular, linear packed-panel rectangles cannot be used as
bank-scattered transfers without an independently checked mapping.

The new families are explicit APIs. They do not automatically replace every
ordinary source route, qualify a real accelerator, synthesize a buffer pool or
provide a generated target adapter. The existing model discovery still selects
its supported single-operation readout; a Phase 2 consumer can explicitly
select the complete checked chain with `output_operations`. Broader floating
readout, runtime bias/broadcast composition and interleaved sibling grouping
remain outside that new chain/grouping scope.

## Exact readout chains and custody

The chain reader composes the actual typed scalar DAGs while preserving casts,
clamps, constants and operation boundaries. Every intermediate must have one
same-block consumer with the same coordinates. Runtime captures, additional
observers, unknown intervening effects and changed conversion order refuse.
A bounded chain must match the original selected physical readout exactly;
the reader cannot turn a declaration into execution qualification.

A compiled source result reopens every chain member and its use-def links,
checks current source geometry/types against actual compiled arguments, and
requires the same selected contracts. A valid compiled program from another
shape or an unrelated same-shape contraction cannot replace the checked result.
Complete native tests execute emitted host loops and an independent byte-memory
integer primitive, including signed modular overflow, saturating boundaries,
repeated invocations and output and storage guards. This is neutral execution, not
a physical accelerator numerical or performance verdict.

## Configuration and finite command compaction

Loop configuration hoisting moves an invariant pure-write prefix immediately
before a nonempty loop. All remaining effects must be known and preserve those
fields. Unknown state, clobbers, pulses, execution annotations and pointer-valued
setup remain barriers. The independent verifier checks actual moved/retained
calls, their projected effects, loop domains and the complete remaining body.
An independent state interpreter verifies outputs and final register state;
for example, seventeen repeated setups can become one without changing uses.

Command compaction infers repeated affine blocks from a bounded finite stream
and emits ordinary host counted loops. A separate verifier evaluates every
actual operand and compares the entire original primitive sequence in order.
Signed integer intermediate checks reject unsafe affine arithmetic. Pointer,
floating, symbolic and execution-annotated calls remain literal. This reduces
static representation, not dynamic device commands; a native stateful-store
regression checks actual generated C, effect order, complete outputs and guards.
Phase 2 may choose unrolling separately within its unchanged code-size budget.
Neither mechanism emits an FSM/hardware-loop instruction.

## Phase 2 selection and validation

Keep the original successful Phase 1 package byte-exact. For EL4 V26 its digest
is `29213759089b71de26f8a5ab26ca6ef1b8071dc0bb690041eb7de5d0efb53f24`.
Existing manually edited target derivatives are historical diagnostics and are
not seeds for this route. All target compiler changes must be generated and
retained through ordinary Phase 2 from that original package.

Export and review a fresh library release, selected equally for control and
candidate before preparation. No existing frozen library or running experiment
is updated in place. The public API inventory exposes the new functions to the
ordinary authoring mechanism; it grants no target adapter or qualification.
Phase 2 can select legal tile sizes, traversal, retention, panel dimensions,
copy distance/concurrency, setup motion and command compaction. It can also
construct new transformations within original edit authority. Original output,
source/effect, instruction and executable-size checks remain mandatory.

Use independent holdout shapes, tails, capacities, aliases and numerical
boundaries to test the compiler rule. Count reductions, requested bytes and
synthetic overlap establish structural changes. They do not establish cycles.
Spike can check functional execution, emitted commands and host instruction
cost; bank/queue overlap needs suitable timing evidence. No new FPGA execution
is part of developing these passes. FPGA use remains exceptional full validation.


## Why these gaps survived

The earlier pieces proved local arithmetic, resource legality and selected
rewrites. Small test shapes often fit whole operands; larger shapes could use
correct single-tile fallbacks. Neither demanded bounded reuse, wide transfers,
common-input retention or overlap. Existing representations for effects and
layout decisions did not themselves implement those transformations. Host-only
improvements also left device command and movement costs untouched.

The added regressions check complete executions and structural improvements
across independent shapes and resource geometries, with deliberate address,
continuation, state, dependency and source-binding corruption. Source facts,
capability declarations and tested transformations remain separate inputs.
Phase 0 representative performance-scale cases expose the missing resource
regimes to Phase 2. They do not change the successful Phase 1 arithmetic result
or require its authoring loop to optimize them. Phase 2 must still demonstrate
that the generated compiler actually uses the passes and improves paired timing;
a neutral pass test cannot catch a missing target consumer.
