---
title: 5. Transform and schedule the program
kind: guide
status: current
owner: compiler
last_verified: 2026-11-09
related: [04-planning, 06-lowering, 11-optimization]
code_refs: [src/merlin/transforms/pipeline.py, src/merlin/transforms/pipeline_state.py, src/merlin/transforms/validation.py, src/merlin/compiler/passes, src/merlin/compiler/schedule, src/merlin/compiler/optimizations, src/merlin/xdsl_dialects/schedule.py, src/merlin/sched/ir]
---

# 5. Transform and schedule the program

[Previous](04-planning.md) | [Guide index](../README.md) | [Next](06-lowering.md)


Merlin provides reusable transformations that a backend or compiler-authoring
agent can compose, modify or extend. Backend resource descriptions and command
effects determine where a transformation is legal; the shared library provides
the algorithm.

## Find the right library

| Work | Implementation owner |
| --- | --- |
| Generic IR rewrites and ordered pipelines | `src/merlin/transforms/` |
| Common frontend normalization and graph transforms | `src/merlin/compiler/frontend/`, `compiler/transforms/` |
| Registered optimization strategies | `src/merlin/compiler/optimizations/` |
| Tiling, operand retention, transfer merging and command ordering | `src/merlin/compiler/schedule/` |
| Native MLIR and xDSL pass invocation | `src/merlin/compiler/passes/` |
| Target instruction selection, encoders and target-specific rewrites | The selected OOT backend |

These are related libraries with different inputs. A scalar IR rewrite does not
consume the same representation as a command-stream scheduling pass.

## Try real transformations

[Notebook 4](../../examples/tutorials/04_transformations.ipynb) starts with a loop
that adds the integers zero through nine. It simplifies a redundant addition,
splits the loop into ordered segments and optionally unrolls it. The notebook
shows the changed IR and calls the independent change checker.

The public pipeline uses `PipelineOptions` and `apply_common_pipeline` from
`merlin.transforms.pipeline`. `PipelineSession` tracks pass results and analysis
invalidation. An analysis computed for the previous IR cannot silently stand in
for a fresh result after a modifying pass.

When adding a rewrite, keep its applicability, numerical ordering and effects
beside the implementation. Record the transformation that actually ran rather
than a separately maintained list of intended passes.

## Understand the scheduling representations

| Representation | What it expresses | Source to open |
| --- | --- | --- |
| xDSL `schedule` dialect | Selected compiler decisions and bindings | [`xdsl_dialects/schedule.py`](../../src/merlin/xdsl_dialects/schedule.py) |
| Python `Kernel`, `Loop`, `Call` records with `mk` text | Counted loops, calls and effects | [`sched/ir/kernel.py`](../../src/merlin/sched/ir/kernel.py) |
| Shared counted block schedules | Explicit loads, compute, storage and completion | [`compiler/schedule/blocks.py`](../../src/merlin/compiler/schedule/blocks.py) |

These are not automatically converted into one another or into the upstream
MLIR Transform dialect. Use the adapter supported by the selected compilation
route. The [architecture reference](../reference/architecture.md) explains the
roles of the core dialects without presenting them as a mandatory linear pipeline.

[Notebook 3](../../examples/tutorials/03_scheduling.ipynb) lets you change block
size, storage capacity, operand reuse and command order. It parses real schedule
operations, checks storage ranges and executes the neutral schedule model.
The default example needs 24 loads with retention and 48 without it; both produce
the same small integer result. The command picture shows ordering, not timing.

## Reuse upstream pass managers

`compiler/passes/native.py` invokes the selected native MLIR environment.
`compiler/passes/xdsl.py` runs ordered xDSL rewrites. Use these boundaries and the
existing upstream pass managers instead of adding a shell wrapper for each pass.

A pipeline needs its owner, operation anchor, options, required dialects and
input/output requirements. Record the actual tool version and resulting
artifacts. Repeated passes can be intentional; do not deduplicate by pass name.
Structural verification and partial conversion do not establish semantic
equivalence or complete target lowering.

## Existing saved-model switches

The saved-model lowering route retains its optional pass catalog:

```sh
merlin-compile --list-passes
merlin-compile --list-passes --json
```

That route accepts repeated `--pass NAME` and `--no-pass NAME` options; existing
launch definitions can select `lowering_passes`. These switches retain their
recorded defaults and numerical requirements. They are not universal flags for
every installed backend. See `llvmlower/optional_passes.py` for the actual
catalog and `compiler/optimizations/registry.py` for registered strategies.

An exact pass preserves the admitted program's numerical behavior. A pass that
changes numerical meaning needs a separately approved contract and reference.
Compiler-generated suggestions do not choose their own acceptance thresholds.
