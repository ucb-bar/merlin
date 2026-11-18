---
title: 5. Transform and schedule the program
kind: guide
status: current
owner: compiler
last_verified: 2026-11-09
related: [04-planning, 06-lowering, 11-optimization]
code_refs: [src/merlin/compiler/transforms/pipeline.py, src/merlin/compiler/transforms/pipeline_state.py, src/merlin/compiler/schedule/blocks.py, src/merlin/compiler/optimizations/registry.py, src/merlin/compiler/dialects/schedule.py, src/merlin/compiler/dialects/opt.py, src/merlin/compiler/kernel_ir/ir.py, src/merlin/compiler/lower/native/invocation.py, src/merlin/compiler/lower/options/optional_passes.py, examples/tutorials/03_scheduling.ipynb, examples/tutorials/04_transformations.ipynb]
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
| Generic IR rewrites and ordered pipelines | [`compiler/transforms/`](../../src/merlin/compiler/transforms/) |
| Common frontend normalization and graph transforms | [`compiler/frontend/`](../../src/merlin/compiler/frontend/), [`compiler/transforms/`](../../src/merlin/compiler/transforms/) |
| Registered optimization strategies | [`optimizations/registry.py`](../../src/merlin/compiler/optimizations/registry.py) |
| Tiling, operand retention, transfer merging and command ordering | [`compiler/schedule/`](../../src/merlin/compiler/schedule/) |
| Native MLIR invocation | [`lower/native/invocation.py`](../../src/merlin/compiler/lower/native/invocation.py) |
| xDSL dialect/pass driver | [`dialects/opt.py`](../../src/merlin/compiler/dialects/opt.py) |
| Target instruction selection, encoders and target-specific rewrites | The selected OOT backend |

These are related libraries with different inputs. A scalar IR rewrite does not
consume the same representation as a command-stream scheduling pass.

## Try real transformations

[Notebook 4](../../examples/tutorials/04_transformations.ipynb) starts with a loop
that adds the integers zero through nine. It simplifies a redundant addition,
splits the loop into ordered segments and optionally unrolls it. The notebook
shows the changed IR and calls `validate_change` on each recorded rewrite. It
uses `plan_only=True`; these structural checks do not issue a formal receipt.

The public pipeline uses `PipelineOptions` and `apply_common_pipeline` from
`merlin.compiler.transforms.pipeline`:

```text
apply_common_pipeline(module, options, *, verifier=None, plan_only=False, context=None)
```

`PipelineOptions` requires an explicit `index_bits`. The driver verifies and
clones the input module, then records actual changes. A covered change outside
planning mode requires the host verifier's live accepted Lean completion for
those exact bytes. A callback value or structural certificate alone is
insufficient. `PipelineSession`, defined in `transforms/pipeline_state.py`,
tracks pass results and invalidates analyses when their IR or context changes.

When adding a rewrite, keep its applicability, numerical ordering and effects
beside the implementation. Record the transformation that actually ran rather
than a separately maintained list of intended passes.

## Understand the scheduling representations

| Representation | What it expresses | Source to open |
| --- | --- | --- |
| xDSL `schedule` dialect | Selected compiler decisions and bindings | [`compiler/dialects/schedule.py`](../../src/merlin/compiler/dialects/schedule.py) |
| Python `Kernel`, `Loop`, `Call` records with `mk` text | Counted loops, calls and effects | [`compiler/kernel_ir/ir.py`](../../src/merlin/compiler/kernel_ir/ir.py) |
| Shared counted block schedules | Explicit loads, compute, storage and completion | [`compiler/schedule/blocks.py`](../../src/merlin/compiler/schedule/blocks.py) |

These are not automatically converted into one another or into the upstream
MLIR Transform dialect. Use the adapter supported by the selected compilation
route. The [architecture reference](../reference/architecture.md) explains the
roles of the core dialects without presenting them as a mandatory linear pipeline.

[Notebook 3](../../examples/tutorials/03_scheduling.ipynb) lets you change block
size, storage capacity, operand reuse and command order. It parses real schedule
operations, checks storage ranges and executes the neutral schedule model.
The notebook compares retained loads with reloading in every group and checks
both results against its small integer fixture. The command picture shows
ordering, not timing or measured hardware performance.

## Reuse upstream pass managers

`compiler/lower/native/invocation.py` exposes
`apply_passes(mlir_text, pipeline, timeout=600)` as a diagnostic seam into the
selected Model2MLIR Python environment's native pass registry. Production scalar
lowering uses `lower/native/policy_pipeline.py` and the selected native driver.
`merlin-opt`, implemented in `compiler/dialects/opt.py`, uses xDSL's opt driver
and discovers Merlin-authored passes through `dialects/lowering/passes.py`.
Its catalog distinguishes production, normalization and prototype passes;
catalog membership does not establish that a compilation invoked a pass.

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
every installed backend. See `compiler/lower/options/optional_passes.py` for the actual
catalog and `compiler/optimizations/registry.py` for registered strategies.

An exact pass preserves the admitted program's numerical behavior. A pass that
changes numerical meaning needs a separately approved contract and reference.
Compiler-generated suggestions do not choose their own acceptance thresholds.
