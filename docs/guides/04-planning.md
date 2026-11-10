---
title: 4. Choose regions, fusion and placement
kind: guide
status: current
owner: compiler
last_verified: 2026-11-09
related: [03-preparation, 05-transformations]
code_refs: [src/merlin/compiler/system.py, src/merlin/compiler/components.py, src/merlin/compiler/planning, src/merlin/compiler/partition/regions.py, src/merlin/compiler/ir/boundary.py, src/merlin/compiler/backend/sourceproposal.py]
---

# 4. Choose regions, fusion and placement

[Previous](03-preparation.md) | [Guide index](../README.md) | [Next](05-transformations.md)


Once the imported arithmetic is known, Merlin can compare implementations for
parts of the program. The backend offers implementations. The planner chooses a
compatible combination for the complete program.

## What a backend proposal means

A proposal is a structured implementation offer for actual source operations.
It identifies the operations covered, live inputs and outputs, numerical and
layout requirements, execution resource, memory needs, guards, ABI and lowering
recipe. Cost estimates include their assumptions; missing costs remain unknown.

An automatic proposal hook means the selected OOT code discovers candidates
from the current IR. It does not mean Merlin invents a target kernel or accepts
every backend claim. `Component.discover()` invokes the selected hook and checks
the resulting proposals against the original source and selected target.

## Follow selection into partitioning

```text
imported program + selected topology + backend components
    -> discover bounded implementation alternatives
    -> check numerical, boundary, memory and resource requirements
    -> choose a compatible region cover
    -> materialize region boundaries
    -> lower selected regions and shared control
```

`SystemCompiler` performs discovery and selection. `PlannedCompiler` consumes an
already selected plan. Both are usable as explicitly selected backend objects
through `merlin.api.compile`; they are not interchangeable with a target name.

The current global selector is bounded. It considers region coverage and shared
resource budgets; it is not a universal optimal solver. Local tiling, instruction
selection and allocation remain inside the selected implementation. A locally
legal schedule must still fit the resource budget promised to the global plan.

## Fusion has more than one owner

| Kind | Example | Owner |
| --- | --- | --- |
| Semantic simplification | Remove a redundant representation change | Shared transforms |
| Region fusion | Group matmul with an eligible epilogue | Discovery, legality checks and planner |
| Target instruction fusion | Emit an accelerator's fused operation | Selected OOT backend |
| Execution coordination | Reuse storage across two engines | Dispatch and SoC integration |

For matmul, bias and activation, useful alternatives include a complete fused
accelerator region, accelerator matmul followed by a vector epilogue, a vector
implementation of the whole chain, and scalar computation.

An accelerator-to-vector route requires the accelerator to expose the needed
intermediate values, compatible memory access, and explicit completion before
the vector code reads them. Two coordinated dispatches are not a single fused
kernel. A per-channel epilogue cannot be replaced by per-tensor scaling merely
because a device readout supports the latter.

The existing OOT fusion-discovery path handles a bounded dense/readout family.
Other patterns need their own implemented proposal and legality checks; the
protocol does not establish universal model fusion coverage.

## Partitioning preserves the program

Boundary analysis finds every live value and effect before outlining. Selected
regions must cover the claimed source operations without dropping outputs or
changing control flow. CPU operations can remain inline. Partitioning materializes
the selected plan; it does not select a new numerical variant.

Runtime guarded variants are a separate mechanism. Guards must be checked before
externally visible effects. A partially executed accelerator region cannot be
retried on scalar code as though nothing happened.

## Open the implementation

| Responsibility | Source |
| --- | --- |
| Whole-program discovery | [`system.py`](../../src/merlin/compiler/system.py), [`components.py`](../../src/merlin/compiler/components.py) |
| OOT fusion proposals | [`planning/fusion.py`](../../src/merlin/compiler/planning/fusion.py) |
| Proposal checks | [`backend/sourceproposal.py`](../../src/merlin/compiler/backend/sourceproposal.py) |
| Global region/resource choice | [`planning/execution.py`](../../src/merlin/compiler/planning/execution.py) |
| Live values and effects | [`ir/boundary.py`](../../src/merlin/compiler/ir/boundary.py) |
| Region materialization | [`partition/regions.py`](../../src/merlin/compiler/partition/regions.py) |
