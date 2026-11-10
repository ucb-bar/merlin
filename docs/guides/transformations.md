---
title: Compose and extend the shared transformation library
kind: guide
status: current
owner: compiler
last_verified: 2026-11-09
related: [scheduling, compiler_contributions, shared_pass_compiler_generation, conditional_compiler_pipelines, ../reference/shared_compiler_passes, ../design/compiler_verification]
code_refs:
  - src/merlin/transforms/pipeline.py
  - src/merlin/transforms/pipeline_state.py
  - src/merlin/transforms/pipeline_policy.py
  - src/merlin/transforms/evaluated_exploration.py
  - src/merlin/transforms/certificates.py
  - src/merlin/transforms/validation.py
  - src/merlin/transforms/loops.py
  - src/merlin/transforms/unrolling.py
  - src/merlin/compiler/transforms/AGENT.md
  - src/merlin/compiler/optimizations/registry.py
  - src/merlin/compiler/schedule/api_catalog.py
  - src/merlin/compiler/schedule/derive.py
  - src/merlin/compiler/schedule/tile_load_coalescing.py
  - src/merlin/compiler/schedule/completion_streams.py
  - src/merlin/targetgen/software_spec.py
  - src/merlin/targetgen/shared_pass_contract.py
  - src/merlin/compiler/passes/xdsl.py
  - src/merlin/compiler/passes/native.py
---

# Compose and extend the shared transformation library

**Tutorial 4 of 5** | [Start here](../../examples/tutorials/README.md) | [Previous: Scheduling](scheduling.md) | [Next: CCA](cca.md)

Start with a loop that adds the integers 0 through 9. Remove a redundant `+ 0`,
split the loop into smaller ordered segments, then optionally unroll it. The
[transformation notebook](../../examples/tutorials/04_transformations.ipynb) runs
these real Merlin passes and shows the before/after IR beside each other.

Use the [shared notebook launcher](../../examples/tutorials/README.md#open-the-notebooks)
to open the examples in the right Python environment.

Open `04_transformations.ipynb` and choose **Kernel -> Restart Kernel and Run All
Cells**. No board or target toolchain is needed. Change the segment size and
unroll factor using widgets or an ordinary call: `explore(tile=4, unroll=0)`.
The coverage picture shows which loop indices belong to each segment, not time.

## Compose existing passes

The notebook uses the public pipeline:

```python
from merlin.transforms.pipeline import PipelineOptions, apply_common_pipeline
from merlin.transforms.pipeline_state import PipelineContext

planned = apply_common_pipeline(
    module,
    PipelineOptions(
        index_bits=64,
        normalize_shapes=False,
        simplify_integers=True,
        reuse_buffers=False,
        tile_iterations=(3,),
        unroll_factors=(2,),
    ),
    context=PipelineContext(),
    plan_only=True,
)
```

`module` is a verified xDSL module; the notebook supplies the complete source.
xDSL is the Python compiler framework used here. The pipeline clones that source,
runs the selected passes and records their changes. It leaves the original
untouched so the next attempt starts from the same program.

The ordered segmenter forwards the accumulator through both loops and handles
the short final segment. It preserves the original addition order. Unrolling
expands the loop body and therefore increases static code size. The selected
index width must also fit the new index arithmetic; the notebook demonstrates
a refusal when four bits are insufficient.

`plan_only=True` lets you inspect a proposed rewrite without formal admission.
`validate_change` independently checks the recorded structural changes. To admit
a covered change, the ordinary pipeline requires its host-owned verification
completion. The notebook shows that omitting it is refused. Structural checks,
formal proof and compiled-code execution remain separate steps.

## Find the pass you want to change

| Work | Start here |
| --- | --- |
| Common xDSL transformations | [transforms/pipeline.py](../../src/merlin/transforms/pipeline.py), with algorithms in [loops.py](../../src/merlin/transforms/loops.py), [normalization.py](../../src/merlin/transforms/normalization.py) and [unrolling.py](../../src/merlin/transforms/unrolling.py) |
| Semantic and host-code transformations | [compiler/transforms](../../src/merlin/compiler/transforms/AGENT.md) |
| Block tiling, retention, transfers and command ordering | [compiler/schedule](../../src/merlin/compiler/schedule/AGENT.md) |
| Selectable optimization strategies | [compiler/optimizations/registry.py](../../src/merlin/compiler/optimizations/registry.py) and its named implementations |
| Pass order and analysis validity | [pipeline_state.py](../../src/merlin/transforms/pipeline_state.py) |
| Native MLIR and xDSL invocation | [compiler/passes](../../src/merlin/compiler/passes) |
| Independent change checks | [validation.py](../../src/merlin/transforms/validation.py) and [verification](../../src/merlin/verification) |

These owners have different jobs. An optimization strategy chooses a useful
implementation or combination of passes. A transformation changes IR. A
scheduling algorithm constructs or changes commands. The invocation layer runs
the selected passes; it does not replace upstream pass managers.

## Add or modify a pass

Agents can compose existing passes, edit their implementations or add a pass
within the selected authoring files. Begin with the algorithm and its real
consumer: a new rewrite is useful only if its output reaches the compiler path
you intend to change.

The notebook adds a small operation-count analysis through `PipelineSession`.
`PassSpec` names its stage, required analyses, produced analyses and whether it
changes IR. The session records those results against the current source and
context. A later edit makes the old count stale, which the example checks.
This bookkeeping does not prove a transformation correct.

For a rewrite, describe the supported numerical domain, outputs, effects,
aliases, tails and resource assumptions. Preserve the original source, update the
actual consumer and required source inventories, and extend the independent
checker when the existing obligation does not cover the change. Do not add a
second copy of the algorithm or a new pass framework.

The notebook replaces the segmenter's clipped upper bound, `min`, with `max`.
The altered IR still parses and has fresh hashes, but the independent loop
recognizer rejects its structure. This shows why valid syntax and a matching
record are not enough. See [compiler verification](../design/compiler_verification.md)
for the additional proof and execution work.

## How software meaning and hardware facts affect a pass

| Input | What the pass can learn |
| --- | --- |
| Actual program | Operations, shapes, dtypes, layouts, effects and source identity |
| Selected software spec | The required arithmetic and supported numerical/ISA contracts |
| Reviewed hardware facts | Observed geometry, widths, addressability and other extracted quantities |
| OOT backend | Implemented lowering, exact command effects, physical mapping and encoding |

An **out-of-tree (OOT) backend** lives in a separate package or repository.
These inputs work together: a memory capacity alone cannot authorize a transfer
rewrite, and an opcode declaration does not establish implemented lowering.
Missing facts stay unknown. Neither the backend nor an editable pass chooses
new accuracy thresholds or changes the independent evaluator.

[software_spec.py](../../src/merlin/targetgen/software_spec.py) reads the selected
software semantics. A pass crossing a quantization boundary must preserve the
realized rounding, scaling, overflow and reduction order, or use a separately
approved numerical alternative.
[derive.py](../../src/merlin/compiler/schedule/derive.py) projects selected
address-space or machine facts into the geometry needed by block scheduling.
That projection omits details such as arbitration and instruction costs; a pass
that needs them must use the appropriate richer contract.

This notebook's transforms need no target geometry, so its hardware/resource
identities remain unknown. A string in `PipelineContext` identifies a selected
input; it does not establish a hardware property.

## Reuse the four shared scheduling families

| Family | Shared implementation examples | What the backend must supply |
| --- | --- | --- |
| Tiling | [dense_planning.py](../../src/merlin/transforms/dense_planning.py), [block.py](../../src/merlin/compiler/schedule/block.py), [loop_schedule.py](../../src/merlin/compiler/schedule/loop_schedule.py) | Primitive geometry, supported tails, available resources and numerical order |
| Operand retention | [staged_rhs_residency.py](../../src/merlin/compiler/schedule/staged_rhs_residency.py), [dense_panel_pipeline.py](../../src/merlin/compiler/schedule/dense_panel_pipeline.py) | Load/compute effects, stable storage and continuation/completion behavior |
| Transfer merging | [tile_load_coalescing.py](../../src/merlin/compiler/schedule/tile_load_coalescing.py), [window_transfer.py](../../src/merlin/compiler/schedule/window_transfer.py) | Address mapping, widths, bounds, side effects and completion behavior |
| Command ordering | [completion_streams.py](../../src/merlin/compiler/schedule/completion_streams.py), [configuration_loops.py](../../src/merlin/compiler/schedule/configuration_loops.py) | Full effects, configuration requirements and dependency/event rules |

These are implemented algorithms with specific domains, not universal switches.
For example, an address mapping does not authorize removal of a completion wait.
Fewer commands also do not establish a speedup. Encoding remains backend-owned;
the output can be LLVM IR, objects, device binaries or command artifacts.

Continue with the [scheduling notebook](../../examples/tutorials/03_scheduling.ipynb),
[API catalog](../../src/merlin/compiler/schedule/api_catalog.py), or
[shared-pass compiler guide](shared_pass_compiler_generation.md). For conditional
selection, [pipeline_policy.py](../../src/merlin/transforms/pipeline_policy.py)
uses current program and selected contract facts.
[evaluated_exploration.py](../../src/merlin/transforms/evaluated_exploration.py)
connects candidate exploration to caller-owned evaluation results; it does not
run a simulator or invent measurements.
