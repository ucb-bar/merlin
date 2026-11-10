---
title: Improve the compiler with passes and CCA
kind: guide
status: current
owner: core
last_verified: 2026-11-09
related: [10-compiler-generation, 12-experiments]
code_refs:
  - src/merlin/transforms/pipeline.py
  - src/merlin/transforms/pipeline_state.py
  - src/merlin/transforms/pipeline_policy.py
  - src/merlin/compiler/schedule/block.py
  - src/merlin/compiler/schedule/derive.py
  - src/merlin/compiler/optimizations/registry.py
  - src/merlin/kernels/cca.py
  - src/merlin/kernels/cca_compare.py
  - src/merlin/kernels/action_catalog.py
  - src/merlin/mining/fork_from_action.py
  - src/merlin/kernels/search_step.py
  - src/merlin/experiments/phase1/optimization_suggestions.py
  - src/merlin/experiments/phase1/operation_optimizations.py
  - src/merlin/experiments/phase1/authoring.py
  - src/merlin/resources/optimizations
  - examples/tutorials/05_cca.ipynb
---

# 11. Improve the compiler with passes and CCA

[Previous](10-compiler-generation.md) | [Guide index](../README.md) | [Next](12-experiments.md)

After [generating a compiler](10-compiler-generation.md), the agent can compose
existing passes, modify their implementations, or add a pass. The shared library
supplies reusable algorithms. The backend supplies the target-specific pieces
needed to use them.

Start with the [CCA notebook](../../examples/tutorials/05_cca.ipynb). It compares
two short instruction listings and puts the suggested compiler edit beside each
difference. The inputs are authored examples; running it needs no hardware.

## Find the implementation

Three things have different jobs:

| What | Where | What to open |
| --- | --- | --- |
| Callable transformations | `src/merlin/transforms/` | `pipeline.py`, then the pass owner |
| Command scheduling | `src/merlin/compiler/schedule/` | `block.py`, `residency.py`, `derive.py` |
| Selectable optimization strategies | `src/merlin/compiler/optimizations/` | `registry.py`, then the selected strategy |
| Authoring advice | `src/merlin/resources/optimizations/` | The four JSON catalogs below |
| Agent instructions | `src/merlin/experiments/phase1/author_skills/` | `pass-library`, `scheduling-api`, and the relevant skill |

A catalog entry describes a mechanism and points to code; it is not the pass
implementation. A prompt tells the agent how to work; it does not make a missing
implementation available.

Shared scheduling includes tiling, operand retention, transfer grouping and
command ordering. These algorithms use declared geometry, command effects and
encoders. Hardware capacities and instruction behavior come from the selected
backend and reviewed target inputs. Keep target encodings and device-specific
lowering in the OOT repository.

## Compose, modify, or add

`transforms.pipeline.apply_common_pipeline` accepts an xDSL module and
`PipelineOptions`. It clones the module, applies selected transformations, and
records changes and their witnesses. Options cover shape normalization, integer
simplification, loop segmentation, storage reuse and explicit pass order.

`PipelineSession` tracks pass requirements and analysis freshness. For a policy
with alternatives, `pipeline_policy.py` defines `RegisteredPass`,
`PipelineVariant`, `PipelinePolicy`, and `evaluate_variants`. These are explicit
registered choices, not a search through every possible pass sequence.

Choose the smallest change that addresses the observed problem:

- **Compose:** use existing passes in an admitted order, with explicit options.
- **Modify:** change the defining implementation in the reviewed editable library
  or candidate compiler workspace. Rebuild all affected artifacts.
- **Add:** implement a missing transformation, declare its prerequisites and
  effects, and connect it to a real pipeline caller and verification path.

Candidate workspaces receive the selected library source and edit permissions.
An agent cannot change the independent evaluator or loosen a numerical threshold
to accept its own optimization. A source pointer in a catalog grants neither
access to that file nor permission to edit it.

## How software contracts and RTL facts help

The software specification describes the operations and numerical forms the
compiler must implement. Reviewed hardware facts constrain capacities, features
and command behavior. For scheduling, `schedule.derive` projects selected facts
into geometry; unknown capacity or unsupported lifetimes must remain refusals.

For authoring advice, `optimization_suggestions.prepare` reads the workspace's
public software declarations, selected contract, available RTL facts and staged
library inventory. It selects applicable catalog entries and writes
`OPTIMIZATION_SUGGESTIONS.json`, retaining the input hashes. Missing inputs can
leave advice unavailable. They must not turn into guessed hardware support.

`operation_optimizations` separately derives operation families from declarations:
shapes, operands, epilogues and program structure. It stages
`OPTIMIZATION_BY_OPERATION.json` and links relevant entries into case feedback.
This can explain why operand retention matters for one operation family without
hardcoding a model name into production lowering.

The current catalog inputs are:

| File under `resources/optimizations/` | Purpose |
| --- | --- |
| `authoring.json` | Reusable mechanisms, requirements and implementation pointers |
| `operations.json` | Principles for each operation family |
| `operations.schema.json` | Structure and cross-reference requirements |
| `extensions.json` | Additional capability-dependent suggestions |

The readers use the explicitly selected source repository. The operation advice
stager still skips when no repository is supplied or its catalog is absent.
This cleanup changes the catalogs' location, not those defaults. Catalog advice
does not choose a schedule, rank a candidate, or establish correctness.

## Compare what the compiler actually emitted

**CCA means Common Compute Abstraction.** It records comparable code properties:
contraction form, accumulator type, vector grouping, operand reuse, dispatches
and more. It is a description of a selected artifact, with scope and provenance.

The current flow is:

```text
reference and candidate artifacts
    -> decode and lift into CCA
    -> compare properties and list missing observations
    -> propose a compiler edit
    -> rebuild and lift the new candidate
    -> verify outputs and measure execution
```

The specification and RTL facts do not directly become a universal CCA record.
Lifters consume supported disassembly, source or schedule representations with
selected decoding information. Their available observations determine which
fields they can populate. `None` means unknown, rather than zero or false.

`cca_compare.compare` reports populated differences; `uncomparable_axes` reports
fields missing on one side. `compare_evidence` requires per-axis witnesses and
compatible scopes. The notebook's authored listings have no such witnesses, so
its strict comparison reports unknown with zero comparable axes.

## Follow the edit and feedback

`action_catalog.build_catalog` returns proposed actions and unrouted differences.
`seam_location` points to the defining compiler edit site. In the mining beam,
`fork_from_action.action_to_fork` turns supported actions into actual overrides.
A missing pass or unsupported override remains a work item requiring code.

After rebuilding, `achieved_residual` compares a proposed action's promised
properties with the new emitted CCA. `search_step.make_step` distinguishes an
unmet promise from an action with no checkable promise. Correctness and measured
speedup enter separately through the evaluator's result.

The Phase 1 assisted authoring loop exposes public CCA/action queries and stages
library advice. `authoring.execute` runs admitted author rounds, certification
and completion. The mining beam and explicit Phase 2 component feedback are
separate existing consumers; CCA is not automatically run by every compile call.

A property match is useful feedback, not independent execution evidence.
Verification must still check the realized numerical contract, actual selected
implementation, ABI and synchronization obligations. The common transformation
pipeline requires trusted checker completion for covered changed IR; without its
verifier, use its explicit `plan_only` mode for inspection.

Continue with [running and inspecting experiments](12-experiments.md) to see
where builds, feedback, source receipts and execution results are retained.
