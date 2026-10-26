---
title: "Per-operation optimization principles for compiler authors"
kind: design
status: current
owner: merlin-experiments
last_verified: 2026-10-26
related: [joint_optimization_catalog, optimization_suggestions, compiler_authoring_prompts]
code_refs:
  - docs/design/operation_optimization_catalog.json
  - docs/design/operation_optimization_catalog.schema.json
  - packages/merlin-experiments/src/merlin_experiments/phase1/operation_optimizations.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/capability_inventory.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/feedback/performance_brief.py
  - packages/merlin-experiments/tests/test_operation_optimizations.py
---

# Per-operation optimization principles

The architecture catalogue ([joint_optimization_catalog](joint_optimization_catalog.md), selected
per target as `OPTIMIZATION_SUGGESTIONS.json`) says which library mechanisms a target's declared
capabilities support. It does not say what usually matters for a *kind of operation*: a single-row
product is bound by its weight bytes, neighbouring convolution outputs share input rows, a
contraction default changes every shape class at once. Compiler authors kept relearning these from
cycles, and a default tuned on one aspect ratio silently slowed the others.

`operation_optimization_catalog.json` states those lessons once, per operation family, as general
principles and pitfalls. It names no architecture, target, run or measurement, sets no thresholds or
tile sizes, and links each family to the generic mechanisms that implement it: architecture-catalogue
ids, author skills and recipe scripts. `operation_optimization_catalog.schema.json` is its JSON schema.

## Families

Families are derived from each case's own declaration by
`merlin_experiments.phase1.operation_optimizations.families_of`, primary first, never from a case or
model name:

| family | derived from |
| --- | --- |
| `contraction` | a contraction family without a window, more than one output row; sibling products sharing one operand also add `connected_chain` |
| `matrix_vector` | a contraction whose output has one row (`shape_classes.describe`), or a declared `gemv*` operation |
| `convolution` | any declared sliding window (`op_form_regimes.has_window`), including pooling windows |
| `attention` | an attention operation or semantic family |
| `elementwise_epilogue` | an elementwise family; added to a producer that declares an epilogue, scale, fused stage or composed family |
| `reduction_normalization` | reduction, normalization or softmax families without a window |
| `data_movement` | the movement family |
| `connected_chain` | a model, component program, multi-node graph, or an unnamed operation consuming several weights |

A `general` entry holds the principles for every operation: same-kind cycle evidence decides,
UNKNOWN timing is missing evidence, instruction counts are diagnostics, defaults govern every shape
(compare with `agent_selfcheck.py --panel FAMILY`), read `optimization_next` and `capacity_use`.

## Entries

Each entry has a `focus` sentence, `principles` (kind `principle`, `pitfall` or `evidence`),
`mechanisms` (`catalog`, `skill` or `recipe`), `evidence` to compare, an optional `panel_family` for
`--panel`, and `related` families. A principle or mechanism may declare `uses_instruction_roles` from
the closed role vocabulary (`merlin.compile.analysis.instruction_roles.ROLES`).

`validate` checks the schema and every cross reference: catalogue ids exist in the architecture
catalogue, skills and recipe scripts exist among the author skills, roles are known, ids are unique.

## Staging and feedback

At launch, `capability_inventory.stage_selected` hands the same public roster it summarizes
(core, scalability, programs) to `operation_optimizations.stage_selected`, which:

1. counts the cases per family (`present_families`);
2. reads `prohibited_instruction_roles` from the joint policy and the declarations, wherever they
   carry it (`prohibited_roles`), and drops every principle or mechanism that uses such a role,
   recording what was dropped in `omitted_for_policy`;
3. writes `OPTIMIZATION_BY_OPERATION.json` (the general entry and the entries of the present
   families, catalogue mechanisms expanded with their API and the target's capability status from
   `OPTIMIZATION_SUGGESTIONS.json`) to the workspace and run directory, and appends a short
   "Optimization principles by operation" section to TASK.md.

Staging is advice: a failure is printed and the launch continues. On every self-check, the
performance brief adds `operation` to each advised case's `optimization_next`: the family, its entry
id in the staged file, its focus and the panel command, plus any secondary families (`also`).

## Changing the data

Edit the JSON, keep every statement architecture-agnostic and free of run names and numbers, run
`packages/merlin-experiments/tests/test_operation_optimizations.py`, and bump `review.reviewed_on`.
A lesson that only holds for one target belongs in that target's operator guidance, not here.
