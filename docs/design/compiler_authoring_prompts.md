---
title: Shared compiler authoring plan and launch-specific tool guidance
kind: design
status: current
owner: merlin-experiments
last_verified: 2026-10-09
related: [component_scale_generalization, fresh_compiler_origin, component_phase2_workflow]
code_refs:
  - src/merlin/targetgen/generalization_prompt.py
  - src/merlin/targetgen/generate_prompt.py
  - src/merlin/targetgen/tool_registry.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/task_staging.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/source_inputs.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/component_origin.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/stage_prompt.py
  - packages/merlin-experiments/tests/test_general_compiler_prompts.py
---

# Shared authoring plan

The task states the destination explicitly: Phase 1 builds a functional compiler
for end-to-end compilation of supported models. Small public cases provide
affordable semantic checks. Later programs can have larger dimensions, longer
reductions and more interacting operations. An observed small-case pass cannot
establish an untested large-input or whole-model property.

`GENERAL_COMPILER_CONTRACT_V1` is a versioned, target-independent requirements
and planning block. Every experiment level receives identical bytes. It names
no target, validation model, reference cycle count, winning schedule or protected
example. It grants no paths, numerical changes, tools or evaluation authority.
Future wording changes require a new version; archived tasks and receipts stay
unchanged.

Its plan is:

1. Read the declared domain, numerical rules, budgets, writable owners and tool inventory.
2. Inventory supported semantics and lowering coverage; identify unsupported or unknown cases.
3. Build the ordinary compiler pipeline and inspect the actual effect of each pass.
4. Test invariants using bounded independent executions and admitted scale checks, with distinct evidence.
5. Use feedback to repair the general rule; choose Phase 2 analysis or measurement by the unresolved question.
6. Record limitations and regressions, complete promotion gates, and freeze before protected evaluation.

This is a common workflow, not a target implementation recipe. The target's
facts and current input IR determine legal lowering. The author chooses its
implementation through the tools granted to its treatment.

## Phase 1 composition and experiment levels

The shared `generate_prompt.render_prompt` inserts the block unconditionally.
Actual task staging also appends it to explicitly selected authored tasks if
they do not already contain it. Both generated and authored tasks preserve the
same requirements. The staged `TASK.md` is copied byte-for-byte into the run
record; provider drivers do not add model-specific strategy instructions.

`task_tool_inventory_block` reads the same selected tool names used for launch:
the bundle's `tools.txt`, with the existing bundle-stem fallback for old bundles,
followed by explicit add/drop choices. It displays tool catalog descriptions and
broker client names. It separately lists the common feedback substrate. The
level's label does not create a capability, and the inventory does not add a
grant. Runtime readiness and mount admission remain separate checks.

Existing level-specific guidance still describes the tools selected by each
treatment. C++ generation, structured compiler tools, inspection and hardware
fact tooling appear through those grants. Their availability changes the means
of implementing the common task, not the task's generality or numerical goal.

The fresh authoring route serves the same block, records the exact prompt as an
invocation input, and pins its source dependency. The ordinary source inventory
also pins the common block and renderer. A changed implementation is a changed
run dependency, not permission to reinterpret a frozen run.

## Phase 2 composition

Phase 2 uses its explicit launch tool/action inventory rather than experiment
levels. The component prompt displays each broker action's purpose, availability,
unavailable reason and required/optional status. Exact argv bindings remain in
the unchanged structured launch document. The ordinary performance prompt keeps
its selected tool table and adds the same compiler requirements and plan.

An unavailable measurement stays unavailable. The plan never turns compilation
into numerical evidence, functional simulation into hardware timing, or
out-of-domain predictions into measured results. It also does not expose final
workload graphs merely by declaring end-to-end compilation as the destination.

## Prompt guidance and enforced evidence

The prompt directs the author toward shape-parametric behavior, capacity
handling, graph composition, complete outputs and numerical contracts. It also
states that static instruction count need not increase with input dimensions:
a loop can execute more work without adding instructions. Emit-only counters are
diagnostics, not proof of shape coverage.

Prompt guidance cannot replace independent qualification. The separate
[generalization protocol](component_scale_generalization.md) still needs budgeted
generation, ordinary compile-through-link grading, discharged scale legality,
synthetic interaction obligations and regime-qualified performance transfer.
Missing mechanisms remain missing even when the author follows the plan.
