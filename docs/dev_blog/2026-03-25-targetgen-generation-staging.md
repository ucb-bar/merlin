# 2026-03-25 TargetGen Generation + Mutation Staging

> **Repro pin:** merlin@[`2903e28b`](https://github.com/ucb-bar/merlin/commit/2903e28b0ec45c4e109fa9c98592b7f0353fcf40) · iree_bar@[`ddf4685ae1`](https://github.com/ucb-bar/iree_bar/commit/ddf4685ae1)
> **Status:** Active

## 1. Context and goal

TargetGen already knew how to:

- normalize capability specs and deployment overlays
- classify targets into Merlin integration families
- emit support plans, task graphs, and verification ladders
- build an execution bundle and prompt packets
- stop the executor at explicit operator and branch gates

What it did not yet do was generate a reviewable staged tree for likely repo
changes. The gap was obvious:

- planning existed
- prompts existed
- gated execution existed
- but there was no deterministic non-live output showing what Merlin, IREE, or
  LLVM surfaces a target would likely add or edit

The goal of this pass was to add that missing middle layer without enabling
live mutation or validation yet.

At the same time, the repository presentation needed a user-first cleanup. The
repo is powerful, but a first-time user currently sees too many layers at once:
compiler internals, runtime bring-up, hardware recipes, benchmarks, research
projects, forks, and dev logs all compete for attention.

## 2. Implementation changes

### TargetGen generation layer

Added new data models in `tools/targetgen/model.py`:

- `GeneratedScaffoldFile`
- `GenerationBundle`
- `MutationCandidate`
- `MutationBundle`

Added a new module `tools/targetgen/generator.py` with two non-live writers:

- `emit_generation_artifacts(...)`
- `emit_mutation_artifacts(...)`

These emit only under `build/generated/targetgen/<target>/` and never touch
repo-tracked files.

### New CLI commands

Extended `tools/targetgen_cmd.py` with:

- `merlin targetgen generate`
- `merlin targetgen stage-mutation`

`generate` now writes:

- planner artifacts
- input snapshots under `inputs/`
- `generation_bundle.json`
- `generation_summary.md`
- `generated/tree/...` with staged scaffold files at prospective repo paths

`stage-mutation` now writes:

- `mutation/mutation_bundle.json`
- `mutation/proposal_brief.md`
- `mutation/worktree_plan.md`
- `mutation/proposed_tree/...`

This is the first point where the system groups only the mutating surfaces and
stages them for later promotion.

### Repo user-experience cleanup

Added `docs/user_paths.md` to route users by workflow instead of by raw folder
name.

Updated:

- `README.md`
- `docs/index.md`
- `docs/getting_started.md`
- `docs/repository_guide.md`
- `docs/architecture/target_generator.md`

The new bias is:

- user layer first
- bring-up layer second
- compiler/runtime/fork layer last

instead of presenting the entire tracked tree as the starting point.

## 3. What worked

- The TargetGen model was already structured enough that generation and
  mutation staging could be added without a planner rewrite.
- Task write scopes were specific enough to anchor scaffold placement.
- The existing execution model already separated shared planning tasks from
  mutating target-family tasks, which made mutation staging straightforward.
- The repo UX problem is documentable without physically moving directories
  around yet. A first-user map is a good intermediate step before any deeper
  repo reorganization.

## 4. What did not work (and why)

- There is still no live mutation path. That remains intentional.
- The generated scaffolds are structurally useful, but they are still staged
  drafts, not source-ready code.
- Some repository conventions are still transitional:
  - `models/*.yaml` remains a compile-target view
  - `target_specs/` is the newer canonical target surface
  - `build_tools/` still mixes multiple concerns

That means the docs can make the repo easier to understand, but the physical
layout still reflects historical growth.

## 5. Debugging notes

The local sandbox in this environment blocked even read-only shell inspection
with:

- `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`

So file inspection for this pass required escalated read-only commands.

No runtime or validation debugging was done in this pass because the goal was
specifically to avoid test or mutation fallout while building the next layer.

## 6. Test coverage and exact commands

Deliberately not run in this pass.

Reason:

- the goal was to build the non-live generation and staging machine first
- the request explicitly asked not to start dealing with random modifications

So this pass should be understood as implementation-only.

## 7. Follow-up tasks

- Add a real source-promotion path from `mutation/proposed_tree/...` into
  isolated worktrees.
- Add validation runners after promotion, not during generation.
- Make the repo migration story explicit:
  - which `models/*.yaml` files become derived
  - which `build_tools/hardware/*.yaml` recipes become derived
  - which repo layers stay handwritten
- Continue the user-first cleanup by separating:
  - end-user docs
  - bring-up docs
  - contributor internals
  - historical engineering logs
