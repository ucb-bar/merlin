---
title: Plan compiler construction from public capabilities
kind: guide
status: current
owner: experiments
last_verified: 2026-10-23
related: [conditional_compiler_pipelines, ../reference/compiler_optimization_workflow, artifact_evaluation]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase1/capability_inventory.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/capability_source.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/capability_query.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/session.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/run_inputs.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/generated_isa_library.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/compiler_library.py
---

# Plan compiler construction from public capabilities

## Optional reviewed generated instruction scaffold

An operator may prepare a **future fresh optimized backend atop generated
scaffolding**. This is an explicitly selected input treatment, not an empty-input
comparison and not an already optimized backend. The submission still begins
empty; the generated instruction library is a separate readonly input. Existing
resume bindings remain immutable.

`phase1.generated_isa_library.prepare_library` extends an explicitly reviewed
compiler-library selection with `reviewed_isa.instruction_dialect`, its inert
package initializer, resolved ISA, generator manifest and source-selection record.
The helper copies only enumerated library members and those exact generated
products. Unrelated package siblings, companion repositories, kernels and
reference backends are not selected.

After reviewing public declaration/facts identities and the generated manifest:

```python
from merlin_experiments.phase1.generated_isa_library import prepare_library

selection = prepare_library(
    reviewed_base_selection, generated_package, fresh_library_directory,
    review_id=reviewed_new_release_id,
    manifest_sha256=reviewed_manifest_bytes_sha256,
    facts_sha256=selected_public_facts_sha256,
    declaration_sha256=selected_public_declaration_sha256,
)
# Save selection.contract.record() as JSON outside fresh_library_directory.
# Explicitly choose this future treatment with the ordinary three flags:
# --compiler-library-root <fresh_library_directory>
# --compiler-library-contract <saved_contract.json>
# --compiler-library-contract-sha256 <selection.contract.sha256>
```

Keep the existing editable-generic release prefix when that separately reviewed
base capability should remain enabled. The helper does not infer edit authority.
It verifies manifest and public input identities, exact output membership,
regenerated dialect bytes, and agreement between the manifest's generator
implementation hashes and the selected library. If those generator leaves changed,
review a matching library release; do not suppress the mismatch. No new sandbox
mount or run-option mechanism is needed: ordinary reviewed-library freezing,
source sealing, import admission and readonly transport apply before authoring.

The generated declarations supply no tensor lowering, fusion, allocation or
schedule. The agent builds those in its backend and permitted compiler fork.
Independent encoding validation remains pending when no assembler/RTL evidence
is supplied. Merely staging the scaffold supplies no numerical, runtime or
performance qualification, and changes no capsule obligation. Initial guidance
explicitly labels the selected scaffolding and its limits.

## Whole-corpus planning

Fresh joint campaigns stage `PUBLIC_CAPABILITIES.md` and
`PUBLIC_CAPABILITIES.json` before the ordinary treatment seal. The initial task
points to the short summary; detailed records remain on demand. Resume preserves
those exact bytes rather than rebuilding guidance from newer source.

The inventory uses the same admitted public capsule selection and frozen
scalability selection as native grading. It preserves the original correctness
roster, adds every selected scalability case and rejects duplicate IDs or hidden
labels. Original required gSIM checks remain distinct from calibration-witness
permissions. An optional operator probe outside the public roster contributes
only an unresolved-whitelist count, never a new public member or dispatch grant.

## Read requirements without flooding context

Start with the Markdown summary, then use the staged read-only helper:

```sh
python capability_query.py --groups
python capability_query.py --group PREFIX
python capability_query.py --capsule PUBLIC_ID
python capability_query.py --group PREFIX --offset 40 --limit 40
```

The helper uses only Python's standard library and its adjacent public inventory.
It has no arbitrary input-path option and no compiler, grader or oracle imports.
Requests are paged and output is bounded to 32 KiB. Capsule details report graph
counts; full graph structure stays in the JSON for an explicit targeted query.
The served and archived helper bytes are sealed with the inventory. Nothing in
the helper can schedule execution or grant a grade.

Records include public input shapes/dtypes/layout requirements, exact declared
numerical policy, source operation counts, graph edges and each case's engine
obligations. The existing interface and typed-MLIR parsers inspect only the
declared source program. No tensor arrays, golden outputs or reference compiler
implementations are opened. Unsupported or missing source analysis is explicit
`unknown`; it does not make the source an empty or supported graph.

`capability_groups` separates source operations, input dtypes, numerical policy
and declared layout requirements while sharing across shape sizes. Each group
suggests a small end-to-end canary and a large generalization case, ordered by
declared input volume. These are advisory starting points, not execution-time
estimates, sufficient test coverage, a dispatch queue or automatic fusion proofs.
Graph indices refer to the accompanying ordered operation list. Public
contraction and movement declarations are source contracts, not selected target
instructions or prescribed device schedules.

## Construct one compiler and evaluate continuously

Give the coordinator the entire compact capability inventory. Assign independent
helpers compiler boundaries such as graph transformations, target scheduling and
host numerical lowering. Each helper receives its relevant requirements and API
references rather than the whole catalog or conversation.

Test a small end-to-end case when a capability first becomes executable. Continue
building other capabilities while immutable snapshots run through the ordinary
rolling queue. Shared ABI or emission failures should be repaired once before
resubmitting affected cases. The inventory does not replace readiness execution,
change the queue, issue a pass or bypass any native prerequisite.

Original correctness and calibration prerequisites can advance concurrently with
large-shape checks. Prioritize scalability performance once numerical execution
works; small original cases remain mandatory correctness regressions and useful
model evidence. Query detailed per-case feedback only when needed. When selected,
use the shared conditional pipeline APIs to enumerate legal alternatives and
consume their transformed artifacts in actual emission. Keep checks and timing
measurements attached to the exact compiler and model versions.

The inventory carries no qualification authority. Final hidden and complete-model
obligations remain outside this public denominator and retain their existing
independent owners.
