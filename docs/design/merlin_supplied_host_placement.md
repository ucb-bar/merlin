---
title: "Design note: Merlin supplies the host part of a whole-program capsule"
kind: design
status: current
owner: merlin-experiments
last_verified: 2026-10-17
related: [capsule_phase_split, agentic_experiment_integrity, derived_capsule_axes]
code_refs: [packages/merlin-experiments/src/merlin/targetgen/frozen_placement.py, packages/merlin-experiments/src/merlin/targetgen/owned_host_model_selection.py, packages/merlin-experiments/src/merlin/targetgen/owned_host_model_execution.py, packages/merlin-experiments/src/merlin_experiments/phase0/program_admission.py, packages/merlin-experiments/src/merlin_experiments/corpus/publication.py, src/merlin/compile/owned_host_model.py]
---

# Merlin supplies the host part of a whole-program capsule

Phase 1 grades the author's accelerator compiler. Any operation Phase 0 places on the host is
Merlin's responsibility. Merlin's own generic host codegen compiles it, and it is never graded as
the author's work. Two consequences follow.

1. **No public capsule is host-only.** Synthesis withholds entries whose declared placement has no
   accelerator work. Generation removes a written member whose admitted screen places nothing on the
   accelerator. Release preparation and the native admission run at seal both refuse any public or
   dev capsule without accelerator placement, failing closed when the placement record is missing.
2. **A mixed whole-program capsule is split.** The author compiles the accelerator groups and Merlin
   compiles everything else. The grade still checks the whole program end to end.

## How a model capsule is graded today

`capsule_runner.run_capsule` sends a model capsule to `_grade_candidate_model_capsule_inline`. When
`owned_model_required` holds, that is the owned-host route:

- `owned_host_model_execution.execute_owned_host_model` binds the descriptor's host lane package
  through `OwnedHostModelCompiler`.
- `plan_before_build(..., granularity=BY_GROUP)` forms Merlin's compute groups. The candidate
  receives only the closed per-group interface programs and returns one device kernel per group.
- `compile_saved_model` builds the host program from the original source using Merlin's host
  package, links the candidate's device objects, and runs the required tiers.
- `_numeric_full_output` compares every output to `golden.npz`.

After the build, `_audit_owned_host_device_build` re-forms the groups from the exact routed source.
It requires every independently eligible operation to sit in a device group, refuses a skipped
sidecar, proves each call reaches its kernel in the linked ELF, and runs
`audit_built_device_host_compute`. That audit refuses host arithmetic or host payload inside any
function that implements an accelerator group, and refuses undefined external symbols.

The other route is the explicit public-program treatment, which Phase 2 measurement selects. There
the candidate emits the whole program, host code included.

## What changes

The owned-host route already has the split this design needs. What it lacks is a binding to Phase 0:
it re-derives eligibility from the frozen source, but it never reads the frozen placement.

**Authority.** The split comes from the `software_screen` that Phase 0 froze into the sealed
`capsule.yaml`:

- `program_sha256` binds it to the capsule's program bytes, which must equal the routed source.
- Its admitted per-operation decisions give, for each lane, the operations Phase 0 placed there.
- The grader reads it from the pinned capsule copy, never from candidate output.

`frozen_placement.read` refuses a screen that is missing, not admitted, unbound to the program,
malformed, or has no accelerator operation.

**Route.** `owned_model_required` returns true for every model capsule whose frozen placement
contains accelerator work. Such a capsule can no longer fall back to the legacy runner graph on an
eligibility exception. A mismatch now fails the owned route instead.

**Split check.** `frozen_placement.require_split` runs twice:

- before the candidate is invoked, against Merlin's groups of the routed source;
- again inside `_source_placement` after the build, against fresh eligibility.

Both runs compare, operation by operation name, the accelerator operations Phase 0 froze, the
members of Merlin's device groups, and the independently eligible operations. Any difference
refuses the grade:

- an accelerator operation Merlin would place on the host;
- a host operation fused into a device group;
- a fresh eligibility that disagrees with Phase 0.

## Integrity

- **Accelerator work cannot move to Merlin's host.** The candidate never chooses the split. If it
  declines a group, the existing sidecar check refuses the grade. If an eligible operation lands in
  a host group, `_source_placement` refuses it. A device group count below the frozen accelerator
  count now refuses as well.
- **Host work cannot move to the author.** The candidate receives only the device-group interface
  programs, and the device groups must hold exactly the frozen accelerator operations. Host
  operations reach only Merlin's host package, whose tree digest is checked before and after the
  build.
- **The host-compute audit is unchanged.** It still runs on every linked device function, and
  unknown remains not clean. The frozen-placement check comes in addition to it, never in its place.
- **Nothing is weakened.** NOT_RUN_IS_NOT_PASS, tier requirements, sandbox masking, and
  golden/hidden/sealed checks are untouched. The end-to-end comparison still covers every output.

## Out of scope

- The Phase 2 public-program treatment still measures a candidate-emitted whole program. Applying
  the same split there is a follow-up for the Phase 2 owner.
- The frozen decisions carry no operation ordinals, so the binding is by operation name and count
  rather than by ordinal. A future Phase 0 schema that keeps the ordinals can tighten it to exact
  operation identity without changing the route.
