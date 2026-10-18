---
title: "Design note: Merlin supplies the host part of a whole-program capsule"
kind: design
status: current
owner: merlin-experiments
last_verified: 2026-10-18
related: [capsule_phase_split, agentic_experiment_integrity, derived_capsule_axes]
code_refs: [packages/merlin-experiments/src/merlin/targetgen/frozen_placement.py, packages/merlin-experiments/src/merlin/targetgen/owned_host_model_selection.py, packages/merlin-experiments/src/merlin/targetgen/owned_host_model_execution.py, packages/merlin-experiments/src/merlin_experiments/phase0/program_admission.py, packages/merlin-experiments/src/merlin_experiments/corpus/publication.py, src/merlin/compile/owned_host_model.py]
---

# Merlin supplies the host part of a whole-program capsule

Phase 1 grades the author's accelerator compiler. Merlin supplies its generic host codegen for
operations outside the accelerator groups. Phase 0 fixes the mandatory accelerator work, while
Merlin's capability-checked group formation may also attach supporting readout and layout operations
or select standalone optional device groups. The agent can implement those groups without recreating a host
compiler. Two consequences follow.

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
It requires every independently eligible operation to sit in a device group, accepts intended host
selection only after proving complete mandatory coverage, proves each routed call reaches its kernel
in the linked ELF, and runs
`audit_built_device_host_compute`. That audit refuses host arithmetic or host payload inside any
function that implements an accelerator group, and refuses undefined external symbols.

The other route is the explicit public-program treatment, which Phase 2 measurement selects. There
the candidate emits the whole program, host code included.

## What changes

The owned-host route binds its independently derived eligibility to Phase 0's frozen accelerator
obligations. Host placements supply the default implementation for remaining work; they do not
prohibit supported fusion or standalone optional accelerator groups. This is a shared policy for
all accelerators; the selected capability data determines which work the device can implement.

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

**Split check.** `frozen_placement.require_split` runs before candidate invocation, after the build,
and again during re-audit. Before invocation and after the build, `_source_placement` also checks
each independently eligible source operation's actual group assignment.

The fresh independent eligibility must equal the frozen accelerator census. Device groups must
include that work and contain only operations from the exact original source. Additional members
selected by Merlin's ordinary capability checks are recorded as `supporting_device_operations`.
An omitted required operation, an operation assigned to the host despite its accelerator obligation,
an invented source member or changed eligibility still refuses qualification. Numerical, emitted
device work and host-compute checks apply to the resulting fused program.

**Optional groups.** Source admission derives a separate optional roster from Merlin's non-host
groups outside the mandatory set. Each record retains the original root's region/node identity and
static output extent; emission cannot invent an optional source group. The source join requires all
mandatory identities and permits any subset of those optional identities. Both rosters reject
duplicate or changed provenance. Every emitted group receives the same linked-body, host-compute and
complete-output checks. Optional linked groups are counted separately and cannot increase mandatory
coverage. A group lacking complete source evidence retains an explicit refusal if selected; ordinary
host placement remains available for work outside the mandatory set.

## Integrity

- **Accelerator work cannot move to Merlin's host.** The candidate never chooses the split. If it
  declines a group, the existing sidecar check refuses the grade. If an eligible operation lands in
  a host group, `_source_placement` refuses it. A device group count below the frozen accelerator
  count now refuses as well.
- **Merlin supplies the host implementation.** The candidate receives closed device-group interface
  programs, including supported attached stages and optional groups. Operations outside those groups reach Merlin's host
  package, whose tree digest is checked before and after the build. Supporting fusion grants no
  additional accelerator coverage credit.
- **The host-compute audit is unchanged.** It still runs on every linked device function, and
  unknown remains not clean. The frozen-placement check comes in addition to it, never in its place.
- **Nothing is weakened.** NOT_RUN_IS_NOT_PASS, tier requirements, sandbox masking, and
  golden/hidden/sealed checks are untouched. The end-to-end comparison still covers every output.

## Input preparation and protocol findings

`input_preparation.source_convolution` binds a simple integer convolution's pointer order and
buffer extents to its admitted original source and the selected logical harness ABI. The host-compute
veto can use `verify.conv_input_preparation` to exclude only independently proven bounded copies,
padding and their address/control work. Private scratch globals require actual initialization and
consumption by decoded accelerator loads. CPU arithmetic on tensor values, CPU output stores,
unknown pointers and unproved effects retain the ordinary audit outcome.
The load-state recipe can select an original FIRRTL byte-count sink and its typed row/column fields.
A shared structural derivation bounds the transfer's element width from every source expression
alternative, checks the decoder's field offsets/widths, and retains unknown for unsupported forms.
The target supplies the source selectors as data, never an assumed element width in library code.

Resident-reuse qualification checks actual transfers, available operand identities and the required
execution tiers. Reissuing identical execution configuration is reported in
`execution_config_advisory` as a performance opportunity; it does not itself demonstrate a weight
reload. Actual redundant resident loads still refuse the declared residency requirement. Decoder
support for constant byte GEPs and closed read-only globals retains original address identities.
These rules apply to new qualifications and never rewrite historical frozen results.

## Out of scope

- The Phase 2 public-program treatment still measures a candidate-emitted whole program. Applying
  the same split there is a follow-up for the Phase 2 owner.
- The frozen decisions carry no operation ordinals, so the binding is by operation name and count
  rather than by ordinal. A future Phase 0 schema that keeps the ordinals can tighten it to exact
  operation identity without changing the route.
