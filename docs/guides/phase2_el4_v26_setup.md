---
title: EL4 V26 Phase 2 setup selection
kind: guide
status: current
owner: experiments
last_verified: 2026-10-20
related: [phase2_setup_execution, shared_pass_phase2, phase2_public_diagnostic_launch]
---

# EL4 V26 Phase 2 setup selection

The selected Phase 1 origin is `el4-phase1-capsule-20261019-fresh-v1`, published
with Merlin `f8a43c657` and experiment archive `8b8fd3629`. The machine-readable
[selection](../../examples/gemmini/phase2/el4_20261018/selected-phase1-v26.json)
records setup inputs and qualification limits. It does not launch Phase 2 or
replace the frozen objective of an earlier campaign.

The adjacent managed `gemmini-mlir` repository preserves the complete original
submission under `phase1_generated/el4_fire_aligned_v26_original_29213759/package`.
All 16 files match the original run inventory and package SHA-256
`29213759089b71de26f8a5ab26ca6ef1b8071dc0bb690041eb7de5d0efb53f24`.
The separate 54-file public archive remains byte-identical; its 14-file compiler
projection has a different digest. V23 and its historical measurements remain
unchanged. Prepare candidate copies separately; never edit these origin packages
or substitute the handwritten backend.

The V26 archive reports 152 public numerical passes, one expected refusal and
eight hidden aggregate passes. Resource preparation failed before invoking the
compiler. The matching-executor fix is integrated, but neither focused tests nor
copying the compiler completes the original resource check. The Phase 1 owner
must issue the original capsule certification/publication after that check,
retaining its 120-second aggregate budget and 1 MiB program limit. V26 complete
ResNet50, TinyLlama and SmolVLA validation is not established by this selection.

The selected facts, functional Spike, extension and historical gSIM window match
the recorded low-level identities. The selected software specification and
descriptor contract have their own V26 hashes. The earlier observed capture
contract is provenance, not a substitute descriptor. Old SDK selections,
admission receipts, numerical contracts and calibration domains do not transfer
automatically. The 159 published timing rows cover seven earlier compiler
snapshots, not the final original compiler; original raw observations and exact
source/schedule bindings are required for calibration intake.

For shared passes, ordinary source-level transformations may retain the public
call ABI. This generated compiler exposes parsing, interface lowering,
command-buffer emission, LLVM lowering and combined analysis. It does **not**
declare `target_ir_consumer` or `emit_target_bundle`. Instruction-level
transforms require an actual consuming interface and checked effects; adding
manifest labels alone supplies neither. An explicitly selected additional
library is permitted, with equal algorithms and source grants for both arms and
fresh functional checks. A new Phase 1 authoring run is not inherently required
to select such a library.

Before automatic execution:

1. Complete original capsule admission or use the clearly scoped public
   diagnostic route. Reconstruct the real linked-ELF admission service from its
   original reviewed selection and source/decoder inputs; a saved receipt or
   neutral callback is insufficient.
2. Check each retained public development source and independent reference under
   the selected V26 contracts. Preserve every refusal and keep guards separate
   from performance objectives. Freeze sources, roles, coverage and boundaries.
3. Select the shared release and compiler edit authority explicitly. Verify the
   enabled mechanisms on emitted programs. Keep target-specific encodings and
   schedules in the target repository and reusable algorithms in Merlin.
4. Validate Spike timing for each required objective using original public
   gSIM evidence. Capacity, attention, carried state and host cost remain
   UNKNOWN unless the required gates pass. Phase 2 iteration stays Spike-only.
5. Complete an unchanged queued control and a distinct candidate through output
   checks, fresh counts, qualified prediction, feedback, decision and shutdown.
   The retained V23 diagnostic produced both arms' outputs and counts but hit
   its original 600-second worker deadline during final verification; its
   partial result does not establish this completion test.

The combined integration is delivered on `feat/el4-experiment` in Merlin and
the companion target repository. It includes shared passes through
`dbc762306`, the Phase 2 infrastructure and portable public inputs. Record the
actual checked-out integration revisions separately from the original Phase 1
harness `f8a43c657`. Publishing this infrastructure does not recertify Phase 1
or launch Phase 2. The dated V26 experiment branch stays frozen; its history and
compiler bytes are not rewritten. Remote publication also does not update an
owner's already active local EL4 checkout.

For another server, use the
[portable instruction-admission guide](portable_generated_target_admission.md)
and the companion target repository's `docs/v26/portable-runtime-and-compiler.md`.
They distinguish the original experiment from subsequent integration commits,
identify externally provisioned runtime/source inputs, and describe
fresh current-host issuance. First-server paths and saved receipts are not a
portable qualification bundle. The newer shared-pass checkpoint `08570e926` was
reported unpushed and was unavailable during this integration; shared passes
through published `dbc762306` are included.

The companion target repository also contains exact public inputs in
`docs/v26/inputs` and three portable public fixtures in
`examples/phase2/public_pass_inputs_v26`. Run `python3 -B verify.py` from that
fixture directory to check its compiler, inputs and independent references.
The check executes no compiler or target. The fixtures have fresh relative-path
descriptors, not transferred capture/admission authority; current-host capture,
linked-ELF checks and complete execution remain required.

## Setup integration on 20 October

The following connections are implemented through ordinary tooling. Each keeps
planning, live issuance and accelerator execution separate:

| Connection | Reproducible entry | Evidence and remaining gate |
| --- | --- | --- |
| Complete development intake | [Phase 0 admission planner](phase2_development_admission_plan.md) | Reopens the pinned 243-member inventory and proposes four bounded jobs; 13 unmapped requirements remain refused. No current V26 member is admitted by planning. |
| Original timing archive intake | `spike-timing archive-count-plan` and `packages/merlin-timing/COUNT_INTAKE.md` | Actual V26 export retains 159 original records, 100 exact source-byte groups and 145 original join groups. All requests remain UNKNOWN until original counter/window/domain reviews and fresh counts are attached. Existing labels are reused without rerunning gSIM. |
| Original instruction admission | [Operator CLI](portable_generated_target_admission.md) | Default check mode was exercised on the real V26 selection. Original live preparation and static ELF admission were separately exercised; these do not establish numerical or timing qualification. |
| Queued development priorities | [Explicit native decision policy](phase2_public_diagnostic_launch.md) | Completed queued feedback can produce a conservative interval priority. Host tests cover refusals and preserve measured-promotion boundaries; a complete target queue run remains pending. |

The launch-critical work remains actual V26 source/profile/capability binding,
verification overhead within the original worker budget, completed control and
distinct-candidate queue runs, source admission/freeze, and qualified timing for
every required performance objective. The archive intake does not itself expand
the calibrated domain. Resolve attention arithmetic explicitly or retain its
refusal; do not remove it from required coverage to obtain a freeze. The Phase 1
owner still owns the original resource check and capsule certification.

Optional unqualified asynchronous, spill and instruction-rewriting mechanisms
remain disabled. Complete roofline attainment remains UNKNOWN where physical
traffic, packing, host feeding or legal overlap are unresolved. No formal Phase 2
campaign or performance promotion is established by this setup integration.
