---
title: EL4 V26 Phase 2 setup selection
kind: guide
status: current
owner: experiments
last_verified: 2026-10-19
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

This setup is held on the managed Phase 2 task branch. The combined integration
tip descends from published `feat/el4-experiment`; its owner can fast-forward
that branch after checking the active EL4 worktree. Do not move a branch ref
behind another active checkout or treat this local integration as a push.

For another server, use the
[portable instruction-admission guide](portable_generated_target_admission.md)
and the companion target repository's `docs/v26/portable-runtime-and-compiler.md`.
They distinguish published objects from local integration commits awaiting
publication, identify externally provisioned runtime/source inputs, and describe
fresh current-host issuance. First-server paths and saved receipts are not a
portable qualification bundle. The newer shared-pass checkpoint `08570e926` was
reported unpushed and was unavailable during this integration; shared passes
through published `dbc762306` are included.
