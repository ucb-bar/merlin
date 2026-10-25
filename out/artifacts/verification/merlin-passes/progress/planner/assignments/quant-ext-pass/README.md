# Assignment: verify Merlin's lower_quant_ext pass

**Current planning priority:** the user-approved
[eight-day scoped paper goal](../../eight-day-paper/README.md), targeting 2026-10-30.
Continue the current active record in the quant-ext checkout: finish provenance-bound frozen
qualification/replay and connect the concrete reciprocal proof to the mixed-pass theorem.
The original sequence below is background, not an instruction to restart completed phases.
Existing holds and ownership remain unchanged.

Assigned 2026-10-22 to the existing Qinner/other-pass worker, after completion of the isolated
Qinner delivery harness. The user requests actual compiler-pass verification next. Preserve
the Qinner delivery branch, review packet, patch, receipts and uncommitted changes. Do not
extend that harness as this assignment. The user-controlled staging/commit/push hold remains.

## Goal and first priority

Implement Lean translation validation for the actual `lower_quant_ext` transformation in
`src/merlin/llvmlower/passes_xdsl.py`, extending beyond the existing signed-i8/i32/f32 narrow
dequantization proof. Start with **per-tensor signed-i8 dequantization with i64 zero points,
f32 scale/result, and the full-range and reduced-range metadata observed in captures**.

The earlier public-capsule census found 627 per-tensor dequantizers with i64 zero points:
295 restricted-range and 332 full-range. These are inventory counts, not selected-model
coverage or Lean premises. They motivate this task; derive all checker facts from actual IR.

This is scalar/tensor/whole-boundary proof and checker work. It must produce acceptance of
actual production-pass output in its supported domain, with complete-output coverage and
explicit refusal elsewhere. A runtime harness, another census or a scalar lemma alone is
not the deliverable. Work autonomously through implementation and qualification for several
hours; do not return after merely identifying the known contract gap.

## Work sequence

1. Inspect the existing Dequant scalar/tensor/site/definedness proofs, the current pass and
   versioned source-reference intake. Read `Rules/W8A8DequantWidth.lean` and existing source
   adapters **read-only**. Write the intended theorem and exact domain before the checker.
2. Define an additive, explicitly provisional source-contract adapter for the supported i64
   per-tensor case and accepted metadata. Retain PyTorch's exact order: signed integer-to-float
   conversion of q and zero point, f32 subtraction, then multiplication by the actual scale.
   Prove compatibility with the existing i32 contract where applicable. Range metadata,
   parameter rank/shape, signedness, widths, attributes and poison are part of the claim.
   The implementer has read the pass; this candidate is not independently approved or
   barrier-authored. Do not install it into frozen dispatch or label it reviewed.
3. Prove the scalar body and lift through actual tensor maps, parameter addressing, allocation,
   writes and SSA. Check complete function/module outputs, signatures, calls, regions, attributes,
   fresh binders and escaping uses. Reuse generic correspondence and existing proved components
   read-only. A target-derived guess must never define the source contract or coverage.
4. Prove source support and initialized-output execution for the supported dequantization
   operation/domain separately from conditional equivalence. Make the typed, initialized domain
   explicit and inhabited; do not assume type annotations imply actual runtime typing. Do not
   turn local definedness into whole-model success or poison freedom.
5. Integrate an isolated checker/driver, byte-bound receipts and replay for this pass. Run actual
   `lower_quant_ext` on complete inputs and verify its entire output. Exercise retained real
   boundaries and report eligible/accepted/refused site counts from their source programs.
   Zero changes are identity evidence only. If a real boundary also contains quantizers, it
   must remain explicitly refused until those changes are accounted for; do not filter out
   rewrites, check isolated sites only, or relabel a modified fixture as that model boundary.
6. After the dequantization milestone, continue toward mixed actual `lower_quant_ext` boundaries:
   connect the emitted per-tensor quantizer lowering to the existing ordered W8A8 source-quantizer
   candidate using read-only adapters. Preserve its binary64 reciprocal then f32 conversion,
   zero-point addition, rounding/clamp order and exact domain. This concerns equality of the
   source quantization expression and its emitted lowering, including poison; **the other worker
   owns upstream W8A8 RMSnorm/gain numerical definedness**. Do not duplicate that work or assume
   numerical laws to force an initialized whole-model result.
7. Qualify with compiler-valid corrupted outputs, relevant pinned-source/native differential
   cases, affected regressions, fresh imported-dependency/axiom audits and full receipt replay
   on a frozen private build. Bind actual source/target exports, source contracts, checker,
   toolchain/exporter and capture bytes. Report all unavailable/refused cases explicitly.

## Required negative controls and limits

- Preserve F6's historical i32 per-channel wrong-order witness and F12's still-open i64
  per-channel discrepancy. For q=1, zp=16777217, scale=1, source integer-subtraction order
  and the production i64 per-channel float-subtraction order disagree. A per-tensor theorem
  must not justify that per-channel path. Refuse it under the faithful source contract;
  never redefine its reference to match the bug or repair production code in this assignment.
- Test i32/i64 endpoints and signed interpretations, full/reduced bounds, zero/nonzero zero
  points, scale bits, NaN/infinity/signed zero as supported by the explicit claim, rank-zero
  and empty tensors, higher-rank shapes, wrong parameter shape/width, altered arithmetic order,
  maps, casts, flags, SSA wiring and declaration/return changes. Choose meaningful cases for
  the implemented fragment. Do not execute undefined/poison conversions natively.
- Per-channel, per-group, other storage widths, mixed precision and unsupported metadata
  remain declared coverage gaps until individually proved. Do not claim all of lower_quant_ext
  correct merely because its high-priority per-tensor forms pass.
- Missing final capture selection or human admission does not prevent additive proof, checker
  and qualification work. Keep unreviewed contracts and actual-model coverage separate.

## Isolation and ownership

Create a new separate worktree/branch from current committed `feat/pass-verification`; do not
switch/reset the dirty main checkout or repurpose the dirty Qinner delivery worktree. Inspect
current refs first and report the exact base. Reuse dependency artifacts read-only; build into
your own private snapshot. Do not modify W8A8's preserved runtime or frontend's build.

Own new `Rules/QuantExtLowering*.lean`, `Check/QuantExtLowering*.lean`,
`Driver/QuantExtLowering.lean`, separate `QuantExtLoweringCandidate.lean`/Lake configuration,
`src/merlin/verify/lean/quant_ext_pass.py`, dedicated tests/fixtures/qualification script and
`out/artifacts/verification/merlin-passes/quant-ext/` evidence/reports. Announce exact owned
paths in your own conversation. Existing generic Dequant/W8A8 modules are read-only inputs;
put additive lemmas/adapters in the new owned files instead of editing another worker's code.

No production-transform changes, frozen Core/Sem/Typing/Types changes, `Spec/`, `Claims/`,
`Assumptions/`, pins, `SPEC_LOCK`, shared Main/CLI/exporter/central ledger/findings edits.
Frontend still owns TorchAO affine captured-call verification and its numerical laws. W8A8
still owns W8A8 source-domain propagation, primitive numerical interfaces and result 1256.
Coordinate a real shared-interface need before touching those files; continue independent
owned work while waiting. No new agents are assigned or needed.

No sorry/admit/unreviewed axioms, native_decide/bv_decide, unsafe/partial/extern or implementation
substitution under MerlinVerify. Audit the actual imported closure. Keep existing tests and
counterexamples unchanged. Report the unrelated structure ceiling failure separately.

## Completion report

Deliver the exact pass claim/domain and proof map, the complete supported source-site census,
accepted actual-pass boundaries and actual-model limits, valid mutation refusals, numerical
counterexamples, tests/audit/replay, remaining source-contract/review/concrete-model assumptions
and the next unsupported variant. Include a coherent reviewable patch with explicit owned paths;
leave it uncommitted and unpushed under the user's hold. Report directly to the user in your
own conversation, without asking the stopped planner for periodic assignments.
