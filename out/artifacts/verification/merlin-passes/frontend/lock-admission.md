# Frontend integration admission request

The user approved the documented f32 source policy and direct IEEE comparison
instance on 2026-10-21; see `f32-policy-approval.json`. This does not authorize this
agent to edit SPEC_LOCK. No frozen semantic or compiler file was changed.

The full guard reports exactly these locked-file differences:

- Added `MerlinVerify/Spec/FrontendIntent.lean` (the inherited A2 file also contains
  bool-to-f32 and linear primitives; the new approval covers only f32-to-bool).
- Changed `schemas/lean_receipt.schema.json`.
- Changed `schemas/lean_chain.schema.json` (inherited from the prior integration).

The audit separately reports 7,860 theorems and zero violations. File admission
and statement pinning remain the integration owner's responsibility. Do not
regenerate the lock merely to remove a failing status.

## Proposed statements to review and pin

| Statement | Module |
| --- | --- |
| `FrontendBoolSimulation.check_sound` | `MerlinVerify.Rules.FrontendBoolSimulation` |
| `FrontendBoolSimulation.check_sound_ieee` | same |
| `FrontendBoolQualified.support_sound` | `MerlinVerify.Rules.FrontendBoolQualified` |
| `FrontendBoolQualified.defined_sound` | same |
| `FrontendBoolChain.chain_after_source_policy` | `MerlinVerify.Rules.FrontendBoolChain` |
| `FrontendBoolChain.chain_after_source_policy_strict` | same |
| `FrontendF32Comparison.model_correct` | `MerlinVerify.Rules.FrontendF32Comparison` |
| `FrontendBoolDefined.compareRule_sound` | `MerlinVerify.Rules.FrontendBoolDefined` |

Names in the first column have namespace prefix `MerlinVerify.Rules.`.
Review exact elaborated statements with `merlin-lean-audit --statements` before
adding them to guard pins. The general statements expose `UneCorrect`; the
`check_sound_ieee` specialization discharges it under the approved direct model.
There is no original numeric-cast equivalence claim.

## Scope of requested admission

Admit the frontend receipt fields (source-program proof, scoped approval and
separate operational qualifications) and conditional policy composition. Preserve
the ordinary chain's original-source gate. Do not infer approval of the remaining
A2 primitives from this f32 decision or claim full SmolVLA execution support.

The exact full-model support refusal and evidence are in `fix-bool-fptosi.md` and
`qualification-20261021.json`. Solving that refusal requires additional support
work or reviewed semantics independently of this completed repair simulation.

## Concrete unapplied candidate

`prepare_admission.py` generates `admission-candidate/admission.patch`,
`SPEC_LOCK.candidate`, `statements.json` and `review.json`. It verifies the
candidate with the authoritative guard comparison and refuses unrelated drift.
The proposal has exactly three file-hash differences plus the eight statement
pins above. Existing frozen statement hashes are unchanged. The real guard
and SPEC_LOCK remain untouched; candidate consistency is not human admission.
