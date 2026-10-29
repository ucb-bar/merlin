# Human review: Merlin's bf16 accumulation boundary

This packet covers `lower_bf16_matmul_f32acc`, a Merlin shared preparation pass between
model2MLIR's emitted IR and the OOT compiler handoff. It asks for approval of the numerical
contract and the precise pass claim below. The optimizing Python implementation remains
untrusted; the Lean acceptance checks establish the claim for accepted before/after artifacts.

## Claim proposed for the paper

For every accepted recorded bf16 preparation boundary, the complete output program equals a
source-derived f32-accumulation reference under every common operation interpretation that
respects SSA renaming and call locality. Every recognized contraction is also checked as an
actual source/target fragment: on all typed, nonpoison tensor operands it strictly implements
the declared policy at the receipt's index width (64 bits on the retained captures). The source
operations and their order are retained in the coverage theorem,
and the operand domain is proved nonempty for every admitted shape and format.

This is a modular pass claim. It does not assert successful nonpoison execution of the entire
SmolVLA capture, correctness of model2MLIR, correctness of the OOT compiler, or correctness of
later shared passes that have not passed their own checks. No unmodelled surrounding operation
is declared verified: the complete-output equality theorem is parametric in its interpretation.
The old whole-capture qualification fields remain separate and false on the retained models.

This follows the verified-checker approach in [Axon §§2.1 and 5.3](https://arxiv.org/html/2605.01660v3).
Human review targets the policy and semantic assumptions; accepted optimization results are
checked mechanically. Axon's §5.1 also makes its floating-point/machine correspondence explicit.
Our scoped claim preserves that separation and does not establish bit-level backend fidelity.

## Numerical contract to approve

For each admitted 16-bit named `linalg.matmul`, or two-input `linalg.generic` contraction
whose body is `yield(acc + a * b)` and whose destination is a `+0.0` splat or fill:

    acc := +0.0 (f32)
    for each loop point mapped to the result cell, in lexicographic loop order:
        acc := addf[F_add](acc, mulf[F_mul](extf(a), extf(b)))
    result := truncf(acc)

Operands are read through the source's own indexing maps. Operand conversions and the final
conversion are attribute-free; the latter uses default rounding. `F_mul` and `F_add` are each
exactly `none` or `contract`. No reassociation or other fast-math permission is inferred.

The current specification consists of:

- `Rules/Bf16F32AccContract.lean`: the six numerical clauses;
- `Rules/Bf16F32AccBody.lean`: scalar instruction templates and their execution theorems;
- `Rules/Bf16F32AccReference.lean`: source eligibility and the source-derived reference transformer.

The pin patch also freezes the shared coordinate/ordered-fold/fill definitions used by the
policy, the local domain and coverage definitions, `ParametricAlpha.Laws` and its operation-head
comparison. These definitions
give names in the claim statements their meaning; pinning only the theorem types would leave
them changeable. This deliberately locks those mixed definition/proof files conservatively.

The implementer read the pass before writing the candidate. Independent human review is
therefore required; this packet does not claim that the specification was barrier-authored.

## Machine-checked evidence

- `Check.Bf16F32AccExact.check_sound`: complete output equality to the source-derived reference,
  for any operation interpretation with the stated SSA and call laws.
- `Check.Bf16F32AccSites.check_sound`: every reconstructed source/target fragment passes strict
  typed, initialized-input policy qualification; complete output matching is also mandatory.
- `Check.Bf16F32AccSites.collectOps_coverage`: no recognized source contraction is omitted or
  duplicated, and source operation positions and order are preserved.
- `Rules.Bf16F32AccNonvacuity.operand_domain_nonempty`: actual initialized tensor operands exist
  for every admitted local domain, including shapes with zero extents.
- Tensor and operation bridge theorems connect emitted arithmetic, indexing, initialization and
  truncation to the contract. Existing conditional and strict suffix theorems remain available.

The retained SmolVLA boundary contains 24 recognized contractions. All 24 local obligations
pass, and its complete output matches the source-derived reference. The SmolVLA segment repeats
those same sites; it does not add another 24 observations. TinyLlama has zero bf16 sites and
provides only identity/suffix evidence. [candidate.json](candidate.json) binds the exact receipts,
checker, policy bytes and elaborated statements; [FINDINGS.md](../FINDINGS.md) records validation.
The packet generator replays the actual receipts before producing the candidate. A regression
requires it to reject a forged green report even after that report's hashes are recomputed.
Publication also checks all locked semantic files and audits every imported Lean dependency of
the actual checker and the contract theorems; it checks bindings again after the long replays.

## Decisions for the reviewer

1. **Initialization:** approve refusal of nonzero destinations, or request a contract starting
   from the extended original destination. The current pass drops nonzero initialization (F14).
2. **Fusion and model:** approve `contract` as a numerical policy permission, including the stated
   trusted-lowering assumptions. `Bf16F32AccFusion.fused_step` proves the uniformly fused model
   case under extension injectivity; it does not prove arbitrary mixed backend fusion. F18 shows
   a real bf16 fused/unfused difference, so a bit-identical IEEE claim would be incorrect.
3. **Eligibility:** approve the checked two-input contraction scope, including the source's own
   maps. Named `batch_matmul` and one-input generics remain outside this policy.
4. **Order:** approve the lexicographic reference order (A5).
5. **Paper wording:** approve the modular pass claim above. A composed claim for the complete
   shared interval additionally requires each other changed boundary and its source-domain
   obligations; these are not supplied by this packet.

## Mechanical steps after approval

`after-approval.patch` is generated but is not applied. It installs the approved contract under
`Spec/`, preserves old imports through a shim, locks the policy templates/transformer, and adds
all boundary, bridge, exact-match, local qualification, coverage and nonvacuity statements to
`PINNED`. It contains no change to the optimizing pass. Apply it only after independent approval:

    git apply --check out/artifacts/verification/merlin-passes/frontend/bf16-f32acc/review/after-approval.patch
    git apply out/artifacts/verification/merlin-passes/frontend/bf16-f32acc/review/after-approval.patch
    .venv/bin/python build_tools/scripts/check_lean_guard.py --update-lock
    .venv/bin/python build_tools/scripts/check_lean_guard.py

Coordinate the lock update with W8A8: every receipt bound to the old lock becomes stale. Rebuild
and regenerate the bf16 receipts using the stages already bound in their reports, replay them,
and publish the packet. The retained validation runner performs those mechanical steps, using
the current receipt bindings (or archived bindings after an interrupted attempt) to recover the
same stage roster:

    .venv/bin/python out/artifacts/verification/merlin-passes/frontend/bf16-f32acc/validation/review_jobs.py

It builds and audits the checker closure, runs the fast tests, regenerates all three receipts,
runs their slow replay tests, and invokes the publisher. A failure stops publication. To replay
and publish already regenerated receipts directly, use:

    .venv/bin/python build_tools/scripts/prepare_bf16_review.py \
      --compiler-python /home/jeremy/Projects/model2MLIR/.venv/bin/python

Before approval, this command can produce only `human_review_pending`. After the installed
specification bytes and every current theorem statement match `SPEC_LOCK`, it can produce the
scoped `claim.json` with `status: verified`. Changed policy bytes or a missing statement pin
revoke approval. Legacy whole-capture receipts keep their own qualification status; approving
this pass claim does not qualify the entire model or the whole shared interval.
