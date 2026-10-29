# Human review requested: zero-init bf16/f16 f32 MACC policy

**Not approved. Agent review is not human approval.** Current scope is the parent assignment's `MACC_ONLY.md`. No toolchain/loader/pinning-system review is requested here.

Merlin's shared `lower_bf16_matmul_f32acc` pass changes half-precision accumulation into f32 accumulation. We want to verify that its actual output implements the following deliberate policy, **not** claim equality to the original half-accumulating program. Please approve or correct this concrete recipe and its scope:

```text
Same-format bf16 or f16 operands and result; static shapes and checked source maps.
Source destination: literal positive-zero splat, or positive-zero fill of tensor.empty.
For each result cell:
    acc = positive-zero f32
    for the source loop points mapping to that cell, in lexicographic order:
        x = attribute-free extf(left element) to f32
        y = attribute-free extf(right element) to f32
        product = mulf[F_mul](x, y)
        acc = addf[F_add](acc, product)
    result = one attribute-free truncf(acc) to the original half format
F_mul and F_add are each exactly none or contract; no other fast-math flags.
```

Concrete policy source: `merlin/lean/MerlinVerify/Rules/Bf16F32Acc{Contract,Body,Reference}.lean`. Exact current source hashes are in `MACC_REVIEW_INPUTS.json`. `ACTUAL_SITE_0.json` shows a real retained SmolVLA source contraction and its complete six-operation replacement; `IR_MAPPING.md` explains the link. These are the artifacts to review, not an intended future implementation.

Please record decisions on these six points:

1. **Zero only:** approve refusing nonzero, negative-zero and unknown initializers. F14 shows that the base pass drops nonzero initial state. Supporting `extf(C0)` is a separate policy; it is not included here.
2. **Ordered arithmetic:** approve the source maps and lexicographic per-cell update order, multiply left then right, add accumulator then product. Source bodies with swapped scalar mul/add operands are recognized but the policy chooses this canonical order; no arbitrary floating commutativity law is assumed.
3. **Fusion:** approve none/contract as an explicit policy permission, without claiming a single bitwise result across arbitrary backend fusion choices. The existing fused-model theorem covers a uniform pattern with one format, injective widening and no unrelated licensed operations. F18 gives bf16 infinity versus finite output for fused/unfused evaluation. Mixed fusion and NaN-payload behavior remain fidelity obligations.
4. **Rounding:** approve attribute-free conversion in the selected interpretation. The proof does not establish target rounding mode or a general IEEE correspondence. MLIR's current documentation distinguishes default folding/canonicalization rounding from backend runtime behavior. No universal RNE or accuracy claim is proposed.
5. **Eligibility and claim:** approve static same-format two-input named matmul or generic contractions directly in single-block functions, with checked maps, literal initialization and complete changed-output comparison. Nested rewrites, named batch_matmul, arbitrary mixed formats and whole-model definedness are excluded.
6. **Review independence:** acknowledge that the candidate was authored after reading the pass. Please identify the independent reviewer, date, exact reviewed source hashes, decisions and any conditions. Agent-written hash files or passing tests do not replace this review.

Proposed paper sentence, conditional on fresh successful checks and independent approval:

> For accepted recorded zero-initialized bf16/f16 MACC preparation boundaries, our verified checkers establish complete output equality to a source-derived f32-accumulation policy reference under explicit SSA-renaming and call-locality laws, and strict policy refinement of every reconstructed contraction on typed, initialized local tensor inputs at index64. This result does not establish equivalence to half-precision accumulation, whole-model definedness, or bit-exact downstream floating-point execution.

The reference is source-derived **with independently permitted flag/annotation choices**, not a unique choice of runtime fusion. There are24 distinct historical SmolVLA sites; the segment repeats them, and TinyLlama has none. Historical evidence is not fresh qualification of this branch. F supplies current build/check/replay outcomes separately. No approval-dependent accepted gate result exists at review preparation time.

References for interpretation: [MLIR Arith](https://mlir.llvm.org/docs/Dialects/ArithOps/) and [MLIR Linalg matmul](https://mlir.llvm.org/docs/Dialects/Linalg/#linalgmatmul-linalgmatmulop); repository `merlin/lean/LEDGER.md` A5,A6,A7,T1–T4,T6. These sources inform review; they do not prove this numerical policy or identify the actual target's floating implementation.
