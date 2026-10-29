# A5 — Frontend repairs and structural rewrites

**Branch:** `pv/frontend`. **Mission:** verify the preparation passes that repair the capture or
restructure it: the bool-cast fixes, the over-rank matmul collapse and scalar-square
canonicalization, plus bf16 accumulation if it fires.

## The key idea

These passes don't preserve the literal MLIR meaning of their input. They repair it. A signed `i1`
cast gives `-1.0` where PyTorch meant `1.0`, and an over-rank matmul isn't valid linalg at all. So
the claim isn't "the output refines the input". It is "the output refines `Intent(input)`", where
`Intent` is a reviewed spec transformer from A2's `Spec/FrontendIntent.lean`. The claim type is
`policy "<name>"`, the same mechanism as W8A8. Agree the shared shape of these claims with A3 and
A6 (A0 coordinates), so composition treats them uniformly.

## You own

- `MerlinVerify/Rules/FrontendIntent*.lean`, `Rules/ScalarSquare*.lean`, `Rules/Bf16*.lean`.
- `MerlinVerify/Assumptions/FloatLaws.lean`, only after the approvals in item 3.
- `merlin/tests/ir/test_lean_frontend*.py`, `test_lean_scalar_square*.py`.
- `out/artifacts/verification/merlin-passes/frontend/`.

## Work, in priority order

Drop any item that doesn't fire on the chosen model (A1's census, confirmed by A0 at Gate A).

1. **`prepare-fix-bool-sitofp` and `prepare-fix-bool-fptosi`** (`fix_bool_sitofp`, 34 lines;
   `fix_bool_fptosi`, 44 lines; in `src/merlin/llvmlower/passes_xdsl.py`). Prove the output refines
   `Intent_bool(input)`: boolean to float gives {0.0, 1.0}, and float to boolean means `x != 0`.
   - Today's preparation gate has a specific refusal regression for the signed-to-unsigned repair.
     Keep that test, and add the new verified path beside it.
   - Mutations: keep the signed cast, reverse the comparison, truncate instead of comparing with 0.

   1–2 days.
2. **`prepare-collapse-overrank-matmul`** (`collapse_overrank_matmul`, `passes_xdsl.py:54`, 62
   lines). The source over-rank matmul is stuck in Lean because it's invalid linalg.
   `Intent_overrank` gives it `aten.linear`'s batched meaning. Prove the collapse, the 2-D matmul and
   the expand compute it.
   - Reuse `Rules/ReshapeGroups.lean` and `Rules/ReshapeIndex.lean` for the reshape algebra.
   - Mutations: a wrong reassociation, swapped operands, a missing expand.

   1–2 days.
3. **`prepare-scalar-squares`** (`canonicalize_scalar_squares`, `src/merlin/llvmlower/scalar_square.py`,
   34 lines; `powf(x, 2.0) → x * x`). This can't be proved with parametric floats.
   - **Approvals:** only after Agustin approves the law (decision 4) and Jeremy approves the file,
     add `Assumptions/FloatLaws.lean` with **one** axiom, stated over `FloatModel`: `powf x 2.0 =
     mul x x`, at the exact formats used. Add its ledger row in `merlin/lean/LEDGER.md` (A0 merges).
   - Prove the rewrite under that axiom. Mutations: `powf(x, 3.0)` must refuse, and so must `x * y`.
   - If the approval doesn't come, mark the boundary assumed and move on.

   About 1 day.
4. **`prepare-bf16-matmul-f32acc`** (`lower_bf16_matmul_f32acc`, 132 lines). Only if A1's census
   shows it fires. It changes accumulation precision, so it's an intent or policy claim, not an
   equivalence. Scope it with A0 first.

## Don't

- Don't change `passes_xdsl.py` or `scalar_square.py`.
- Don't add any float law other than the single approved one.
- Don't touch `lower_quant_ext` (A4) or the `lower_model` preprocessing in `passes_xdsl.py` (A6
  and the existing gate).
