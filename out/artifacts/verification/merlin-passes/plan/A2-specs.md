# A2 — Spec author (blind)

**Branch:** `pv/specs`, in a sparse worktree. **Mission:** write the independent specifications
that the other agents prove Merlin's passes against. Your specs define what "correct" means, so they
must come from the declared policy and the PyTorch/TorchAO references, **never** from Merlin's
implementation. Otherwise a spec copies the bug it should catch (design §1.2 rule 6).

## Information barrier

- Your worktree omits Merlin's quantization and repair implementations. Don't read those paths
  anywhere else either: not in another checkout, not with `git show`, `git log -p` or `grep` over
  history, and not through tests that exercise them.
- **Allowed sources:**
  - PyTorch 2.10.0 source and docs (the pinned oracle in
    `build_tools/toolchains/lean-torch-requirements.txt`);
  - TorchAO source at the exact version the capture environment uses;
  - the declared recipe (`Int8DynamicActivationInt8WeightConfig`, the int8 dynamic-activation,
    int8-weight scheme);
  - MLIR dialect docs;
  - the existing Lean semantics (`MerlinVerify/Sem/`);
  - `Spec/Dequant.lean` as a style model;
  - design §5.5 as a statement of intent.
- **TorchAO isn't pinned in this repo.** Find the version installed in the capture environment
  (ask A1 or A0 for the interpreter path), record its version and commit in your spec notes, and
  cite `file:line` at that version for every operation-order decision.
- If design §5.5 and TorchAO's source disagree, **TorchAO wins**. Record the difference.

## You own

- `merlin/lean/MerlinVerify/Spec/TorchAO.lean`, `Spec/FrontendIntent.lean`, `Spec/Policy/W8A8.lean`.
- `out/artifacts/verification/merlin-passes/specs/NOTES.md`: citations, open questions, and the
  differences from design §5.5.
- Oracle tests for your specs: `merlin/tests/ir/test_lean_spec_<area>_torch.py`, using
  `MERLIN_TORCH_PYTHON`.

## Deliverables

1. **D3: `Spec/TorchAO.lean`.** The source contract for the TorchAO affine quantization ops that
   appear in the captures. A1's census lists which ones; ask A0 if it isn't out yet. Likely
   candidates are the choose-qparams, quantize and dequantize affine ops the recipe uses. For each
   op give:
   - typed domains (widths, shapes, granularity);
   - the exact operation order;
   - the rounding mode (`roundeven` versus round-half-away);
   - clamp bounds and the scale floor;
   - zero points;
   - poison propagation.

   Use `Spec/Dequant.lean`'s structure: scalar contract first, then the tensor contract over
   coordinates.
2. **D3: `Spec/FrontendIntent.lean`.** The intended meaning of three capture repairs, each citing
   the PyTorch op it reflects:
   - boolean to float: `True → 1.0`, `False → 0.0` (MLIR's signed `i1` cast would give `-1.0`);
   - float to boolean: `x != 0`;
   - an over-rank matmul from `aten.linear`: a batched matmul over the leading dimensions.

   Express each as a spec transformer on the source program (claim type `policy "<name>"`). Its
   meaning is the intended computation, independent of what MLIR would do with the literal ops.
3. **D5: `Spec/Policy/W8A8.lean`.** The spec transformer `P` for the W8A8 policy, as TorchAO's
   recipe defines it:
   - **Eligibility:** which contractions are quantized, stated from the recipe.
   - **Activation quantization:** granularity (per row over the contraction dimension), the scale
     with its floor, multiply by reciprocal versus divide (whatever TorchAO does), rounding mode,
     clamp bounds, and conversion to `i8`.
   - **Weights:** captured `i8` values with per-output-channel scales. State the axis.
   - **Accumulation:** `i8 × i8 → i32`.
   - **Requantization:** the multiply order of the accumulator and the two scales, exactly as
     TorchAO computes it.
   - **NaN and infinity inputs:** state what the policy means (for example, outside the domain).
     A3 proves accordingly.
   - The side-condition definitions A3 needs, such as the contraction length bound for exact
     `i32` accumulation.
4. **Oracle tests.** For each spec, a test that evaluates the Lean spec, through the existing
   differential adapter or an independent NumPy transcription you write from the spec, against
   real PyTorch/TorchAO on finite sample inputs. Include the edge cases: zero rows (the scale
   floor), ties at .5, values at the ±127 clamp, and negative zero.

## Acceptance

- `lake build` is clean, there are no axioms, and the guard passes.
- Every decision has a source citation in `NOTES.md`.
- Statements go to A0 for Jeremy's review by D3 (TorchAO, intent) and D5 (W8A8). A0 pins them.
- After pinning, changes go only through A0 with a written reason.

## Don't

- Don't write checkers or proofs about Merlin passes. That's A3, A4 and A5.
- Don't edit `Sem/`. Request changes through A0.
- Don't break the barrier, even to "check" something. Ask A0 a question instead.
