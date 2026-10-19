# A4 — Quantization lowerings

**Branch:** `pv/quant-lowerings`. **Mission:** verify the quantization-related preparation
boundaries other than W8A8 itself: the TorchAO affine lowering, the quant-inner tag propagation and
lift, and any `quant_ext` variants the model uses.

## Read first

- The dequantization template (see A3's brief), which is the pattern for the TorchAO lowering.
- Design §4.6 (argument and result maps) for the qinner lift.
- `Spec/Annotations.lean`: which attributes are semantics-free (`prov.*`) or ABI-only.

## You own

- `MerlinVerify/Rules/TorchAO*.lean`, `Rules/QInner*.lean`, and additions to the dequant rules for
  new `quant_ext` variants.
- `merlin/tests/ir/test_lean_torchao*.py`, `test_lean_qinner*.py`.
- `out/artifacts/verification/merlin-passes/quant-lowerings/`.

## Work, in priority order

Drop any item A1's census shows doesn't fire on the chosen model (A0 confirms at Gate A).

1. **`prepare-propagate-quant-inner`** (`_propagate_quant_inner`,
   `src/merlin/runtime/dispatch_runtime.py:1210`, 33 lines). Find out exactly what it changes.
   - If it only adds or changes `prov.*` or other semantics-free attributes, show that the existing
     checker accepts the boundary as unchanged semantics, and add a test.
   - If it changes anything the semantics reads, scope a rule and tell A0.

   About half a day.
2. **`prepare-qinner-lift`** (`qinner.lift`, `src/merlin/llvmlower/qinner.py`). It lifts each tagged
   `tensor.empty` to a trailing `@forward` argument.
   - **The problem.** Under the current semantics the source reads an uninitialized tensor, which
     is poison, so the source is undefined and any refinement claim is vacuous.
   - **The fix.** It depends on decision 6 (plan §7). The tagged empty means "the runtime binds
     this from the quantized subclass's inner tensor".
     1. Work with A2 (through A0) on a small annotation spec that gives tagged empties that meaning.
     2. Prove the lift is an `equivArgs` boundary with the explicit argument map: the new trailing
        argument corresponds to the bound inner tensor.
     3. Check that `c_runtime.generate` derives the same list. A test can compare the two lists.

   2–3 days.
3. **`prepare-torchao-affine`** (`lower_torchao_affine_quant`, `src/merlin/llvmlower/torchao_affine.py`).
   This replaces opaque TorchAO calls with arithmetic. Prove it against A2's `Spec/TorchAO.lean`
   (pinned around D3), following the dequant layering:
   1. scalar body proof for every float model and every input bit pattern, including poison;
   2. tensor and layout proof (maps, granularity, axis, empty tensors);
   3. a source rule so the opaque op has the spec's meaning (the way `Sem/Ops/Dequant.lean`
      dispatches the dequant contract);
   4. the site rule;
   5. definedness on initialized inputs;
   6. a mutation matrix (rounding mode, clamp, scale floor, zero point, axis);
   7. a PyTorch-oracle comparison.

   Until the spec is pinned, prepare the layout and site-matching parts. 2–4 days.
4. **`prepare-lower-quant-ext` variants.** The existing proof covers signed-`i8` storage, `i32` zero
   points and `f32` scale/result, for both granularities. Extend it only to variants A1's census
   shows on the chosen model.

## Don't

- Don't change `qinner.py`, `torchao_affine.py` or `dispatch_runtime.py`. Report bugs through
  findings and A0.
- Don't make poison or uninitialized reads "defined" by changing `Sem/`. The meaning comes from a
  reviewed spec, not from a semantics edit.
