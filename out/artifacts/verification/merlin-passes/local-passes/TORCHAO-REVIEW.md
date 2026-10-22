# Provisional affine contract: review questions

**Status: not independently reviewed or admitted.** This is an implementation-aware proposal,
not a replacement for barrier-authored A2 specifications. Frozen files and locks are unchanged.

## Reference and intended operation order

Use the actual capture receipt to select the reference. The Gemmini branch's current static
grouped-16 recipes are not the dynamic per-token recipe below. Its historical r53b guide separately
names TorchAO 0.15.0/0.16.0 and a historical recipe. The intake pins all of those documents without
equating their selections.

For the provisional **f32 dynamic per-token, symmetric, −127…127, epsilon 1e-5** case,
TorchAO's versioned primitive source specifies:

1. Reduce each nonempty row by `amin` and `amax`, include zero in both bounds, then form
   `max(-min_value, max_value)`.
2. Divide by f32 127; clamp the scale below by f32 1e-5 (`0x3727c5ac`). Zero point is zero.
3. Form f32 `1 / scale`, multiply each input by it, round half to even, **add zero point**,
   clamp to −127…127 and convert to int8.

The lowering instead reduces an abs/max fold and omits the +zero. The Lean proposal exposes these
as two separate unproved concrete-model obligations. It does not invent a floating-point identity.
The scalar evaluator proof establishes the exact implemented operation order for every parametric
float model, not the missing reference equalities.

## Observed domain differences requiring a decision

### Nonfinite results

The pinned CPU oracles produce NaN scale for a NaN-containing row, infinity scale for an
infinity-containing row, and integer zeros after quantizing the tested nonfinite rows. MLIR
`arith.fptosi` of a NaN produces poison. The installed PyTorch 2.10 `Tensor.to` documentation
explicitly describes non-fitting float-to-integer conversion as undefined/platform-dependent.
Thus those CPU zeros are observations, not a universal defined-zero reference contract or a proved
compiler counterexample. The current Lean conversion extension explicitly uses MLIR's conversion
contract; its relationship to the reviewed Torch reference still needs to be stated and proved.

A reviewed contract must establish a local finite/in-range conversion domain, specify the
undefined/out-of-domain cases, or deliberately require behavior needing a separate compiler repair.
Matching argument types does not establish that domain. If the contract requires finite inputs
and scales, explain how call-site values satisfy it; if it uses the weaker domain of a finite
clamped result, state that domain precisely. There is no authorized compiler change in this work.

### Empty tensors

For input shape `[0,4]`, both pinned TorchAO versions reach the empty-zero-point path, replace the
zero point with `None`, and raise `TypeError` at tensor + `None`. The lowered native MLIR can return
empty outputs. The candidate checker now refuses empty extents. This refusal is explicit; the
empty case remains in qualification and is not counted as numerical agreement.

### Captured constants and scales

The opaque call signature loses mapping type, quantization bounds, epsilon, block-size constants
and zero-point information. A module scheme string is not independently sufficient evidence that
the producer used these particular values. Bind the capture receipt, actual recipe bytes/hash,
capture versions, stored scales and any zero-point-dropping evidence before admitting a contract.

## Review decisions requested

- Select the actual capture and reference version/recipe. Confirm whether this dynamic lowering
  occurs at all on that selected static Gemmini path.
- Approve the exact arithmetic order and any input/result-domain preconditions; resolve nonfinite
  conversion behavior explicitly.
- Define the intended empty-input behavior. The current candidate refuses it.
- Review the concrete IEEE model and proofs for reduction replacement and zero-point elision,
  including NaN, infinity and signed-zero behavior within the admitted domain.
- Independently author/admit the specification. The verifier author has read the implementing pass
  and cannot provide independent specification approval.

The captured-call → tensor-reference bridge is still engineering work. The current receipts prove
only materialized-program correspondence plus the stated scalar component, and say so explicitly.
