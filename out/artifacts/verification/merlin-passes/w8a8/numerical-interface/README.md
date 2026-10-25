# Proposed additive bounded-product interface

Later implementation: the [rotary milestone](../rotary-domain/README.md) supplies proved
rounded binary32 product bounds, interval-derived casts, actual single-write geometry and
scale/zero-derived source byte-weight bounds. It does not admit the proposal below as a
shared law. The [active eight-day goal](../../progress/planner/eight-day-paper/README.md)
prioritizes modular pass correspondence; whole-model attention/1533 remains open.

2026-10-22. Interface proposal for the planner and frontend worker, not an admitted law,
implemented checker or approved domain. Existing `W8A8FiniteLaws.Laws` and
`W8A8DynamicLaws.Laws` meanings remain unchanged.

## Actual source dependency at retained result 1256

The preserved authoritative `0_before.json` has:

```text
%979  : tensor<1x8x2048xf32>   embedding result from original input %0
%1230 : tensor<1x8x1xf32>      mean-square + epsilon
%1236 : tensor<1x8x1xf32>      rsqrt(%1230)
%1241 : tensor<1x8x2048xf32>   %979 * broadcast(%1236)
%1247 : tensor<1x8x2048xf32>   broadcast(original gain input %1) * %1241
%1256 : tensor<1x8x2048xi8>    quantize_per_tensor(%1247, scale, zero)
```

The two multiplication bodies use exact attribute-free `arith.mulf` (`fastmath = none`).
The quantizer scale is literal f32 bits `0x3c7d4494`; zero point is i64 zero; clamp bounds are
-128 and 127. These are observations of this development capture, not production eligibility
criteria or selected-model coverage. The source reference reciprocal is f64 followed by f32
narrowing. The generated dynamic recipe uses its separate f32 reciprocal and -127..127 range.

The current entry domain supplies finite f32 values but no magnitudes for the embedding or
gain inputs. Intermediate arbitrary f32 arithmetic forgets finiteness. No source fact says
that RMS-normalization or gain multiplication cannot overflow.

## Minimal reusable scalar interface to consider

Proposed definitions below use exact unsigned magnitude-word ordering on finite f32 bits.
For finite values that order agrees with nonnegative absolute magnitude; a concrete numerical
model still needs to prove the operational law. The snippets are types for discussion and
have not been added to any shared module.

```lean
-- Proposed owned namespace: W8A8BoundedProducts, additive file only.
def AbsLE (bound : Nat) (x : FBits .f32) : Prop :=
  W8A8FiniteLaws.Finite x ∧ W8A8ByteFloat.magnitudeWord x ≤ bound

def MulWithin (M : FloatModel) (left right output : Nat) : Prop :=
  ∀ x y : FBits .f32, AbsLE left x → AbsLE right y →
    AbsLE output
      (M.op (Spec.Dequant.defaultKey "arith.mulf") .f32 [x, y])
```

An adapter can derive output finiteness from `MulWithin` and checked operand facts. It cannot
invent `MulWithin`, input bounds or a universal instance from finite inputs. A concrete instance
must prove rounding and overflow safety for its chosen bounds; a real-number product inequality
alone is not automatically a floating-point rounding theorem. Output bounds should be checked
against the largest finite magnitude word `0x7f7fffff`.

Uniform tensor versions quantify over every in-bounds coordinate and use the actual source
maps. At %1241 they require a bound on %979 and the actual rsqrt result %1236. At %1247 they
require a bound on gain input %1 and %1241. The rsqrt bound additionally needs a justified
positive lower bound on the exact computed %1230, with reduction order, division, epsilon,
exceptional values and rounding accounted for. A better relational RMSnorm bound could avoid
an overly conservative independent product bound, but needs its own proof.

## Fixed existing law interfaces

- `W8A8FiniteLaws.Laws base` fields: `widen`, `reciprocal`, `narrow`, `integer`, `multiply`,
  `round`, `add`. In particular `multiply` concludes **non-NaN**, not finite, for finite operands.
- `W8A8DynamicLaws.Laws base` embeds `source` and adds `absolute`, `scaleDivide`, `reciprocal32`.
- Neither includes arbitrary finite-product closure, positive rsqrt behavior, RMSnorm bounds
  or magnitude bounds on input weights.
- `W8A8ByteModel.extend base` overrides selected byte/conversion/min/max behavior; it is not
  definitionally an arbitrary frontend model. Existing `mulf_eq` explicitly transports this
  attribute-free multiplication back to `base`; use proved operation-specific adapters.
- The frontend owns `AbsReductionCorrect`, `ZeroPointElisionCorrect` and its row/call bridge.
  Do not identify those laws with these source/dynamic contracts or add implicit instances.

## Decisions before implementation

1. Decide whether to prove a stronger relational property for the actual RMSnorm path or use
   explicit floating magnitude facts. Check whether finite-entry input scope is actually enough.
2. Obtain legitimate input facts: constants/byte-bound archive values or an explicitly declared,
   inhabited bounded input domain. The current domain cannot silently be narrowed.
3. Prove corresponding concrete-model multiplication/rsqrt/reduction laws; merely adding a
   hypothesized adapter does not finish concrete numerical verification.
4. If considering non-NaN quantizer input rather than finite input, treat that as a new source
   domain/primitive theorem. Current multiplication law does not justify infinity × reciprocal
   automatically, especially without a proved nonzero reciprocal. No such change is made here.

Owner: W8A8 worker for additive adapters. Planner coordinates numerical interfaces; independent
review remains required for contracts/domains. No frozen Float/Spec/lock or frontend file changes.

## New evidence: finiteness is not automatic

`rmsnorm-domain-diagnosis-20261022T193312Z/` retains a diagnostic slice containing the actual
21 normalization/gain producer operations from the original authoritative export. Only the
embedding result and gain become explicit inputs and three intermediate tensors are returned.
Operation bodies, reduction order, constants, maps and attributes are preserved. The native
probe observes the first element; it is not a new selected-model capture or whole-pass receipt.

Every 2048-element input row is `[1, 0, ..., 0]`. Native MLIR gives inverse and normalized value
bits `0x42333195` (44.79841995239258). All probe gains are finite:

| Gain bits | Gain | Scaled output bits |
|---|---|---|
| `0x3f800000` | +1 | `0x42333195` (finite control) |
| `0x7f7fffff` | largest positive finite f32 | `0x7f800000` (+infinity) |
| `0xff7fffff` | largest negative finite-magnitude f32 | `0xff800000` (-infinity) |

Thus a universal claim that finite embedding/gain inputs make this intermediate finite is
false for this actual fragment. This is a counterexample to a proposed finiteness implication,
**not** a compiler-pass correctness counterexample. It does not establish the rest of the
model's execution. It changes the next action: a finite-output proof needs legitimate stronger
facts/domain, or source reasoning that correctly handles the resulting nonfinite cases.

The separate Lean fixture `merlin/tests/fixtures/lean_w8a8_finite_products.lean` defines a
deliberately non-IEEE abstract model satisfying all existing seven source and three dynamic
laws while multiplication returns infinity. `finite_product_not_entailed` proves that those
laws cannot entail universal finite-product closure, even on +1/+1. All three fixture theorems
are kernel checked and depend on no axioms. This negative model is not a proposed numerical
implementation or admitted assumption.

Two regressions in `merlin/tests/ir/test_lean_w8a8_rmsnorm_domain.py` pass: the Lean law-interface
countermodel and the native actual-slice three-gain observations. The captured MLIR fixture is
retained at `merlin/tests/fixtures/w8a8_rmsnorm_domain.mlir`. Results and hashes are in
`rmsnorm-domain-qualification-v1.json` and `rmsnorm-domain-qualified-v1.{xml,log}`. Historical
six-source and 752-file runtime hashes remained unchanged. No source contract was narrowed,
no floating law was changed, and no original JSON failure was declared fixed.
