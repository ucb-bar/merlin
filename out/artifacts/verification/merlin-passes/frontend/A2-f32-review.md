# A2 f32 boolean repair: reviewed policy and model

**Status: the user approved the documented f32 policy and direct IEEE instance on
2026-10-21.** The exact reply was “Approve the documented f32 policy and direct IEEE
instance”. [The scoped approval record](f32-policy-approval.json) binds the primitive,
source interpretation, eligibility/materialization and comparison definitions.
This is semantic approval, not a claim of human review of every implementation line.
It excludes other A2 primitives, capture correctness, code generation, other float
formats, frozen semantics changes and SPEC_LOCK admission. A2's unchanged file has
its historical pending-review header; this external record approves only the f32
subset described here. No approval is inferred from a proof or passing test.

## Intended behavior

For initialized IEEE binary32 bits `b`, return i1 `(b mod 2^31 != 0)`.
Both signed zeros are false; every other encoding is true, including all quiet and
signaling NaNs of either sign and any payload. Poison stays poison. Tensor results
apply this rule per coordinate, preserving shape. Values and poison are observed;
floating exception flags, traps and signaling-NaN exception effects are unmodeled.

| Input | Representative bits | Intended i1 |
| --- | --- | --- |
| +0, -0 | `00000000`, `80000000` | 0, 0 |
| +0.5, -0.5 | `3f000000`, `bf000000` | 1, 1 |
| +1, -1 | `3f800000`, `bf800000` | 1, 1 |
| Smallest positive/negative subnormal | `00000001`, `80000001` | 1, 1 |
| +infinity, -infinity | `7f800000`, `ff800000` | 1, 1 |
| Quiet NaN, signaling NaN | `7fc00000`, `7f800001` | 1, 1 |
| Poison | no initialized bits | poison |

The independent basis is the bool specialization in
[PyTorch v2.10.0 TypeCast.h](https://github.com/pytorch/pytorch/blob/v2.10.0/c10/util/TypeCast.h),
which applies C++ `static_cast<bool>` to a real source. The
[C++ Boolean conversion rule](https://eel.is/c++draft/conv.bool) sends zero to false
and other arithmetic values to true. Poison and coordinate-wise lifting are
explicit model extensions. They do not assert that PyTorch exposes LLVM poison.
Pinned PyTorch 2.10.0+cpu execution samples support this reading; they do not prove
the universal bit-pattern rule.

## Capture convention and intentional differences

The approved convention treats every eligible captured f32-to-i1 `arith.fptosi`
**or** `arith.fptoui` at this repair checkpoint as frontend bool conversion.
Syntax alone would not establish the intent of arbitrary handwritten numeric IR.
Eligibility is recomputed from the exported source AST, types and attributes.
Nondefault/unsupported casts are refused, not omitted.

Original casts are truncating numeric conversions with representability conditions.
For example, ±0.5 can truncate to zero but produce true under bool intent. Signed
+1, infinities and NaNs can poison the original cast but have initialized true
results under this policy. The claim therefore repairs both defined values and
poison-producing conversions. It is not equivalence/refinement of the original
numeric-cast program and gives no permission to reinterpret wider integer casts.

## Approved source-program interpretation

`FrontendBoolSimulation.policyDenote before ctx f args` inventories the **original
before AST**, then runs the existing Lean MLIR evaluator with
`FrontendBoolPolicy.sem`. Each statically eligible scalar f32-to-i1 cast directly
calls A2's independent `f32ToBool` primitive. All surrounding operations use frozen
`opSem`; recursive regions and internal calls use the same source policy.
This definition neither constructs nor reads the target/comparison program.
Ill-typed runtime operands stay unsupported.

`FrontendBoolSimulation.check_sound_ieee` proves exact outcome equality of that
source interpretation and the actual accepted after program. It derives operand
lookup, site occurrence, freshness and replacement-use facts from the checker.
The proof covers whole blocks, regions, tensor bodies, every function, internal
calls, external declarations and call fuel. The independent structural check also
requires whole-program correspondence with `FrontendBoolProgram.intent?`, which
materializes a fresh positive-f32-zero plus no-fast-math `cmpf une` at every site
and preserves unrelated IR. No caller supplies an intended module as a proof fact.

The old C1 module filter is replaced in the integrated path by the complete direct
source-policy execution proof. This extends execution correspondence to internal
calls that C1 conservatively refuses; it retains whole-boundary structural checks.
It does not weaken a stuck-sensitive claim to get acceptance.

## Exact floating-point premise and its discharge

The general theorem assumes precisely:

```lean
UneCorrect M := forall x : BitVec 32,
  M.cmp uneKey f32 x 0 = (if x.toNat % 2^31 = 0 then false else true)
```

`uneKey` is exactly `arith.cmpf` with modifiers
`fm1:0;rm4:none;pred13:attr 13 : i64`; the right operand has positive-zero bits.
This law quantifies initialized encodings. Frozen eager poison propagation proves
poison behavior separately, without assuming any comparison law on poison.
An arbitrary frozen `FloatModel` cannot imply the law (`alwaysFalse_not_une`).

The approved `FrontendF32Comparison.model(base)` overrides only this key at f32:

```lean
une x y = !(UnpackedFloat.beq (unpack binary32 x) (unpack binary32 y))
```

It uses Lean 4.35's existing IEEE unpack/comparison definitions; all other model
entries delegate to `base`. `une_zero` and `model_correct` prove the law for all
2^32 inputs, including all NaNs, by exponent/significand/sign cases. No axiom,
enumeration assumption or native decision procedure is used. `check_sound_ieee`
has **no unproved `UneCorrect` premise**. Standard logical dependencies are only
`propext`, `Classical.choice`, `Quot.sound`.

The compiler's [pinned ArithOps.cpp](https://github.com/llvm/llvm-project/blob/a47bddcce/mlir/lib/Dialect/Arith/IR/ArithOps.cpp)
defines UNE as comparison other than `APFloat::cmpEqual`, including unordered.
This supports the interpretation; it is not a formal proof of compiler codegen.

### Historical host adapter: excluded bridge

`MerlinDifferential.hostModel` widens via `(Float32.ofBits x).toFloat` before
comparing. Lean 4.35's `Float32.toFloat` is an opaque extern with no logical
equation (`Init/Data/Float/Float32.lean:556`). The exact unproved alternative is:

```lean
forall x : FBits f32,
  hostModel.cmp uneKey f32 x 0 = FrontendF32Comparison.une x 0
```

A reviewed logical widening model/proof would be needed if that adapter were
required. It is unused by the approved direct-instance theorem, is not asserted
as an axiom, and is not established by native samples. No frozen-file/toolchain
change is needed for the approved interpretation.

## Composition and execution qualification

`FrontendBoolChain.chain_after_source_policy` composes acceptance and proved
ordinary suffixes into `RefinesC(final, policyDenote(exact repair.before))` on
explicit context/input domains, with `UneCorrect` discharged by the direct model.
A prefix does not transport policy to the original capture; multiple repair
checkpoints are explicitly unsupported by the composer. The ordinary chain's
original-source refinement verdict remains abstracted across the repair.

Exact outcome equality includes equality of stuck outcomes. Separate
`FrontendBoolQualified.support_sound` proves strict refinement/policy support on
checked source-signature domains. `defined_sound` proves identical successful,
nonpoison results on typed nonpoison inputs when its checker accepts. A source
numeric-cast support receipt is never substituted for policy support. These
qualifications are replayed and cannot be forged by changing receipt status.

See [the final report](fix-bool-fptosi.md) for actual SmolVLA refusal, positive
synthetic qualifications, tests, versions, bindings and remaining lock admission.
The approved scope remains f32 only. f16/bf16/f64 proofs are unsupported.
