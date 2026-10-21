# A2 f32 boolean repair: human review packet

**Status: proposed; awaiting human review. No approval is recorded.**
Scope: `policy "frontend-float-to-bool"` at one recorded
`prepare-fix-bool-fptosi` checkpoint. No f16, bf16 or f64 extension is proposed.
The independent A2 primitive remains unchanged, including its pending-review header.
Its SHA-256 is `6f84e09fbf40206461924ff8dc72ce8842bbce9e13abaecb0b0d87e46c8f8e32`.

## Behavior to review

For an initialized IEEE binary32 encoding `b`, produce an i1 with value
`(b mod 2^31 != 0)`. Both signed zeros are false; all other encodings are true.
This includes subnormals, fractions, infinities, and quiet and signaling NaNs of
either sign and any payload. No NaN canonicalization is required to determine
this Boolean. In the Lean extension, an input poison produces output poison.
Tensor results apply this rule independently at each coordinate, preserving shape.
The observation is values and poison; floating environment flags, traps and
other signaling-NaN exception effects are not modeled or claimed.

| Input | Representative f32 bits | Intended i1 |
| --- | --- | --- |
| +0, -0 | `00000000`, `80000000` | 0, 0 |
| +0.5, -0.5 | `3f000000`, `bf000000` | 1, 1 |
| +1, -1 | `3f800000`, `bf800000` | 1, 1 |
| Smallest positive/negative subnormal | `00000001`, `80000001` | 1, 1 |
| +infinity, -infinity | `7f800000`, `ff800000` | 1, 1 |
| Quiet NaN, signaling NaN | `7fc00000`, `7f800001` | 1, 1 |
| Poison | no initialized bit pattern | poison |

The independent source basis is the bool specialization in
[PyTorch v2.10.0 TypeCast.h](https://github.com/pytorch/pytorch/blob/v2.10.0/c10/util/TypeCast.h),
which applies C++ `static_cast<bool>` to a real source. The
[C++ Boolean conversion rule](https://eel.is/c++draft/conv.bool) maps zero to false
and other arithmetic values to true. Poison and tensor-coordinate behavior above
are explicit modeling extensions, not claims that PyTorch exposes LLVM poison.
The pinned PyTorch 2.10.0+cpu tests are execution evidence supporting this reading;
they are not the proof of the universal bit-pattern rule.

## Intentional differences from the captured program

The source is sometimes captured as `arith.fptosi` or `arith.fptoui` to i1. These
are truncating numeric conversions with representability requirements. The repair
implements the frontend bool contract instead. For example, +0.5 and -0.5 can
truncate to zero in the original cast and become true under this policy. Signed
+1, infinities, and NaNs can produce poison in the captured signed cast; the
intended initialized results are true. This is both a defined-value change and a
repair of poison-producing conversions. It is not equivalence to the original
MLIR or permission to reinterpret other numeric conversions.

The repair applies to both signed and unsigned **f32-to-i1** capture spellings.
This requires review of the capture convention: these sites must represent
frontend boolean conversions. Syntax alone does not prove that an arbitrary
user-written f32-to-i1 numeric conversion had this intent.

## Exact program policy to review

`Rules/FrontendBoolProgram.intent?` is the proposed materialization of this policy
on the authoritative exported before AST. At every eligible f32-to-i1 signed or
unsigned cast, it emits a fresh f32 +0 constant and `arith.cmpf une` with no
fast-math flags, reuses the cast result ID, and preserves the other operations,
regions, operands, types and attributes. It refuses unsupported sites instead of
omitting them. `check` compares the entire materialization with the authoritative
after AST and invokes the existing whole-module evaluator checker.

Approval of A2's scalar primitive alone is **not** approval of this program policy.
A2 currently describes itself as scalar/coordinate primitives, not a completed
source-program transformer. The reviewed policy needs to name this materialization,
the capture convention above, and its checkpoint. The proved whole-program
reference is this materialized comparison program. There is no separate theorem
identifying a source-level PyTorch program interpreter with that reference.

## Smallest comparison interpretation proposal

`UneCorrect M` currently assumes exactly:

```
forall x : BitVec 32,
  M.cmp uneKey f32 x 0 = (if x.toNat % 2^31 = 0 then false else true)
```

Here `uneKey` is exactly `arith.cmpf` with modifiers
`fm1:0;rm4:none;pred13:attr 13 : i64`. The second operand is positive zero.
The assumption covers initialized bit patterns only. Poison propagation is
proved separately from `Sem.Ops.opSem`'s eager poison handling, without a float law.

The frozen `FloatModel` record intentionally has arbitrary comparison functions;
`alwaysFalse_not_une` is a counterexample to proving this for every such record.
There is an existing concrete executable adapter, `MerlinDifferential.hostModel`.
For f32 it widens with `(Float32.ofBits bits).toFloat`, then compares host Floats.
In Lean 4.35.0, `Float32.toFloat` is an opaque extern without a logical equation
(`Init/Data/Float/Float32.lean:556`). Float unpacking, NaN tests and comparison do
have logical definitions; the precise missing bridge is the widening conversion.
No universal `UneCorrect hostModel` proof has been obtained from this model.

Proposed instance `Rules/FrontendF32Comparison.model(base)` overrides **only** the
exact `uneKey` at f32, using the existing Lean 4.35 IEEE definitions:

```
une x y = !(UnpackedFloat.beq (unpack binary32 x) (unpack binary32 y))
```

Everything else uses `base`. `une_zero` and `model_correct` prove the required
law for all 2^32 encodings, by unpacking exponent, significand and sign. There is
no enumeration assumption or native evaluation axiom. `model_site`,
`accepted_site_exec` and `model_tensor` discharge the comparison premise for the
actual evaluator, the accepted generated site with arbitrary continuation, and
the poison-aware tensor extension, respectively. Audits report only `propext`,
`Classical.choice`, and `Quot.sound`.

This proposal chooses an explicit instance of the existing record; it changes no
frozen semantic file, `Differential.hostModel`, A2 primitive or `SPEC_LOCK`.
It is **not activated as the approved pipeline interpretation**. Review must
accept its interpretation of MLIR predicate 13. The compiler's
[pinned ArithOps.cpp](https://github.com/llvm/llvm-project/blob/a47bddcce/mlir/lib/Dialect/Arith/IR/ArithOps.cpp)
implements `UNE` as comparison result other than `APFloat::cmpEqual`, including
unordered comparisons. This is source justification, not a Lean proof about the
compiler or machine code.

If the existing host adapter must be the approved reference, the remaining exact
bridge is:

```
forall x : FBits f32,
  MerlinDifferential.hostModel.cmp uneKey f32 x 0 = FrontendF32Comparison.une x 0
```

A justified logical model of `Float32.toFloat` and a widening proof would be
needed to establish it. This proposition is not introduced as an axiom. Native
samples cannot discharge it. No approval for a frozen or toolchain edit is inferred.

## Composition and assumptions

`FrontendBoolChain.chain_after_repair_refinesC` composes acceptance with proved
ordinary suffix boundaries, under explicit context and input domains. Its
conclusion is conditional refinement of the final program against
`Intent(exact repair.before)`. An ordinary prefix is recorded, but does not
transport this policy backwards to `Intent(original capture)`. Multiple repair
checkpoints are refused by the integrated composer.

`chain_after_repair_strict` additionally requires that the intent program is
not stuck on that same domain. Successful, nonpoison final execution additionally
needs intent definedness. Support/definedness of the captured source is not a
substitute. Full SmolVLA support/definedness is not established. The ordinary
chain verdict remains abstracted across a repair. A nested verified policy
composition record describes this conditional theorem, not human approval or
original-source refinement.

Trust also includes authoritative MLIR export/round trip, the checked SSA/type
representation, existing evaluator semantics and rulebook, Lean's kernel and
standard logical axioms, and eventual correctness of downstream code generation.
The local site theorem states source lookup, occurrence, result-type and fresh-ID
premises explicitly; there is no global proof of a separately defined source
macro interpreter with those premises erased. The accepted-program theorem is
against the comparison-based materialization described above.

## Review decisions left open

- Approve or reject the f32 bit-level bool contract and its poison extension.
- Approve or reject treating both eligible capture cast spellings as bool intent.
- Approve or revise the exact `FrontendBoolProgram.intent?` materialization as the
  program repair policy at this checkpoint.
- Approve the proposed direct IEEE comparison instance, or request a reviewed
  logical widening model and proof for the existing host adapter.

These decisions are pending. No reviewer name, approval token or review date is
filled in by this implementation.
