# prepare-scalar-squares: provisional review packet

The complete SmolVLA and TinyLlama boundaries match the source-derived multiplication
reference at all 111 rewritten sites. Lean proves this complete-output correspondence,
site coverage, correspondence to actual operation pairs, and a local semantic theorem
conditional on an explicit numerical law. **Original-source bitwise equivalence is not
established. Native counterexamples refute that law for the tested implementation.**

Status: `abstracted`, numerical law `provisional_not_discharged`. No axiom, reviewed Spec
entry, theorem pin or SPEC_LOCK change was introduced. This packet was written with access
to the implementation; it is not an independently authored or human-reviewed specification.

## Exact proposed numerical contract

The rule is scalar only. The result and both source operands have the same type, either
IEEE binary32 (`f32`) or binary64 (`f64`). The exponent is a lexically available
`arith.constant` with the exact positive-two encoding:

| Format | Exponent bits | Target |
| --- | --- | --- |
| f32 | `0x40000000` | one `arith.mulf(x, x) : f32` |
| f64 | `0x4000000000000000` | one `arith.mulf(x, x) : f64` |

For a chosen `FloatModel M`, format `k` and preserved attributes `a`, the provisional law is
exactly `MerlinVerify.Rules.ScalarSquareContract.Law M k a`:

```text
for every x : BitVec(width(k)),
  M.op(key(math.powf, a), k, [x, positiveTwo(k)])
    = M.op(key(arith.mulf, a), k, [x, x]).
```

This is equality of returned **bits**, not equality of real numbers, approximate equality,
NaN-class equivalence, or a bound on error. It requires agreement on every input encoding,
including negative finite values, both zeros, subnormals, infinities, quiet/signaling NaNs
and every NaN sign/payload. Underflow, overflow, NaN quieting and payload selection must
agree if this law is to be instantiated. There is no implicit finite-only domain.

`FloatModel` is uninterpreted and deterministic. In particular, Lean does not implement
IEEE rounding or a libm accuracy guarantee here. A real-arithmetic identity `x^2 = x*x`
does not prove the proposed floating-point law. A correctly rounded power implementation
and correctly rounded multiplication in the same fixed rounding mode would agree on the
finite real square, but that premise, exceptional behavior and backend fidelity would
still need justification. No such implementation premise is currently discharged.

If the target multiplication is interpreted as IEEE round-to-nearest, ties-to-even with
gradual underflow, its intended exceptional-value policy is:

| Base | Intended multiplication result |
| --- | --- |
| `+0` or `-0` | `+0` |
| `+infinity` or `-infinity` | `+infinity` |
| Finite, nonzero | The exact real square rounded once to the result format; a subnormal, `+0`, or `+infinity` can result |
| Quiet or signaling NaN | NaN; exact sign/payload/quieting follows the chosen implementation |

The table explains an intended IEEE instantiation, not a theorem about arbitrary
`FloatModel`s. The proposed law additionally requires the source power to produce the
same bits. Both native paths observed matching zero/infinity behavior and NaN results on
the selected edge cases; sampling does not establish all exceptional encodings.

Rounding mode is not an ambient parameter of this Lean model. Runtime rounding, subnormal
handling and concrete arithmetic implementations belong to the interpretation of `M`.
Explicit rounding-mode attributes are outside this checker’s admitted rule. MLIR’s default
round-to-nearest, ties-to-even for folding/canonicalization is not a runtime guarantee;
the runtime behavior without an explicit mode belongs to the backend.
[MLIR Math documentation](https://mlir.llvm.org/docs/Dialects/MathOps/)

`errno`, floating exception status, traps and observation of a changing rounding environment
are outside the Lean contract. The C harness records `errno` and floating exceptions
separately, and differences were observed. These observations must not be promoted to an
effects-preservation claim. LLVM’s unconstrained power intrinsic is specified using libm
return values without traps or `errno`, with special NaN allowances; this is not a
correct-rounding or exact-NaN-payload guarantee.
[LLVM power semantics](https://llvm.org/docs/LangRef.html#llvm-pow-intrinsic)

## Fast-math contract

The checker admits absent flags, explicit `none`, or **only** `contract`. It preserves the
actual attribute list exactly, including permitted semantics-free annotations. The flag
mask is part of each floating-operation key; the `contract` obligation is distinct from
the `none` obligation. The multiplication retains the original result binding and uses
the source base operand twice. No certificate can supply a numerical-law proof.

`contract` permits contraction; it does not imply finite inputs, negligible signed zeros,
approximate functions or reassociation. This packet checks the isolated boundary and local
operation result. It does not prove a later fused multiply/add execution agrees with a
separately rounded multiplication. The actual 111 captured rewrites all have `none`.
[LLVM fast-math flags](https://llvm.org/docs/LangRef.html#fast-math-flags)

`nnan`, `ninf`, `nsz`, `afn`, `reassoc`, `arcp`, `fast`, combinations containing these,
unknown semantic attributes and unsupported flag encodings are refused. In particular,
`nnan`/`ninf` introduce poison conditions, `nsz` relaxes zero signs, and `afn` permits
approximation. Preserving such an attribute would not by itself prove the rewrite valid.
The separate source census includes literal-two candidates with unsupported flags, so
they cannot disappear from coverage and produce an accepted identity check.
[LLVM flag semantics](https://llvm.org/docs/LangRef.html#fast-math-flags)

Poison is distinct from NaN in the shared semantics. `poison_sound` proves that both actual
operations propagate a poison base, independently of the numerical law. The scalar law
theorem addresses non-poison floating values of the required format. This is not a proof
that a surrounding captured program always supplies such values or executes successfully.

## What Lean proves

Source files, relative to the repository:

- `merlin/lean/MerlinVerify/Rules/ScalarSquareContract.lean`
- `merlin/lean/MerlinVerify/Check/ScalarSquare.lean`
- `merlin/lean/MerlinVerify/Driver/ScalarSquare.lean`

| Result | Exact scope |
| --- | --- |
| `check_sound` | Accepted complete output equals the source-derived reference; every recorded site has local equality conditional on its numerical law and typed operand values |
| `complete_output_sound` | The complete output and reference have identical denotations under **any** operation semantics, including errors; this follows from full AST equality |
| `coverage` | Recorded site paths equal the independent recursive source-candidate census, in order |
| `accepted_pairs`, `pairCheck_sound`, `site_source` | Recorded sources and targets occur at the claimed positions in the actual modules, with source eligibility rechecked in the actual lexical scope |
| `Site.scalar_sound`, `ScalarSquareContract.scalar_sound` | Equality of the actual scalar operation outcomes, for every floating bit pattern, conditional on `Law` |
| `poison_sound` | Equal poison propagation in the shared operation semantics |
| `law_nonempty` | The law has model witnesses, avoiding an impossible universal-model hypothesis; the witness is not a native implementation |
| `law_not_universal` | The law cannot be derived for every unconstrained `FloatModel` |

The reference recursively traverses every operation, block and region. It changes only
eligible powers and retains all other operations, attributes, properties, SSA operands,
result bindings, block arguments, successors and module headers. Lexical scopes account
for captured constants and block-argument shadowing. Complete output comparison rejects
unrelated changes as well as corrupt individual rewrites. It does not merely compare
the number of powers and multiplies.

**Composition limit:** there is no theorem in this packet deriving original whole-program
source/target denotation equality from the numerical law alone. The checked complete
output correspondence and local semantic theorems are separate results. Lifting the
local typed theorem through source environments, region evaluation and calls, and proving
whole-capture definedness, are additional proof obligations for that stronger claim.
Do not describe `complete_output_sound` as original-source equivalence: its right-hand
side is the multiplication reference.

Each receipt binds the audited source/import closure, compiled Lean modules, toolchain,
SPEC_LOCK, launcher, exporter and actual before/after files and exports. Its report contains
the elaborated statements of all 11 listed theorems. The imported project environment
audit checked **1,348 theorems with zero violations**. The permitted axioms are only
`propext`, `Classical.choice` and `Quot.sound`.

## Actual boundary evidence

| Capture | Literal-two rewrites | Format / flags | Complete output | Replay |
| --- | ---: | --- | --- | --- |
| SmolVLA | 66 | f32 / none | match | fresh, reproduced |
| TinyLlama | 45 | f32 / none | match | fresh, reproduced |

All 111 sites lie in nested regions (seven-component recorded paths). Native MLIR parsing,
export, Lean printing and native re-export round-trip identically on both complete sides.
An independent Python source census agrees with the Lean paths. SmolVLA also has 56
dynamic-exponent f32 powers and one dynamic-exponent f64 power; these remain unchanged.
The actual captures contain no rewritten f64 site. f64 coverage uses synthetic valid MLIR,
the actual pass, the Lean checker and native execution.

Receipts: [SmolVLA](../SY_model_smolvla/receipt.json),
[TinyLlama](../SY_model_tiny_llama/receipt.json). The exact captured stage metadata and
before/after hashes are bound there. [Run summary](../summary.json) binds both receipts,
replay results and native reports. Historical source paths happen to be under the W8A8
capture tree; they are read-only inputs to this separate boundary check.

## Native numerical evidence and counterexamples

The retained run executes 4,116 f32 samples and 4,115 f64 samples. They include raw random
bit patterns, finite values around one, positive/negative zeros, extreme/subnormal values,
infinities, and quiet/signaling NaNs with payloads. The seed is 9203; the exact input list
and returned bits are retained. Raw bits cross the native wrapper rather than Python
floating conversions. NaN payloads remain visible even where the target-vs-C oracle check
uses the weaker NaN-class relation.

Two paths are kept distinct: actual MLIR before/after execution, and forced indirect libm
calls compared with C multiplication (`FE_TONEAREST`, builtins and fast-math disabled).
The actual operation extracted from each capture is executed as a scalar probe, alongside
independent f32/f64 cases. This is not native execution of a complete model capture.
Compiler, library and exporter identities, lowering pipelines, source snapshots and native
results are retained beside each `differential.json`.

Both paths produce these ordinary finite counterexamples:

| Format | Input bits | Power result bits | Multiplication result bits |
| --- | --- | --- | --- |
| f32 | `0x3f10d742` | `0x3ea3e5bf` | `0x3ea3e5be` |
| f32 | `0x3f2fa800` | `0x3ef10e3d` | `0x3ef10e3c` |
| f64 | `0x3fefb55750829a12` | `0x3fef6b5cd0aa0e82` | `0x3fef6b5cd0aa0e83` |
| f64 | `0x3fed510f0375636c` | `0x3feadbb88a4d51ba` | `0x3feadbb88a4d51b9` |

Each row differs by one ULP. The retained f32 sample has seven mismatching observations
at five distinct inputs (two inputs appear twice); f64 has five at five distinct inputs.
The same f32 discrepancy occurs in both extracted capture probes. This refutes a universal
bitwise identity for the tested native implementation, including on finite nonzero inputs.
It does not establish a universal one-ULP error bound. Restricting only NaNs, infinities or
signed zero would not remove these counterexamples.

The native tests pass when measurements are recorded faithfully and the emitted
multiplication agrees with the independent multiplication oracle; a passing test is
**not** an assertion that all before/after bits match. `contract` and `none` are both
tested for f32/f64. Rejection tests cover wrong exponents, mixed/wrong operands, altered
and unsupported flags, omitted rewrites, corruption of the last nested site, unrelated
return changes, and a forged green report that must fail recomputation on replay.

Qualification: **51 fast tests and two slow retained-boundary replay tests passed**, with
zero skips. JUnit records: [fast](../validation/fast.xml), [slow](../validation/slow.xml).

## Replay and regeneration

From the repository root, regenerate the checker receipts, native evidence and qualification
tests using the exact retained command (the current machine paths are recorded):

```sh
.venv/bin/python - <<'PY'
import json, subprocess
from merlin.common.paths import artifacts_dir
base = artifacts_dir() / "verification/merlin-passes/frontend/scalar-square"
subprocess.run(json.loads((base / "run-command.json").read_text()), check=True)
PY
```

To replay just a receipt, use a fresh work directory:

```sh
.venv/bin/python -m merlin.verify.lean.scalar_square replay \
  out/artifacts/verification/merlin-passes/frontend/scalar-square/SY_model_smolvla/receipt.json \
  out/artifacts/verification/merlin-passes/frontend/scalar-square/SY_model_smolvla/review-replay-001 \
  --compiler-python /home/jeremy/Projects/model2MLIR/.venv/bin/python
```

Replay rechecks bindings, native round trips and all executable Lean report fields. It
refuses stale inputs and detects changed reports even when their digest is rewritten.
The standalone driver builds only its own Lean module and uses its own launcher. It does
not relink the shared `merlin-lean` executable or modify shared CLI, Main, lakefile, frozen
semantics, W8A8 or boolean-repair code/evidence.

## Human review decisions

1. Review the rule, flags, exceptional values, effect exclusions and proof scope independently.
   The production pass’s prose currently describes a multiplication identity more strongly
   than the native evidence supports; the pass was left unchanged.
2. Reject an unconditional bitwise-equivalence claim for the tested implementation. If the
   intended specification is a multiplication-based numerical policy, review and name that
   policy explicitly. If a different backend’s power implementation is to satisfy `Law`,
   provide its own justification rather than approving the refuted host-law instance.
3. Decide whether exact NaN bits are really the desired observation relation. Any weaker
   relation needs its own statement and proof; finite counterexamples still remain.
4. Require a separate composition proof and definedness evidence before claiming original
   whole-capture source/target refinement. The current output/reference equality does not
   supply either obligation.
5. Only after the final claim and assumptions are reviewed, coordinate theorem pins and a
   SPEC_LOCK update with other workstreams. Regenerate affected receipts then; no such
   approval or update is implied by this packet.

An accurate present-tense claim is: “A Lean checker validated complete output correspondence
to a scalar-multiplication reference on 111 actual rewritten sites, with audited local
semantic soundness conditional on an explicit, unapproved numerical law. Native tests
exposed counterexamples to exact power/multiplication equality.”
