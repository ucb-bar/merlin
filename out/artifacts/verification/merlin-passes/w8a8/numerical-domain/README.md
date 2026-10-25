# W8A8 numerical-domain candidate

Worker implementation, 2026-10-22. Scope: numerical source execution needed by Merlin's
`lower_contraction_int8`, between model2MLIR and the OOT compiler. LLVM pass verification,
frontend captured-call correspondence and `lower_quant_ext` lowering correspondence remain
outside this worker's assignment.

**This is a separately named conditional candidate, not full W8A8 verification.** The original
finite-input domain, original bounded checker, six qualified source/test files and historical
runtime capsule are preserved. Staging, commits, integration and pushes remain on hold.

## Exact candidate domain and premises

Every f32 input element must be finite and nonzero, including both signs of nonzero values and
subnormals. Other input types keep the original typed/initialized requirements and declared
per-function integer intervals. Infinities are allowed as intermediate values. Positive and
negative zero f32 inputs are outside this candidate; that is a stronger restriction than the
original domain and is not silently substituted for it.

`PolicyW8A8NumericalInhabited.check_sound` proves existence of inputs for each accepted witness
check. It constructs checked scalar seeds and full tensors without enumerating large shapes.
Both complete qualification and entry-prefix certification require this nonvacuity check.
Input-domain inhabitation does **not** prove that a concrete floating interpretation satisfies
all numerical premises.

The original seven source and three dynamic primitive laws are unchanged. The new
`W8A8ArithmeticModel.Agreement` additionally names exact default-key arithmetic interpretations:

- f32 multiplication, addition, division, subtraction and reciprocal square root, restricted
  to non-NaN results of the concrete computation;
- integer-to-f32 conversion;
- the actual source reciprocal's f32-to-f64 widening, f64 division and f32 narrowing, restricted
  to admissible source scales. Its result is recomputed and checked finite/nonzero, never assumed.

The concrete functions use Lean 4.35's pure `UnpackedFloat` model. Reciprocal square root is
represented as rounded square root followed by rounded division of one. **Universal agreement
with a selected MLIR/native implementation, reciprocal-square-root fidelity and a joint model
instance for all premises remain unproved.** No new axiom or admitted Spec was added.

Native development comparisons showed differing NaN sign bits for invalid operations. The
arithmetic agreement was narrowed so these unused NaN-result cases do not require exact NaN
bits. Raw model/native differences are retained; classification comparisons are explicitly
distinguished from exact non-NaN bit comparisons. This is not a universal native-model proof.

## Implemented proof and checker path

| Component | What is proved/checked |
| --- | --- |
| `Rules/W8A8IEEE32.lean` | Concrete classification lemmas for finite operands, nonnegative squaring/addition/division and rsqrt; overflow and `rsqrt(+0) = +infinity` are permitted |
| `Rules/W8A8ArithmeticModel.lean` | Explicit arithmetic mapping, actual source reciprocal, and nonpoison source quantization of non-NaN inputs under the unchanged source laws |
| `Rules/W8A8SquareBody.lean` | Execution of the actual recognized square body, with or without the captured unused constant |
| `Rules/W8A8DequantNumerics.lean` | Source dequantization finite-output checking over all 256 byte patterns, with actual scale/zero bits and checked non-NaN arithmetic prerequisites |
| `Check/PolicyW8A8Numerical{Facts,Implication,Scalar,Ops}.lean` | Typed, initialized numerical facts, proved implication, exact operation keys and scalar transfer rules |
| `...Numerical{Square,Extract,Reduce,Quantize,Dequant}.lean` | Complete scalar-body recognition, bound-checked extraction, named reduction, source quantizer and source dequantizer integration |
| `...Numerical{Defined,Called,Qualified,Chain}.lean` | Actual function/helper execution and complete policy-boundary/suffix checking on the same candidate domain |
| `...NumericalInhabited.lean` | Executable proof of input-domain nonvacuity |
| `...NumericalPrefix.lean` | Actual entry binding and ordered prefix execution; initialized values and numerical facts in the final SSA environment; no statement about later refused operations |
| `Driver/W8A8Numerical.lean`, `W8A8NumericalMain.lean` | Standalone driver, separate complete-boundary and prefix results, finite dequantizer inventory and explicit unresolved-premise flags |
| `src/merlin/verify/lean/numerical_{evidence,policy}.py` | Strict evidence validation, authoritative exports, byte-bound receipts, fresh theorem audits and recomputed replay |

The reduction rule derives the operand split, dimensions, indexing maps, iterator order and
body from actual IR/facts. It proves agreement with the frozen named-reduction semantics and
uses the existing ordered generic-loop theorem. It does not reassociate floating sums. Its
invariant forgets finite initializer facts that do not survive arbitrary floating addition.
Existing finite-only structured contracts retain their independently checked fallback analysis.

The source dequantization rule proves numerical facts for the source interpretation; it does
not prove the production dequantization lowering. The other worker retains that task and its
`QuantExtLowering*` files.

## Actual retained development boundary

The preserved eight-token TinyLlama development boundary's original before/after export bytes
were rechecked. The numerical candidate advances from the earlier 902-operation refusal at
source quantizer **1256** to a **1,060-operation** source prefix. Its policy-reference prefix
advances to 1,087 operations. Both prefixes include initialized source quantizer 1256 and
both next refuse quantizer **1533**, consuming result 1528 after the attention computation.
The accepted source prefix also certifies **158 finite source dequantizers**, derived from
actual scale/zero facts and the complete byte table.

This is development evidence on the stronger domain. It does not prove the original domain,
bind the selected EL4 model bundle, establish all actual archive inputs satisfy the restriction,
or qualify the whole model. The complete structural match passes; complete numerical
qualification remains refused. Retained dequantized weights have literal scales and byte
storage; the next work should use their checked bounds rather than invent arbitrary weight
magnitudes or assume finite floating contraction products.

The development route is under `retained-development/`; it predates the final arithmetic
premise refinement and is preserved as history. Fresh qualification uses a new destination
and identity. The old unrelated JSON decode failure was not isolated and is not marked fixed.

The final receipt is `retained-qualified-v1/receipt.json`; its independently regenerated replay
is `retained-replay-v1/receipt.json`. Both use runtime identity
`1f31ba11458ce608e87c7ff3a729e5609e8468c0a8aca6cae63eb81527c2f675` and the original export bytes.

### Further domain insufficiency: rotary input 201

`rotary-domain-diagnosis-v3/` retains **38 exact original producer operations** for positional
angles and rotary sine/cosine, changing only function input/output plumbing and module provenance
annotations. Original input 201 is the 32-element f32 frequency tensor. The original position
generation is preserved. At position two, a uniform frequency of +1 produces angle 2 and finite
sine/cosine. A uniform maximum finite f32 frequency produces +infinity and NaNs (`0xffc00000`).
Both supplied frequency tensors have finite, nonzero elements, so the candidate restriction
does not exclude this local witness. The manifest binds the original export, retained operations,
probe, native pipeline and exact outputs. Earlier v1/v2 directories are failed probe-construction
attempts and are not numerical evidence.

This is an actual source-fragment counterexample to assuming all rotary intermediates non-NaN
on the candidate domain. It is not a full-model execution, an observed production-weight failure,
or a counterexample to the W8A8 transformation. Do not execute its eventual poison conversion
natively or claim quantizer 1533 fully exercised by this fragment. Further progress requires
justified frequency bounds or concrete archive/capture binding, as well as appropriate
transcendental and attention proofs. Merely adding an unconditional non-NaN fact would be unsound.

## Qualification and reproduction

Final qualification: **292 tests passed**, zero failures/errors/skips, in 504.73 seconds.
This comprises 239 affected existing W8A8 regressions, 51 new numerical-candidate tests and two
preserved numerical-domain regressions. The full Lean guard audits **9,663 theorems with zero
violations** and passes the existing frozen-file/statement lock. Fifteen elaborated claim/premise
statements are retained separately. No new lock or assumption admission was performed.

All **1,230 frozen records** and all **752 historical runtime records** remain unchanged after
tests and receipt replay. The original six qualified source/test files also retain their hashes.
`freeze-v2.json` is the qualified freeze; v1 was superseded by destination-alias protection before
qualification began. Thirteen curated positive case receipts replay: nine actual changed-pass
boundaries (including one suffix) and four explicitly unchanged reduction-rule probes. The actual
TinyLlama refusal receipt replays separately. Protocol baseline duplicates and deliberately
tampered/unmeasured fixtures are not promoted into this retained positive roster.

Native evidence includes non-NaN arithmetic bit comparisons, explicitly recorded NaN sign
differences, every byte value for a retained dequantization scale, i64 conversion endpoints and
finite/nonzero RMSnorm input overflow followed by defined source clipping. Universal primitive
fidelity is not inferred from these samples. Native evidence and a kernel example log are under
`native-evidence-v1/`; the rotary diagnosis above is additional native evidence outside the test count.

Ruff, artifact layout and package/document freshness checks pass. The structure gate retains
only the unrelated `zephyr_model.py` 3,557 > 3,537 line-ceiling failure. Development failures and
freshness refusals during active implementation are preserved but not counted as final passes.
Historic test totals overlap and must not be added to this batch.

`qualification-v1.json` records exact identities, counts and remaining premises;
`implementation-v1.patch` contains only the 30 owned additive source/test files, without staging.

New private build: `out/build/lean-w8a8-numerical-domain/repo`. Preserve the old
`out/build/lean-w8a8-source-integration/repo` capsule. Use the environment in the adjacent W8A8
handoff, replacing `MERLIN_REPO_ROOT` with the new private build. Set
`MERLIN_LEAN_REQUIRED=1` and retain successful test artifacts.

```sh
PYTHONPATH=src .venv/bin/python -m merlin.verify.lean.numerical_policy \
  /path/to/actual/w8a8/stage /path/to/optional/suffix/stage \
  --destination /path/to/new/receipt-directory \
  --input-contracts /path/to/per-function-contracts.json \
  --m2m-python "$MERLIN_M2M_PYTHON" --scope captured
```

The route always returns draft `abstracted`, exit 1, when execution/evidence are valid. Execution
failures produce `unmeasured`, exit 2. Full conditional acceptance, prefix acceptance and
independent review are separate fields. Replay uses
`numerical_policy.replay(receipt_path, new_destination, m2m_python=...)` and regenerates exports,
audits and checks. Forged statuses, changed subjects, missing runtime inventory and stale bytes
cannot qualify their own replay.

## Next work and claim limits

1. Resume at result 1533: bind/justify input-201 frequency bounds before asserting rotary
   finiteness, then propagate justified bounds through source-derived contractions and
   attention/softmax. Transcendental interpretation and nonzero denominators need explicit
   proofs/contracts. The current broad input restriction does not rule out the retained witness.
2. Justify the concrete arithmetic interpretation and original ten laws jointly. Preserve
   the observed NaN-bit differences and review native rsqrt behavior on the declared domain.
3. Obtain independent domain/policy admission and exact selected-capture/version/recipe binding.
4. Extend remaining production variants, compose every required changed boundary and qualify
   reviewed integration centrally. Shared Main/CLI, generated documentation indices and locks
   remain the coordinator's responsibility.

`whole_pass_verified`, `policy_reviewed`, `primitive_model_laws_verified` and
`concrete_context_inhabited` remain false. F11/F12/F13 and all prior numerical counterexamples
are preserved. No production pass, frozen Core/Sem/Typing/Types, Spec, Claims, Assumptions, pin,
lock, frontend, Qinner or QuantExtLowering source was changed.
